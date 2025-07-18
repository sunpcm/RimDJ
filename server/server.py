import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Spark-TTS')))
import asyncio
import queue
import threading
import time
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import config
import llm_service
import tts_service
import mp3_service
import bgm_service
BGM_SERVICE_TYPE = "librosa"
from semantic_queue import SemanticAwareQueue
from base_service import AsyncServiceBase, safe_get_item, print_processing_info

# Global state for broadcasting
class AudioStreamState:
    def __init__(self):
        self.clients = {}
        self.lock = threading.Lock()
        self.sample_rate = config.SAMPLE_RATE
        self.keepalive_data = self._generate_keepalive_data()
        self.keepalive_task = None
        self.keepalive_active = False
    
    def _generate_keepalive_data(self):
        """生成保活数据 - 使用低音量白噪音"""
        try:
            import numpy as np
            import lameenc
            
            # 生成100ms的低音量白噪音
            duration_ms = 100
            samples = int(config.SAMPLE_RATE * duration_ms / 1000)
            
            # 生成白噪音并设置很低的音量（-60dB左右）
            noise = np.random.normal(0, 1, samples)
            # 将音量降低到几乎听不见的程度
            noise = noise * 0.001  # 约-60dB
            
            # 转换为int16格式
            noise_int16 = (noise * 32767).astype(np.int16)
            
            # 编码为MP3
            encoder = lameenc.Encoder()
            encoder.set_bit_rate(128)
            encoder.set_in_sample_rate(config.SAMPLE_RATE)
            encoder.set_channels(1)
            encoder.set_quality(2)
            
            mp3_data = encoder.encode(noise_int16.tobytes())
            mp3_data += encoder.flush()
            
            return mp3_data
            
        except Exception as e:
            print(f"Error generating keepalive noise: {e}")
            # 如果生成失败，回退到空字节
            return b''

    def add_client(self, client_id, audio_queue):
        with self.lock:
            self.clients[client_id] = audio_queue
            return True

    def remove_client(self, client_id):
        with self.lock:
            if client_id in self.clients:
                try:
                    # Try to put a sentinel value to signal stream end
                    self.clients[client_id].put(None, timeout=0.1)
                except:
                    pass  # Queue might be full or closed
                del self.clients[client_id]
                print(f"Client {client_id} removed from broadcast list")

    def broadcast_audio(self, mp3_data):
        with self.lock:
            disconnected_clients = []
            successful_broadcasts = 0
            
            for client_id, q in list(self.clients.items()):
                try:
                    # Try to put data with a reasonable timeout
                    q.put(mp3_data, timeout=0.5)  # 增加超时时间到0.5秒
                    successful_broadcasts += 1
                except queue.Full:
                    # Don't immediately remove client, just skip this chunk
                    # Client might recover on next chunk
                    print(f"Client {client_id} queue full, skipping chunk")
                    continue
                except Exception as e:
                    print(f"Error broadcasting to client {client_id}: {e}")
                    disconnected_clients.append(client_id)
            
            # Only remove clients that had actual errors, not full queues
            for client_id in disconnected_clients:
                self.remove_client(client_id)
                
            # If no successful broadcasts, we might want to slow down
            if successful_broadcasts == 0 and len(self.clients) > 0:
                print(f"No successful broadcasts to {len(self.clients)} clients")
    
    def broadcast_keepalive(self):
        """广播保活数据，不通过队列系统"""
        with self.lock:
            if len(self.clients) == 0:
                return
            
            disconnected_clients = []
            for client_id, q in list(self.clients.items()):
                try:
                    # 发送空字节作为keep-alive，不会被浏览器当作音频处理
                    q.put(self.keepalive_data, timeout=0.1)
                except queue.Full:
                    # keep-alive数据可以跳过，不是关键数据
                    continue
                except Exception as e:
                    print(f"Error broadcasting keepalive to client {client_id}: {e}")
                    disconnected_clients.append(client_id)
            
            for client_id in disconnected_clients:
                self.remove_client(client_id)
    
    def start_keepalive(self):
        """启动连接保活任务"""
        if self.keepalive_task is None or self.keepalive_task.done():
            self.keepalive_active = True
            self.keepalive_task = asyncio.create_task(self._keepalive_loop())
    
    def stop_keepalive(self):
        """停止连接保活任务"""
        self.keepalive_active = False
        if self.keepalive_task and not self.keepalive_task.done():
            self.keepalive_task.cancel()
    
    async def _keepalive_loop(self):
        """连接保活循环"""
        try:
            while self.keepalive_active:
                # 检查是否有客户端连接
                if len(self.clients) > 0:
                    self.broadcast_keepalive()
                await asyncio.sleep(2.0)  # 每2秒发送一次keep-alive，降低频率
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Keepalive error: {e}")

# 语义感知队列，支持广播稿完整性
text_queue = queue.Queue(maxsize=50)  # 保持普通队列用于文本输入
broadcast_script_queue = SemanticAwareQueue(maxsize=20, name="BroadcastScript")

# 新的WAV级别音频处理队列 - 优化队列大小以避免BGM卡顿
tts_wav_queue = SemanticAwareQueue(maxsize=config.WAV_QUEUE_SIZE, name="TTS_WAV")  # TTS生成的WAV音频队列
bgm_wav_queue = SemanticAwareQueue(maxsize=config.BGM_WAV_QUEUE_SIZE, name="BGM_WAV")  # BGM的WAV音频队列
mixed_wav_queue = SemanticAwareQueue(maxsize=config.MIXED_WAV_QUEUE_SIZE, name="MixedWAV")  # 混合后的WAV队列
broadcast_mp3_queue = SemanticAwareQueue(maxsize=config.MP3_QUEUE_SIZE, name="BroadcastMP3")  # 最终广播的MP3队列

class BroadcasterService(AsyncServiceBase):
    """音频广播服务"""
    
    def __init__(self, audio_state, broadcast_mp3_queue):
        super().__init__("Broadcaster")
        self.audio_state = audio_state
        self.mp3_stream_queue = broadcast_mp3_queue  # 使用最终广播的MP3队列
        self.broadcast_cache = {}  # 缓存广播稿的所有块
        self.last_broadcast_time = 0  # 记录最后一次广播时间
        self.last_cleanup_time = 0  # 记录最后一次清理时间
        self.cleanup_interval = 5.0  # 清理间隔（秒）
    
    async def stop(self):
        """停止服务"""
        # 强制清理所有缓存的广播稿
        self._cleanup_expired_broadcasts(max_age_seconds=0, force=True)
        await super().stop()
    
    async def process_item(self) -> bool:
        """处理MP3流队列中的项目"""
        # 获取MP3项目（无论是否有客户端连接都要消费队列以防止积压）
        mp3_item = safe_get_item(self.mp3_stream_queue, timeout=0.1)
        if not mp3_item:
            # 如果没有新的音频数据，检查是否需要启动连接保活
            await self._check_keepalive()
            return False
        
        # 检查是否有客户端连接
        if len(self.audio_state.clients) == 0:
            # 没有客户端时仍然消费队列但不广播，防止队列积压
            # print_processing_info("Broadcaster", mp3_item.broadcast_id, f"discarded chunk {mp3_item.sequence_number} (no clients)")
            # 仍需要处理广播缓存以防止内存泄漏
            broadcast_id = mp3_item.broadcast_id
            if broadcast_id not in self.broadcast_cache:
                self.broadcast_cache[broadcast_id] = {
                    'items': {},
                    'total_items': mp3_item.total_items,
                    'priority': mp3_item.priority,
                    'timestamp': time.time()
                }
            
            self.broadcast_cache[broadcast_id]['items'][mp3_item.sequence_number] = mp3_item.content
            
            # 检查广播稿是否完整并清理
            cache_entry = self.broadcast_cache[broadcast_id]
            if mp3_item.is_last or (
                cache_entry['total_items'] and 
                len(cache_entry['items']) == cache_entry['total_items']
            ):
                del self.broadcast_cache[broadcast_id]
            
            # 清理过期的不完整广播稿
            self._cleanup_expired_broadcasts()
            return True
        
        # 有新的音频数据时，停止连接保活
        self.audio_state.stop_keepalive()
        
        broadcast_id = mp3_item.broadcast_id
        
        # 检查是否为BGM连续流（不需要缓存）
        is_bgm_stream = (broadcast_id.startswith('bgm_') and 
                        mp3_item.total_items is None and 
                        not mp3_item.is_last)
        
        if not is_bgm_stream:
            # 只对非BGM流进行缓存管理
            if broadcast_id not in self.broadcast_cache:
                self.broadcast_cache[broadcast_id] = {
                    'items': {},
                    'total_items': mp3_item.total_items,
                    'priority': mp3_item.priority,
                    'timestamp': time.time()
                }
            
            self.broadcast_cache[broadcast_id]['items'][mp3_item.sequence_number] = mp3_item.content
        
        # 立即广播收到的MP3块（仅在有客户端时）
        if mp3_item.content and len(mp3_item.content) > 0:
            self.audio_state.broadcast_audio(mp3_item.content)
            self.last_broadcast_time = time.time()  # 更新最后广播时间
            
            # 为BGM流提供简化的日志
            if is_bgm_stream:
                if mp3_item.sequence_number % 200 == 1:  # 每200个块打印一次
                    print_processing_info("Broadcaster", broadcast_id, f"streaming BGM chunk {mp3_item.sequence_number}")
            # else:
                # print_processing_info("Broadcaster", broadcast_id, f"broadcasted chunk {mp3_item.sequence_number}")
        
        # 检查广播稿是否完整（仅对非BGM流）
        if not is_bgm_stream and broadcast_id in self.broadcast_cache:
            cache_entry = self.broadcast_cache[broadcast_id]
            if mp3_item.is_last or (
                cache_entry['total_items'] and 
                len(cache_entry['items']) == cache_entry['total_items']
            ):
                # print_processing_info("Broadcaster", broadcast_id, f"completed ({len(cache_entry['items'])} chunks)")
                del self.broadcast_cache[broadcast_id]
                # 广播完成后，启动连接保活任务
                self.audio_state.start_keepalive()
        
        # 清理过期的不完整广播稿
        self._cleanup_expired_broadcasts()
        
        return True
    
    async def _check_keepalive(self):
        """检查是否需要启动连接保活任务"""
        current_time = time.time()
        # 如果最后一次广播超过5秒，且有客户端连接，启动连接保活
        if (current_time - self.last_broadcast_time > 5.0 and 
            len(self.audio_state.clients) > 0 and 
            not self.audio_state.keepalive_active):
            self.audio_state.start_keepalive()
    
    def _cleanup_expired_broadcasts(self, max_age_seconds: float = 30, force: bool = False):
        """清理过期的不完整广播稿
        
        Args:
            max_age_seconds: 广播稿过期时间（秒）
            force: 是否强制清理，忽略清理间隔
        """
        current_time = time.time()
        
        # 检查是否需要清理（基于时间间隔）
        if not force and (current_time - self.last_cleanup_time) < self.cleanup_interval:
            return
        
        expired_broadcasts = [
            bid for bid, cache_entry in self.broadcast_cache.items()
            if current_time - cache_entry.get('timestamp', current_time) > max_age_seconds
        ]
        
        if expired_broadcasts:
            for bid in expired_broadcasts:
                del self.broadcast_cache[bid]
            
            # 批量报告清理结果，减少日志噪音
            if len(expired_broadcasts) == 1:
                print_processing_info("Broadcaster", expired_broadcasts[0], "expired broadcast cleaned up")
            else:
                print_processing_info("Broadcaster", "batch_cleanup", f"cleaned up {len(expired_broadcasts)} expired broadcasts")
        
        # 更新最后清理时间
        self.last_cleanup_time = current_time


def start_broadcaster(state, broadcast_mp3_queue):
    """启动广播服务"""
    service = BroadcasterService(state, broadcast_mp3_queue)
    return service.start()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting lifespan")
    state = AudioStreamState()
    app.state.audio_state = state

    # Start services
    print("Starting LLM service")
    llm_service.start_llm_service(text_queue, broadcast_script_queue)
    print("Starting TTS service")
    tts_service.start_tts_service(broadcast_script_queue, tts_wav_queue)
    
    # 根据BGM_ENABLED配置决定音频处理流程
    if config.BGM_ENABLED:
        print("BGM enabled - starting BGM and Audio Mixer services")
        # Start BGM service
        print("Starting BGM service")
        music_dir = os.path.join(os.path.dirname(__file__), "..", "music")
        bgm_service.start_bgm_service(bgm_wav_queue, music_dir)
        app.state.bgm_controller = bgm_service.get_bgm_controller()
        
        # Start Audio Mixer service
        print("Starting Audio Mixer service")
        import audio_mixer_service
        audio_mixer_service.start_audio_mixer_service(tts_wav_queue, bgm_wav_queue, mixed_wav_queue)
        
        # Start MP3 service with mixed audio
        print("Starting MP3 service (with BGM mixing)")
        mp3_service.start_mp3_service(mixed_wav_queue, broadcast_mp3_queue)
        
        # Auto-start BGM playback
        if app.state.bgm_controller.start_bgm():
            print("BGM auto-started successfully")
        else:
            print("BGM auto-start failed - no music files or already playing")
    else:
        print("BGM disabled - TTS audio will bypass mixer")
        # 创建一个空的BGM控制器以避免错误
        app.state.bgm_controller = None
        
        # Start MP3 service directly with TTS audio (bypass mixer)
        print("Starting MP3 service (TTS only, no BGM)")
        mp3_service.start_mp3_service(tts_wav_queue, broadcast_mp3_queue)

    # Start broadcaster
    print("Starting broadcaster thread")
    start_broadcaster(state, broadcast_mp3_queue)

    yield
    
    # Cleanup on shutdown
    print("Shutting down services...")
    # Stop keepalive task
    state.stop_keepalive()
    # Signal all clients to disconnect
    with state.lock:
        for client_id in list(state.clients.keys()):
            state.remove_client(client_id)
    print("All clients disconnected")
    print("Shutdown complete")

app = FastAPI(lifespan=lifespan)

# 挂载静态文件
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "statics")), name="static")

# 默认路由重定向到静态页面
@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(os.path.dirname(__file__), "statics", "index.html"))

@app.get("/health")
async def health_check():
    bgm_status = "disabled" if not config.BGM_ENABLED else "stopped"
    if config.BGM_ENABLED and hasattr(app.state, 'bgm_controller') and app.state.bgm_controller and app.state.bgm_controller.bgm_service:
        bgm_status = "playing" if app.state.bgm_controller.bgm_service.is_playing else "stopped"
    
    return {
        "status": "healthy",
        "active_clients": len(app.state.audio_state.clients),
        "bgm_status": bgm_status,
        "bgm_service_type": BGM_SERVICE_TYPE,
        "text_queue_size": text_queue.qsize(),
        "broadcast_queue": {
            "size": broadcast_script_queue.qsize(),
            "pressure_level": broadcast_script_queue.get_pressure_level(),
            "active_broadcasts": len(broadcast_script_queue.get_active_broadcasts()),
            "semantic_status": broadcast_script_queue.get_semantic_status()
        },
        "tts_wav_queue": {
            "size": tts_wav_queue.qsize(),
            "pressure_level": tts_wav_queue.get_pressure_level(),
            "active_broadcasts": len(tts_wav_queue.get_active_broadcasts()),
            "semantic_status": tts_wav_queue.get_semantic_status()
        },
        "bgm_wav_queue": {
            "size": bgm_wav_queue.qsize(),
            "pressure_level": bgm_wav_queue.get_pressure_level(),
            "active_broadcasts": len(bgm_wav_queue.get_active_broadcasts()),
            "semantic_status": bgm_wav_queue.get_semantic_status()
        },
        "mixed_wav_queue": {
            "size": mixed_wav_queue.qsize(),
            "pressure_level": mixed_wav_queue.get_pressure_level(),
            "active_broadcasts": len(mixed_wav_queue.get_active_broadcasts()),
            "semantic_status": mixed_wav_queue.get_semantic_status()
        },
        "broadcast_mp3_queue": {
            "size": broadcast_mp3_queue.qsize(),
            "pressure_level": broadcast_mp3_queue.get_pressure_level(),
            "active_broadcasts": len(broadcast_mp3_queue.get_active_broadcasts()),
            "semantic_status": broadcast_mp3_queue.get_semantic_status()
        }
    }

@app.post("/text")
async def submit_text(request: Request):
    try:
        data = await request.json()
        text = data.get('text')
        if text:
            text_queue.put(text, timeout=0.1)
            return {"status": "success"}
        return {"status": "error", "message": "No text provided"}
    except queue.Full:
        return {"status": "error", "message": "Server queue full, try again later"}
    except Exception as e:
        return {"status": "error", "message": "Internal server error"}

@app.post("/bgm/start")
async def start_bgm():
    """启动BGM播放"""
    try:
        if not config.BGM_ENABLED:
            return {"status": "error", "message": "BGM is disabled in configuration"}
        if hasattr(app.state, 'bgm_controller') and app.state.bgm_controller:
            success = app.state.bgm_controller.start_bgm()
            if success:
                return {"status": "success", "message": "BGM started"}
            else:
                return {"status": "error", "message": "BGM already playing or no music files"}
        return {"status": "error", "message": "BGM service not available"}
    except Exception as e:
        print(f"Error starting BGM: {e}")
        return {"status": "error", "message": "Failed to start BGM"}

@app.post("/bgm/stop")
async def stop_bgm():
    """停止BGM播放"""
    try:
        if not config.BGM_ENABLED:
            return {"status": "error", "message": "BGM is disabled in configuration"}
        if hasattr(app.state, 'bgm_controller') and app.state.bgm_controller:
            app.state.bgm_controller.stop_bgm()
            return {"status": "success", "message": "BGM stopped"}
        return {"status": "error", "message": "BGM service not available"}
    except Exception as e:
        print(f"Error stopping BGM: {e}")
        return {"status": "error", "message": "Failed to stop BGM"}

@app.post("/bgm/volume")
async def set_bgm_volume(request: Request):
    """设置BGM音量级别"""
    try:
        if not config.BGM_ENABLED:
            return {"status": "error", "message": "BGM is disabled in configuration"}
        
        data = await request.json()
        normal_volume = data.get('normal', 0.3)
        ducked_volume = data.get('ducked', 0.1)
        
        if not (0.0 <= normal_volume <= 1.0) or not (0.0 <= ducked_volume <= 1.0):
            return {"status": "error", "message": "Volume must be between 0.0 and 1.0"}
        
        if hasattr(app.state, 'bgm_controller') and app.state.bgm_controller:
            app.state.bgm_controller.set_volume_levels(normal_volume, ducked_volume)
            return {
                "status": "success", 
                "message": "Volume levels updated",
                "normal_volume": normal_volume,
                "ducked_volume": ducked_volume
            }
        return {"status": "error", "message": "BGM service not available"}
    except Exception as e:
        print(f"Error setting BGM volume: {e}")
        return {"status": "error", "message": "Failed to set volume"}

@app.get("/bgm/status")
async def get_bgm_status():
    """获取BGM状态"""
    try:
        if not config.BGM_ENABLED:
            return {
                "status": "success",
                "bgm_enabled": False,
                "message": "BGM is disabled in configuration"
            }
        
        if hasattr(app.state, 'bgm_controller') and app.state.bgm_controller and app.state.bgm_controller.bgm_service:
            bgm_service = app.state.bgm_controller.bgm_service
            return {
                "status": "success",
                "bgm_enabled": True,
                "is_playing": bgm_service.is_playing,
                "is_ducked": bgm_service.is_ducked,
                "current_volume": bgm_service.current_volume,
                "ducked_volume": bgm_service.ducked_volume,
                "music_files_count": len(bgm_service.music_files),
                "current_track_index": bgm_service.track_index
            }
        return {"status": "error", "message": "BGM service not available"}
    except Exception as e:
        print(f"Error getting BGM status: {e}")
        return {"status": "error", "message": "Failed to get BGM status"}

@app.get("/test_audio")
async def test_audio():
    """返回测试音频流"""
    import io
    import wave
    import numpy as np
    
    # 生成1秒的440Hz正弦波测试音频
    sample_rate = 16000
    duration = 1.0
    frequency = 440
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave_data = np.sin(frequency * 2 * np.pi * t)
    
    # 转换为立体声int16格式
    audio_data = (wave_data * 32767).astype(np.int16)
    stereo_data = np.column_stack((audio_data, audio_data)).flatten()
    
    # 使用lameenc编码为MP3
    try:
        import lameenc
        encoder = lameenc.Encoder()
        encoder.set_bit_rate(128)
        encoder.set_in_sample_rate(sample_rate)
        encoder.set_channels(2)
        encoder.set_quality(2)
        
        mp3_data = encoder.encode(stereo_data.tobytes())
        mp3_data += encoder.flush()
        
        return StreamingResponse(
            io.BytesIO(mp3_data),
            media_type="audio/mpeg",
            headers={
                "Cache-Control": "no-cache",
                "Content-Length": str(len(mp3_data))
            }
        )
    except Exception as e:
        print(f"Test audio generation error: {e}")
        return {"error": "Failed to generate test audio"}

@app.get("/audio_stream")
async def audio_stream(request: Request):
    client_id = str(uuid.uuid4())
    # 大幅增加客户端队列大小以避免BGM卡顿
    client_queue = queue.Queue(maxsize=config.CLIENT_QUEUE_SIZE)
    
    print(f"New client {client_id} connecting from {request.client.host}")
    
    if not app.state.audio_state.add_client(client_id, client_queue):
        return {"error": "Server busy"}

    async def stream_generator():
        consecutive_errors = 0
        max_consecutive_errors = 5
        last_data_time = time.time()
        
        try:
            while True:
                try:
                    # Check if client is still connected
                    if await request.is_disconnected():
                        print(f"Client {client_id} disconnected")
                        break
                    
                    current_time = time.time()
                    
                    # Check for client timeout (no data for 60 seconds)
                    if current_time - last_data_time > 60:
                        print(f"Client {client_id} timeout, disconnecting")
                        break
                        
                    mp3_chunk = client_queue.get(timeout=1.0)  # Longer timeout for better buffering
                    if mp3_chunk is None:  # Sentinel value to stop streaming
                        print(f"Received sentinel for client {client_id}")
                        break
                        
                    yield bytes(mp3_chunk)
                    last_data_time = current_time
                    consecutive_errors = 0  # Reset error counter
                    
                except queue.Empty:
                    # Only send keep-alive if client is still connected
                    if not await request.is_disconnected():
                        yield b''  # Keep-alive
                    await asyncio.sleep(0.1)  # Shorter sleep for better responsiveness
                    
                except Exception as e:
                    consecutive_errors += 1
                    print(f"Stream generator error for client {client_id} ({consecutive_errors}/{max_consecutive_errors}): {e}")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"Too many errors for client {client_id}, disconnecting")
                        break
                        
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            print(f"Stream generator outer error for client {client_id}: {e}")
        finally:
            print(f"Removing client {client_id}")
            app.state.audio_state.remove_client(client_id)

    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
        "Connection": "keep-alive",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET",
        "Access-Control-Allow-Headers": "*",
        "Transfer-Encoding": "chunked",
        "Content-Type": "audio/mpeg"
    }
    
    return StreamingResponse(
        stream_generator(), 
        media_type="audio/mpeg",
        headers=headers
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=config.HOST, port=config.PORT, log_level=config.LOG_LEVEL, workers=1, reload=False)