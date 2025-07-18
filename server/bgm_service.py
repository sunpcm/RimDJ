import asyncio
import os
import random
import threading
import time
import numpy as np
import librosa
import soundfile as sf
import soxr
import config
from semantic_queue import create_wav_item
from base_service import AsyncServiceBase, safe_put_item, print_processing_info


class BGMService(AsyncServiceBase):
    """背景音乐服务 - 支持多音轨合并和音量控制"""
    
    def __init__(self, bgm_wav_queue, music_dir="../music"):
        super().__init__("BGM")
        self.bgm_wav_queue = bgm_wav_queue
        self.music_dir = os.path.abspath(music_dir)
        
        # BGM状态控制
        self.is_playing = False
        self.current_volume = 0.8  # 正常BGM音量 (80%)
        self.ducked_volume = 0.3   # 播音时的BGM音量 (30%)
        self.is_ducked = False     # 是否处于音量衰减状态
        
        # 音乐播放状态
        self.current_track = None
        self.current_position = 0  # 当前播放位置(毫秒)
        self.music_files = []
        self.track_index = 0
        self.current_broadcast_id = None  # 当前音轨的broadcast_id
        self.chunk_sequence = 0  # 当前音轨的块序号
        
        # 音频处理
        self.chunk_duration_ms = config.CHUNK_DURATION_MS  # 统一使用配置中的块时长
        self.sample_rate = config.SAMPLE_RATE
        
        # 线程控制
        self.bgm_thread = None
        self.stop_event = threading.Event()
        
        # 加载音乐文件列表
        self._load_music_files()
        
    def _load_music_files(self):
        """加载音乐目录下的所有MP3文件"""
        try:
            if os.path.exists(self.music_dir):
                self.music_files = [
                    os.path.join(self.music_dir, f) 
                    for f in os.listdir(self.music_dir) 
                    if f.lower().endswith('.mp3')
                ]
                random.shuffle(self.music_files)  # 随机打乱播放顺序
                print(f"BGM: Loaded {len(self.music_files)} music files")
            else:
                print(f"BGM: Music directory not found: {self.music_dir}")
        except Exception as e:
            print(f"BGM: Error loading music files: {e}")
            self.music_files = []
    
    def start_bgm(self):
        """启动BGM播放"""
        if not self.music_files:
            print("BGM: No music files available")
            return False
            
        if not self.is_playing:
            self.is_playing = True
            self.stop_event.clear()
            self.bgm_thread = threading.Thread(target=self._bgm_loop, daemon=True)
            self.bgm_thread.start()
            print("BGM: Started background music")
            return True
        return False
    
    def stop_bgm(self):
        """停止BGM播放"""
        if self.is_playing:
            self.is_playing = False
            self.stop_event.set()
            if self.bgm_thread and self.bgm_thread.is_alive():
                self.bgm_thread.join(timeout=2.0)
            print("BGM: Stopped background music")
    
    def duck_volume(self):
        """降低BGM音量(播音时调用)"""
        self.is_ducked = True
        print("BGM: Volume ducked for speech")
    
    def restore_volume(self):
        """恢复BGM音量(播音结束时调用)"""
        self.is_ducked = False
        print("BGM: Volume restored")
    
    def set_volume(self, normal_volume=0.8, ducked_volume=0.3):
        """设置BGM音量级别"""
        self.current_volume = normal_volume
        self.ducked_volume = ducked_volume
        print(f"BGM: Volume levels set - Normal: {normal_volume}, Ducked: {ducked_volume}")
    
    def _bgm_loop(self):
        """BGM播放主循环"""
        print(f"BGM: BGM loop started, is_playing={self.is_playing}, music_files={len(self.music_files)}")
        loop_count = 0
        while self.is_playing and not self.stop_event.is_set():
            try:
                loop_count += 1
                if loop_count % 100 == 1:  # 每100次循环打印一次状态
                    print(f"BGM: Loop iteration {loop_count}, is_playing={self.is_playing}")
                
                # 选择下一首音乐
                if not self._load_next_track():
                    print(f"BGM: Failed to load next track, retrying in {config.BGM_LOAD_RETRY_DELAY} seconds...")
                    time.sleep(config.BGM_LOAD_RETRY_DELAY)
                    continue
                
                # 播放当前音轨
                self._play_current_track()
                
            except Exception as e:
                print(f"BGM: Error in BGM loop: {e}")
                time.sleep(1.0)
        print(f"BGM: BGM loop ended after {loop_count} iterations")
    
    def _load_next_track(self):
        """加载下一首音乐"""
        try:
            if not self.music_files:
                return False
            
            # 循环播放，随机选择下一首
            if self.track_index >= len(self.music_files):
                self.track_index = 0
                random.shuffle(self.music_files)  # 重新打乱顺序
            
            track_path = self.music_files[self.track_index]
            self.track_index += 1
            
            # 使用librosa加载音频文件
            audio_data, original_sr = librosa.load(track_path, sr=None, mono=True)
            
            # 标准化音频 (RMS normalization)
            rms = np.sqrt(np.mean(audio_data**2))
            if rms > 0:
                audio_data = audio_data / rms * 0.5  # 标准化到合适的音量（增加到50%）
            
            # 重采样到目标采样率
            if original_sr != self.sample_rate:
                audio_data = soxr.resample(audio_data, original_sr, self.sample_rate)
            
            # 存储音频数据
            self.current_track = audio_data
            self.current_position = 0
            
            # 为新音轨生成broadcast_id并重置序号
            self.current_broadcast_id = f"bgm_track_{int(time.time() * 1000)}"
            self.chunk_sequence = 0
            
            track_name = os.path.basename(track_path)
            print(f"BGM: Loaded track: {track_name} ({len(audio_data)/self.sample_rate:.2f}s)")
            return True
            
        except Exception as e:
            print(f"BGM: Error loading track: {e}")
            return False
    
    def _play_current_track(self):
        """播放当前音轨"""
        if self.current_track is None:
            return
        
        track_length = len(self.current_track)
        track_duration_seconds = track_length / self.sample_rate
        track_start_time = time.time()
        
        print(f"BGM: Starting track playback, duration: {track_duration_seconds:.2f}s")
        
        # 快速生成所有chunk，填充队列
        chunk_count = 0
        while (self.current_position < track_length and 
               self.is_playing and 
               not self.stop_event.is_set()):
            
            try:
                # 计算音频块大小（样本数）
                chunk_samples = int(self.chunk_duration_ms * self.sample_rate / 1000)
                end_pos = min(self.current_position + chunk_samples, track_length)
                
                # 提取音频块
                if end_pos > len(self.current_track):
                    # 如果超出长度，用零填充
                    audio_data = np.zeros(chunk_samples, dtype=np.float32)
                    available_samples = len(self.current_track) - self.current_position
                    if available_samples > 0:
                        audio_data[:available_samples] = self.current_track[self.current_position:]
                else:
                    audio_data = self.current_track[self.current_position:end_pos]
                    # 如果块大小不足，用零填充
                    if len(audio_data) < chunk_samples:
                        padded_data = np.zeros(chunk_samples, dtype=np.float32)
                        padded_data[:len(audio_data)] = audio_data
                        audio_data = padded_data
                
                if len(audio_data) == 0:
                    break
                
                # 应用音量控制
                current_vol = self.ducked_volume if self.is_ducked else self.current_volume
                audio_data = audio_data * current_vol
                
                # 编码为MP3并发送
                success = self._encode_and_send_chunk(audio_data)
                if not success:
                    # 如果队列满了，稍微等待一下再继续
                    time.sleep(config.BGM_QUEUE_FULL_WAIT)
                    continue
                
                # 更新播放位置
                self.current_position = end_pos
                chunk_count += 1
                
            except Exception as e:
                print(f"BGM: Error playing chunk: {e}")
                break
        
        # 计算剩余播放时间
        elapsed_time = time.time() - track_start_time
        remaining_time = track_duration_seconds - elapsed_time
        
        print(f"BGM: Generated {chunk_count} chunks in {elapsed_time:.2f}s, waiting {remaining_time:.2f}s for track completion")
        
        # 等待音轨实际播放完成
        if remaining_time > 0 and self.is_playing and not self.stop_event.is_set():
            time.sleep(remaining_time)
        
        # 音轨播放完成后，添加间隔时间
        if self.is_playing and not self.stop_event.is_set():
            print(f"BGM: Track completed, waiting {config.BGM_TRACK_GAP_SECONDS}s before next track...")
            time.sleep(config.BGM_TRACK_GAP_SECONDS)
    
    def _encode_and_send_chunk(self, audio_data):
        """发送WAV音频块到队列"""
        try:
            # 确保音频数据在正确范围内并直接转换为int16格式
            audio_data = np.clip(audio_data, -1.0, 1.0)
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # 使用统一的broadcast_id，避免每个块都生成新的时间戳
            # 使用当前音轨的broadcast_id确保连续性
            bgm_item = create_wav_item(
                broadcast_id=self.current_broadcast_id,  # 使用音轨级别的broadcast_id
                wav_data=audio_int16,
                sequence_number=self.chunk_sequence,
                is_last=False,  # BGM是连续流，不标记为最后一个
                priority=10,   # BGM优先级最低
                total_items=None  # BGM是连续流，没有固定总数
            )
            
            # 递增块序号
            self.chunk_sequence += 1
            
            # 发送到BGM WAV队列(使用较短超时时间，避免阻塞)
            success = safe_put_item(self.bgm_wav_queue, bgm_item, timeout=0.1)
            if not success:
                # 队列满时返回失败状态
                if self.chunk_sequence % 100 == 1:  # 减少日志频率
                    print(f"BGM: 队列已满，暂停生成 - 块序号: {self.chunk_sequence}")
                    print(f"BGM: BGM WAV队列当前大小: {self.bgm_wav_queue.qsize()}")
                return False
            else:
                # 进一步减少调试日志频率
                if self.chunk_sequence % 200 == 1:  # 每200个块打印一次
                    print(f"BGM: 成功发送音频块到BGM WAV队列 (块序号: {self.chunk_sequence})")
                    print(f"BGM: BGM WAV队列当前大小: {self.bgm_wav_queue.qsize()}")
                return True
                
        except Exception as e:
            print(f"BGM: Error sending WAV chunk: {e}")
            return False
    
    async def process_item(self) -> bool:
        """AsyncServiceBase接口实现 - BGM服务不需要处理队列项目"""
        await asyncio.sleep(0.1)
        return False


class BGMController:
    """BGM控制器 - 提供外部接口控制BGM"""
    
    def __init__(self):
        self.bgm_service = None
        self.speech_active = False
        self.last_speech_time = 0
        self.fade_delay = 2.0  # 播音结束后2秒恢复BGM音量
        
    def initialize(self, bgm_wav_queue, music_dir="../music"):
        """初始化BGM服务"""
        self.bgm_service = BGMService(bgm_wav_queue, music_dir)
        return self.bgm_service
    
    def start_bgm(self):
        """启动BGM"""
        if self.bgm_service:
            return self.bgm_service.start_bgm()
        return False
    
    def stop_bgm(self):
        """停止BGM"""
        if self.bgm_service:
            self.bgm_service.stop_bgm()
    
    def on_speech_start(self):
        """播音开始时调用"""
        self.speech_active = True
        if self.bgm_service:
            self.bgm_service.duck_volume()
    
    def on_speech_end(self):
        """播音结束时调用"""
        self.speech_active = False
        self.last_speech_time = time.time()
        
        # 延迟恢复BGM音量
        def delayed_restore():
            time.sleep(self.fade_delay)
            if not self.speech_active and self.bgm_service:
                self.bgm_service.restore_volume()
        
        threading.Thread(target=delayed_restore, daemon=True).start()
    
    def set_volume_levels(self, normal=0.8, ducked=0.3):
        """设置音量级别"""
        if self.bgm_service:
            self.bgm_service.set_volume(normal, ducked)


# 全局BGM控制器实例
bgm_controller = BGMController()


def start_bgm_service(bgm_wav_queue, music_dir="../music"):
    """启动BGM服务"""
    bgm_service = bgm_controller.initialize(bgm_wav_queue, music_dir)
    if bgm_service:
        # 启动异步服务(虽然不处理队列项目，但保持一致的接口)
        bgm_service.start()
        return bgm_service
    return None


def get_bgm_controller():
    """获取BGM控制器实例"""
    return bgm_controller