import asyncio
import time
import numpy as np
from collections import deque
import config
from semantic_queue import create_wav_item
from base_service import AsyncServiceBase, safe_get_item, safe_put_item, print_processing_info


class AudioMixerService(AsyncServiceBase):
    """音频混合服务 - 混合TTS和BGM的WAV音频"""
    
    def __init__(self, tts_wav_queue, bgm_wav_queue, mixed_wav_queue):
        super().__init__("AudioMixer")
        self.tts_wav_queue = tts_wav_queue  # TTS WAV音频队列
        self.bgm_wav_queue = bgm_wav_queue  # BGM WAV音频队列
        self.mixed_wav_queue = mixed_wav_queue  # 混合后的WAV输出队列
        
        # 混合状态
        self.mix_sequence = 0
        self.last_activity_time = time.time()
        
        # 音频缓冲区
        self.tts_buffer = deque()
        self.bgm_buffer = deque()  # BGM音频缓冲区，用于对齐
        self.bgm_continuous_buffer = np.array([], dtype=np.int16)  # 连续BGM音频缓冲
        
        # 音频混合参数
        self.chunk_size = config.CHUNK_SIZE  # 标准chunk大小
        self.sample_rate = config.SAMPLE_RATE
        
        # BGM音量控制
        self.bgm_volume = 0.3  # BGM在有TTS时的音量
        self.bgm_solo_volume = 0.8  # BGM单独播放时的音量
        
        # BGM缓冲区管理
        self.max_bgm_buffer_size = self.sample_rate * 10  # 最大缓冲10秒的BGM音频
        
        print("AudioMixer: 初始化完成，使用WAV级别的音频混合模式，支持音频块对齐")
        
    def _collect_bgm_audio(self):
        """收集BGM音频到连续缓冲区"""
        while True:
            bgm_item = safe_get_item(self.bgm_wav_queue, timeout=0.01)
            if not bgm_item:
                break
                
            # 将BGM音频添加到连续缓冲区
            self.bgm_continuous_buffer = np.concatenate([self.bgm_continuous_buffer, bgm_item.content])
            
            # 限制缓冲区大小，避免内存过度使用
            if len(self.bgm_continuous_buffer) > self.max_bgm_buffer_size:
                excess = len(self.bgm_continuous_buffer) - self.max_bgm_buffer_size
                self.bgm_continuous_buffer = self.bgm_continuous_buffer[excess:]
    
    def _get_aligned_bgm(self, target_length):
        """获取与目标长度对齐的BGM音频"""
        # 确保有足够的BGM音频
        self._collect_bgm_audio()
        
        if len(self.bgm_continuous_buffer) < target_length:
            # BGM不足，用零填充或重复现有BGM
            if len(self.bgm_continuous_buffer) > 0:
                # 重复BGM直到达到目标长度
                repeat_times = (target_length // len(self.bgm_continuous_buffer)) + 1
                repeated_bgm = np.tile(self.bgm_continuous_buffer, repeat_times)
                aligned_bgm = repeated_bgm[:target_length]
            else:
                # 没有BGM，返回静音
                aligned_bgm = np.zeros(target_length, dtype=np.int16)
        else:
            # BGM充足，截取所需长度
            aligned_bgm = self.bgm_continuous_buffer[:target_length]
            # 从缓冲区移除已使用的部分
            self.bgm_continuous_buffer = self.bgm_continuous_buffer[target_length:]
        
        return aligned_bgm

    async def process_item(self) -> bool:
        """处理一个音频混合任务"""
        try:
            # 尝试获取TTS音频
            tts_item = safe_get_item(self.tts_wav_queue, timeout=0.1)
            
            if tts_item:
                # 有TTS音频，进行混合处理
                return await self._process_with_tts(tts_item)
            else:
                # 没有TTS音频，处理纯BGM
                return await self._process_bgm_only()
                
        except Exception as e:
            print(f"AudioMixer: 处理音频时出错: {e}")
            return False
    
    async def _process_with_tts(self, tts_item):
        """处理TTS音频与BGM的混合"""
        try:
            # 获取与TTS音频长度对齐的BGM音频
            tts_length = len(tts_item.content)
            aligned_bgm = self._get_aligned_bgm(tts_length)
            
            # 混合TTS和BGM
            mixed_audio = self._mix_audio_chunks(
                tts_item.content, 
                aligned_bgm, 
                self.bgm_volume
            )
            
            # 创建混合后的音频项
            mixed_item = create_wav_item(
                broadcast_id=tts_item.broadcast_id,
                wav_data=mixed_audio,
                sequence_number=self.mix_sequence,
                is_last=False
            )
            
            # 发送混合后的音频
            success = safe_put_item(self.mixed_wav_queue, mixed_item, timeout=1.0)
            if success:
                self.mix_sequence += 1
                self.last_activity_time = time.time()
                print(f"AudioMixer: 成功混合TTS音频，长度: {tts_length}, BGM对齐长度: {len(aligned_bgm)}")
                return True
            else:
                print("AudioMixer: 混合音频队列已满，跳过此音频块")
                return False
                
        except Exception as e:
            print(f"AudioMixer: TTS音频混合处理出错: {e}")
            return False
    
    async def _process_bgm_only(self):
        """处理纯BGM音频"""
        try:
            # 获取标准长度的BGM音频（使用配置的chunk_size）
            bgm_audio = self._get_aligned_bgm(self.chunk_size)
            
            if len(bgm_audio) == 0:
                # 没有BGM音频可处理，生成静音数据以保持流的连续性
                if self.mix_sequence % 200 == 0:  # 每200次输出一次调试信息
                    print(f"AudioMixer: 没有BGM音频，生成静音数据，序号: {self.mix_sequence}")
                
                # 生成静音数据
                silence_audio = np.zeros(self.chunk_size, dtype=np.int16)
                
                # 创建静音音频项
                silence_item = create_wav_item(
                    broadcast_id=f"silence_{self.mix_sequence}",
                    wav_data=silence_audio,
                    sequence_number=self.mix_sequence,
                    is_last=False
                )
                
                # 发送静音音频
                success = safe_put_item(self.mixed_wav_queue, silence_item, timeout=1.0)
                if success:
                    self.mix_sequence += 1
                    self.last_activity_time = time.time()
                    return True
                else:
                    await asyncio.sleep(0.01)
                    return False
            
            # 应用BGM单独播放时的音量
            bgm_audio = (bgm_audio * self.bgm_solo_volume).astype(np.int16)
            
            # 添加调试信息：检查BGM音频数据的有效性
            if self.mix_sequence % 100 == 1:
                audio_stats = {
                    'length': len(bgm_audio),
                    'min': np.min(bgm_audio),
                    'max': np.max(bgm_audio),
                    'mean': np.mean(bgm_audio),
                    'std': np.std(bgm_audio)
                }
                print(f"AudioMixer: BGM音频统计 - 长度: {audio_stats['length']}, 范围: [{audio_stats['min']}, {audio_stats['max']}], 均值: {audio_stats['mean']:.2f}, 标准差: {audio_stats['std']:.2f}")
            
            # 创建BGM音频项
            bgm_item = create_wav_item(
                broadcast_id=f"bgm_solo_{self.mix_sequence}",
                wav_data=bgm_audio,
                sequence_number=self.mix_sequence,
                is_last=False
            )
            
            # 发送BGM音频
            success = safe_put_item(self.mixed_wav_queue, bgm_item, timeout=1.0)
            if success:
                self.mix_sequence += 1
                self.last_activity_time = time.time()
                if self.mix_sequence % 100 == 1:  # 减少日志频率
                    print(f"AudioMixer: 处理纯BGM音频块，序号: {self.mix_sequence}")
                return True
            else:
                print("AudioMixer: BGM音频队列已满，跳过此音频块")
                return False
                
        except Exception as e:
            print(f"AudioMixer: BGM处理出错: {e}")
            return False
    
    def _mix_audio_chunks(self, tts_audio, bgm_audio, bgm_volume):
        """混合TTS和BGM音频块"""
        # 应用BGM音量
        bgm_audio_scaled = (bgm_audio * bgm_volume).astype(np.int16)
        
        # 确保两个音频块长度一致
        min_length = min(len(tts_audio), len(bgm_audio_scaled))
        max_length = max(len(tts_audio), len(bgm_audio_scaled))
        
        # 创建输出数组
        mixed = np.zeros(max_length, dtype=np.int32)  # 使用int32避免溢出
        
        # 混合音频
        mixed[:min_length] = tts_audio[:min_length].astype(np.int32) + bgm_audio_scaled[:min_length].astype(np.int32)
        
        # 处理剩余部分
        if len(tts_audio) > min_length:
            mixed[min_length:] = tts_audio[min_length:].astype(np.int32)
        elif len(bgm_audio_scaled) > min_length:
            mixed[min_length:] = bgm_audio_scaled[min_length:].astype(np.int32)
        
        # 防止溢出并转换回int16
        mixed = np.clip(mixed, -32768, 32767).astype(np.int16)
        
        return mixed
    

    

    
    def get_status(self):
        """获取混合器状态"""
        return {
            'last_activity': self.last_activity_time,
            'mix_sequence': self.mix_sequence,
            'tts_wav_queue_size': self.tts_wav_queue.qsize(),
            'bgm_wav_queue_size': self.bgm_wav_queue.qsize(),
            'mixed_wav_queue_size': self.mixed_wav_queue.qsize(),
            'bgm_volume': self.bgm_volume,
            'bgm_solo_volume': self.bgm_solo_volume,
            'bgm_buffer_size': len(self.bgm_continuous_buffer),
            'bgm_buffer_duration_seconds': len(self.bgm_continuous_buffer) / self.sample_rate if len(self.bgm_continuous_buffer) > 0 else 0
        }


def start_audio_mixer_service(tts_wav_queue, bgm_wav_queue, mixed_wav_queue):
    """启动音频混合服务"""
    service = AudioMixerService(tts_wav_queue, bgm_wav_queue, mixed_wav_queue)
    return service.start()