import torch
import numpy as np
import re
import asyncio
import random
import time
from transformers import AutoTokenizer
from tensorrt_llm.llmapi import LLM, SamplingParams, KvCacheConfig 
from sparktts.models.audio_tokenizer import BiCodecTokenizer
import config
import os
from semantic_queue import create_wav_item
from base_service import AsyncServiceBase, safe_get_item, safe_put_item, print_processing_info


def trim_audio_silence(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    silence_threshold: float = 0.005,  # 更严格的阈值
    min_audio_length: float = 2.0  # 最小音频长度（秒）
) -> np.ndarray:
    """
    简化版本的静音去除函数
    
    Args:
        waveform (np.ndarray): 输入音频波形
        sample_rate (int): 采样率
        silence_threshold (float): 静音阈值
        min_audio_length (float): 最小保留音频长度
    
    Returns:
        np.ndarray: 处理后的音频
    """
    
    if len(waveform) == 0:
        return waveform
    
    # 计算绝对值的移动平均来检测音频活动
    window_size = int(0.05 * sample_rate)  # 50ms窗口
    abs_audio = np.abs(waveform)
    
    # 使用卷积计算移动平均
    if len(abs_audio) < window_size:
        return waveform
    
    moving_avg = np.convolve(abs_audio, np.ones(window_size)/window_size, mode='valid')
    
    # 找到最后一个超过阈值的位置
    threshold = np.max(moving_avg) * silence_threshold
    active_indices = np.where(moving_avg > threshold)[0]
    
    if len(active_indices) == 0:
        # 如果没有找到活动音频，返回最小长度
        min_samples = int(min_audio_length * sample_rate)
        return waveform[:min(min_samples, len(waveform))]
    
    # 最后一个活动位置
    last_active = active_indices[-1] + window_size  # 补偿卷积的偏移
    
    # 添加一些缓冲
    buffer_samples = int(0.2 * sample_rate)  # 0.2秒缓冲
    end_point = min(last_active + buffer_samples, len(waveform))
    
    # 确保最小长度
    min_samples = int(min_audio_length * sample_rate)
    end_point = max(end_point, min_samples)
    
    return waveform[:end_point]


class TTSService(AsyncServiceBase):
    """TTS处理服务"""
    
    def __init__(self, broadcast_script_queue, wav_queue):
        super().__init__("TTS")
        self.broadcast_script_queue = broadcast_script_queue
        self.wav_queue = wav_queue
        self._initialize_models()
    
    def _initialize_models(self):
        """初始化TTS模型"""
        # 创建KV cache配置，限制显存使用
        kv_cache_config = KvCacheConfig(
            max_tokens=4096,  # 减少最大token数量
            free_gpu_memory_fraction=0.3,  # 只使用30%的剩余显存
            enable_block_reuse=True  # 启用block重用
        )

        ENGINE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "trt_engines", "spark-tts-tensorrt", "bf16", "1-gpu"))
        TOKENIZER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "spark-tts-merged"))
        AUDIO_TOKENIZER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Spark-TTS-0.5B"))
        
        self.audio_tokenizer = BiCodecTokenizer(AUDIO_TOKENIZER_DIR, "cuda")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
        self.model = LLM(model=ENGINE_DIR, tokenizer=self.tokenizer, kv_cache_config=kv_cache_config)
    
    async def process_item(self) -> bool:
        """处理广播稿队列中的项目"""
        # 获取广播稿项目
        broadcast_item = safe_get_item(self.broadcast_script_queue, timeout=0.1)
        if not broadcast_item:
            return False
        
        text = broadcast_item.content
        broadcast_id = broadcast_item.broadcast_id
        
        if not text or not text.strip():
            return True
        
        print_processing_info("TTS", broadcast_id, f"processing: {text[:50]}...")
        
        try:
            # 分割文本为句子
            sentences = self._split_text_by_punctuation(text)
            total_sentences = len(sentences)
            
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    await self._process_sentence(
                        sentence, broadcast_item, i, total_sentences
                    )
                    await asyncio.sleep(0.01)  # 句子间小延迟
                    
        except Exception as e:
            print_processing_info("TTS", broadcast_id, f"processing error: {e}")
        
        return True
    
    async def _process_sentence(self, sentence: str, broadcast_item, sequence_number: int, total_sentences: int):
        """处理单个句子"""
        broadcast_id = broadcast_item.broadcast_id
        
        try:
            print_processing_info("TTS", broadcast_id, f"sentence {sequence_number+1}/{total_sentences}: {sentence[:30]}...")
            
            # 在线程池中运行TTS生成（带重试机制）
            loop = asyncio.get_event_loop()
            wav = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    self._generate_speech_from_text_with_retry,
                    sentence, 0.6, 50, 0.95, 2048, 3, 0.1
                ),
                timeout=config.TTS_TIMEOUT
            )
            
            if wav is not None and wav.size > 0:
                # 对生成的音频进行尾部静音处理
                try:
                    wav_trimmed = trim_audio_silence(
                        wav, 
                        sample_rate=16000,  # 假设采样率为16kHz
                        silence_threshold=0.005,
                        min_audio_length=2.0  # 最小音频长度2.0秒，避免过度裁剪
                    )
                    print_processing_info("TTS", broadcast_id, f"trimmed audio from {wav.shape} to {wav_trimmed.shape}")
                    wav = wav_trimmed
                except Exception as e:
                    print_processing_info("TTS", broadcast_id, f"audio trimming failed: {e}, using original audio")
                
                # 创建WAV项目
                wav_item = create_wav_item(
                    broadcast_id=broadcast_id,
                    wav_data=wav,
                    sequence_number=sequence_number,
                    total_items=total_sentences,
                    is_last=(sequence_number == total_sentences - 1),
                    priority=broadcast_item.priority
                )
                
                success = safe_put_item(self.wav_queue, wav_item, timeout=2.0)
                if success:
                    print_processing_info("TTS", broadcast_id, f"generated audio {sequence_number+1}/{total_sentences}, shape: {wav.shape}")
                else:
                    print_processing_info("TTS", broadcast_id, f"queue full, dropping audio chunk {sequence_number+1}")
            else:
                print_processing_info("TTS", broadcast_id, f"empty audio for sentence: {sentence[:30]}...")
                
        except asyncio.TimeoutError:
            print_processing_info("TTS", broadcast_id, f"timeout for sentence: {sentence[:30]}...")
        except Exception as e:
            print_processing_info("TTS", broadcast_id, f"sentence error: {e}")
    
    def _split_text_by_punctuation(self, text: str, min_length: int = 50) -> list:
        """按标点符号分割文本"""
        pattern = r'([。！？]|[.!?]\s*)'
        parts = re.split(pattern, text)
        sentences = []
        current_sentence = ""
        
        for part in parts:
            if part.strip():
                if re.match(r'^[。！？]$', part) or re.match(r'^[.!?]\s*$', part):
                    current_sentence += part
                    if len(current_sentence.strip()) >= min_length:
                        sentences.append(current_sentence.strip())
                        current_sentence = ""
                else:
                    current_sentence += part
        
        if current_sentence.strip():
            if len(current_sentence.strip()) < min_length and sentences:
                sentences[-1] += current_sentence
            else:
                sentences.append(current_sentence.strip())
        
        # 过滤空字符串并替换中文标点为逗号
        result = []
        for s in sentences:
            if s:
                # 替换中文标点符号：。！？→，
                s = s.replace('。', '，')
                s = s.replace('！', '，')
                s = s.replace('？', '，')
                result.append(s)
        return result
    
    def _generate_speech_from_text_with_retry(self, text: str, temperature: float = 0.6, 
                                            top_k: int = 50, top_p: float = 0.95, 
                                            max_new_audio_tokens: int = 2048, 
                                            max_retries: int = 3, base_delay: float = 0.1) -> np.ndarray:
        """带重试机制的语音生成"""
        for attempt in range(max_retries):
            try:
                # 生成随机种子
                seed = random.randint(0, 2**32 - 1)
                
                # 设置随机种子
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)
                
                wav = self._generate_speech_from_text(
                    text, temperature, top_k, top_p, max_new_audio_tokens
                )
                
                if wav is not None and wav.size > 0:
                    return wav
                else:
                    raise ValueError("生成的音频为空")
                    
            except Exception as e:
                text = text + "。"
                if attempt == max_retries - 1:  # 最后一次尝试
                    raise e
                
                # 简单延迟重试
                delay = base_delay
                print_processing_info("TTS", "retry", f"尝试 {attempt + 1} 失败: {str(e)}, {delay:.1f}秒后重试...")
                time.sleep(delay)
        
        return None
    
    @torch.inference_mode()
    def _generate_speech_from_text(self, text: str, temperature: float = 0.6, 
                                 top_k: int = 50, top_p: float = 0.95, 
                                 max_new_audio_tokens: int = 2048) -> np.ndarray:
        """从文本生成语音（单次尝试）"""
        prompt = "".join([
            "<|task_tts|>", "<|start_content|>", text, 
            "<|end_content|>", "<|start_global_token|>"
        ])
        
        sampling_params = SamplingParams(
            max_tokens=max_new_audio_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            end_id=self.tokenizer.eos_token_id,
            pad_id=self.tokenizer.pad_token_id
        )
        
        outputs = self.model.generate([prompt], sampling_params=sampling_params)
        predicts_text = outputs[0].outputs[0].text
        
        # 提取语义token
        semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", predicts_text)
        if not semantic_matches:
            return np.array([], dtype=np.float32)
        
        pred_semantic_ids = torch.tensor([int(token) for token in semantic_matches]).long().unsqueeze(0)
        
        # 提取全局token
        global_matches = re.findall(r"<\|bicodec_global_(\d+)\|>", predicts_text)
        pred_global_ids = (
            torch.tensor([int(token) for token in global_matches]).long().unsqueeze(0) 
            if global_matches else torch.zeros((1, 1), dtype=torch.long)
        )
        pred_global_ids = pred_global_ids.unsqueeze(0)
        
        # 解码为音频
        self.audio_tokenizer.device = self.device
        self.audio_tokenizer.model.to(self.device)
        wav_np = self.audio_tokenizer.detokenize(
            pred_global_ids.to(self.device).squeeze(0),
            pred_semantic_ids.to(self.device)
        )
        
        return wav_np


def start_tts_service(broadcast_script_queue, wav_queue):
    """启动TTS服务"""
    service = TTSService(broadcast_script_queue, wav_queue)
    return service.start()