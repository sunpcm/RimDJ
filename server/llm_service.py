import asyncio
import time
import queue
from openai import AsyncOpenAI
import config
import system_prompt
from semantic_queue import create_broadcast_id, create_script_item
from base_service import AsyncServiceBase, safe_put_item, print_processing_info


class LLMService(AsyncServiceBase):
    """LLM处理服务"""
    
    def __init__(self, text_queue, broadcast_script_queue, mp3_stream_queue=None):
        super().__init__("LLM")
        self.text_queue = text_queue
        self.broadcast_script_queue = broadcast_script_queue
        self.mp3_stream_queue = mp3_stream_queue
        self.client = AsyncOpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
        self.last_process_time = time.time()
        self.is_processing = False  # 添加处理状态标志
    
    async def process_item(self) -> bool:
        """处理文本队列中的项目"""
        # 检查MP3队列状态和LLM处理状态, 如果mp3_stream_queue长度大于50(约3s)，则暂时等待
        if self.mp3_stream_queue and self.mp3_stream_queue.qsize() >= 50:
            return False
        
        # 当前正在请求，也等待
        if self.is_processing:
            return False
        
        # 检查是否应该处理，收集待处理的日志文本
        texts = self._collect_texts()
        if not texts:
            return False
        
        # 处理文本
        combined_text = '\n'.join(texts)
        broadcast_id = create_broadcast_id()
        
        print_processing_info("LLM", broadcast_id, f"processing {len(texts)} texts")
        
        self.is_processing = True  # 设置处理状态
        try:
            # 调用LLM API（流式）
            stream = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=config.LLM_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt.SYSTEM_PROMPT},
                        {"role": "user", "content": combined_text}
                    ],
                    stream=True
                ),
                timeout=config.LLM_TIMEOUT
            )
            
            # 处理流式响应
            accumulated_text = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    accumulated_text += chunk.choices[0].delta.content
                    
                    # 检查是否需要切分
                    segments = self._split_text_by_rules(accumulated_text)
                    
                    # 处理完整的段落
                    for i, segment in enumerate(segments[:-1]):  # 除了最后一个未完成的段落
                        if segment.strip():  # 确保不是空段落
                            script_item = create_script_item(
                                broadcast_id=broadcast_id,
                                content=segment.strip(),
                                priority=0
                            )
                            
                            success = safe_put_item(self.broadcast_script_queue, script_item, timeout=1.0)
                            if success:
                                print_processing_info("LLM", broadcast_id, f"queued segment: {segment[:30]}...")
                            else:
                                print_processing_info("LLM", broadcast_id, "queue full, dropping segment")
                    
                    # 保留最后一个未完成的段落
                    accumulated_text = segments[-1] if segments else ""
                    
                    # 如果累积文本过长且没有找到分隔符，强制切分
                    if len(accumulated_text) >= 80:  # 超过80字符强制切分
                        # 尝试在合适的位置切分（优先在空格、逗号等处）
                        cut_pos = self._find_best_cut_position(accumulated_text, 100)
                        if cut_pos > 20:  # 确保切分的部分足够长
                            forced_segment = accumulated_text[:cut_pos]
                            accumulated_text = accumulated_text[cut_pos:]
                            
                            script_item = create_script_item(
                                broadcast_id=broadcast_id,
                                content=forced_segment.strip(),
                                priority=0
                            )
                            
                            success = safe_put_item(self.broadcast_script_queue, script_item, timeout=1.0)
                            if success:
                                print_processing_info("LLM", broadcast_id, f"queued forced segment: {forced_segment[:30]}...")
                            else:
                                print_processing_info("LLM", broadcast_id, "queue full, dropping forced segment")
            
            # 处理最后剩余的文本
            if accumulated_text.strip():
                script_item = create_script_item(
                    broadcast_id=broadcast_id,
                    content=accumulated_text.strip(),
                    priority=0
                )
                
                success = safe_put_item(self.broadcast_script_queue, script_item, timeout=1.0)
                if success:
                    print_processing_info("LLM", broadcast_id, f"queued final segment: {accumulated_text}")
                else:
                    print_processing_info("LLM", broadcast_id, "queue full, dropping final segment")
            
            self.last_process_time = time.time()
            print_processing_info("LLM", broadcast_id, "stream processing completed")
                
        except asyncio.TimeoutError:
            print_processing_info("LLM", broadcast_id, "request timeout, retrying")
            self._requeue_texts(texts)
            
        except Exception as e:
            print_processing_info("LLM", broadcast_id, f"processing failed: {e}")
            self._requeue_texts(texts)
        
        finally:
            self.is_processing = False  # 重置处理状态
        
        return True
    
    def _collect_texts(self) -> list:
        """收集待处理的文本"""
        queue_size = self.text_queue.qsize()
        current_time = time.time()
        
        # 判断是否应该处理
        should_process = (
            queue_size > 0 and 
            (queue_size >= 10 or (current_time - self.last_process_time) >= 30.0)
        )
        
        if not should_process:
            return []
        
        # 收集文本
        texts = []
        for _ in range(min(queue_size, 10)):
            try:
                texts.append(self.text_queue.get_nowait())
            except queue.Empty:
                break
        
        return texts
    
    def _split_text_by_rules(self, text: str) -> list:
        """根据规则切分文本"""
        import re
        
        # 定义切分规则的正则表达式
        # 中文的。？！和英文的. ! ?（后面跟空格）
        split_pattern = r'([。？！]|[.!?]\s)'
        
        # 使用正则表达式分割，保留分隔符
        parts = re.split(split_pattern, text)
        
        segments = []
        current_segment = ""
        
        i = 0
        while i < len(parts):
            current_segment += parts[i]
            
            # 如果当前部分是分隔符，添加到当前段落
            if i + 1 < len(parts) and re.match(split_pattern, parts[i + 1]):
                current_segment += parts[i + 1]
                i += 2
                
                # 检查长度是否满足要求（至少30个字符）
                if len(current_segment) >= 30:
                    segments.append(current_segment)
                    current_segment = ""
            else:
                i += 1
        
        # 添加剩余的文本
        if current_segment:
            segments.append(current_segment)
        
        # 如果没有找到分割点，但文本长度超过100字符，强制分割
        if len(segments) == 1 and len(segments[0]) > 100:
            text = segments[0]
            segments = []
            for i in range(0, len(text), 80):  # 每80字符强制分割
                segments.append(text[i:i+80])
        
        return segments if segments else [text]
    
    def _find_best_cut_position(self, text: str, target_pos: int) -> int:
        """在目标位置附近找到最佳的切分位置"""
        if target_pos >= len(text):
            return len(text)
        
        # 定义优先切分的字符（按优先级排序）
        cut_chars = ['，', ',', '、', '；', ';', ' ', '的', '了', '在', '和', '与']
        
        # 在目标位置前后20个字符范围内寻找最佳切分点
        search_start = max(0, target_pos - 20)
        search_end = min(len(text), target_pos + 20)
        
        # 优先在目标位置之前寻找切分点
        for char in cut_chars:
            for i in range(target_pos, search_start - 1, -1):
                if i < len(text) and text[i] == char:
                    return i + 1  # 在分隔符后切分
        
        # 如果前面没找到，在目标位置之后寻找
        for char in cut_chars:
            for i in range(target_pos + 1, search_end):
                if i < len(text) and text[i] == char:
                    return i + 1  # 在分隔符后切分
        
        # 如果都没找到合适的切分点，就在目标位置切分
        return target_pos
    
    def _requeue_texts(self, texts: list):
        """将文本重新放回队列"""
        for text in reversed(texts):
            try:
                self.text_queue.put(text, timeout=0.1)
            except queue.Full:
                print(f"Failed to requeue text: {text[:50]}...")
                break


def start_llm_service(text_queue, broadcast_script_queue, mp3_stream_queue=None):
    """启动LLM服务"""
    service = LLMService(text_queue, broadcast_script_queue, mp3_stream_queue)
    return service.start()