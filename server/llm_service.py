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
        # 检查MP3队列状态和LLM处理状态
        if self.mp3_stream_queue and self.mp3_stream_queue.qsize() >= 50:
            return False
        
        if self.is_processing:
            return False
        
        # 检查是否应该处理
        texts = self._collect_texts()
        if not texts:
            return False
        
        # 处理文本
        combined_text = '\n'.join(texts)
        broadcast_id = create_broadcast_id()
        
        print_processing_info("LLM", broadcast_id, f"processing {len(texts)} texts")
        
        self.is_processing = True  # 设置处理状态
        try:
            # 调用LLM API
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=config.LLM_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt.SYSTEM_PROMPT},
                        {"role": "user", "content": combined_text}
                    ]
                ),
                timeout=config.LLM_TIMEOUT
            )
            
            processed_text = response.choices[0].message.content
            print_processing_info("LLM", broadcast_id, f"generated: {processed_text[:50]}...")
            
            # 创建并放入广播稿
            script_item = create_script_item(
                broadcast_id=broadcast_id,
                content=processed_text,
                priority=0
            )
            
            success = safe_put_item(self.broadcast_script_queue, script_item, timeout=2.0)
            if success:
                self.last_process_time = time.time()
                print_processing_info("LLM", broadcast_id, "successfully queued")
            else:
                print_processing_info("LLM", broadcast_id, "queue full, dropping")
                
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