"""基础服务模块

提供所有服务的公共基类和工具函数
"""

import asyncio
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional, Any
import config


class AsyncServiceBase(ABC):
    """异步服务基类
    
    提供标准的异步工作器模式和错误处理
    """
    
    def __init__(self, name: str, max_consecutive_errors: int = None):
        self.name = name
        self.max_consecutive_errors = max_consecutive_errors or config.MAX_CONSECUTIVE_ERRORS
        self.consecutive_errors = 0
        self.is_running = False
    
    @abstractmethod
    async def process_item(self) -> bool:
        """处理单个项目
        
        Returns:
            bool: 是否成功处理项目，False表示没有项目可处理
        """
        pass
    
    async def worker_loop(self):
        """主工作循环"""
        self.is_running = True
        print(f"{self.name} service started")
        
        while self.is_running:
            try:
                # 处理项目
                processed = await self.process_item()
                
                if processed:
                    # 成功处理，重置错误计数
                    self.consecutive_errors = 0
                else:
                    # 没有项目可处理，短暂休眠
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                self.consecutive_errors += 1
                print(f"{self.name} service error ({self.consecutive_errors}/{self.max_consecutive_errors}): {e}")
                
                if self.consecutive_errors >= self.max_consecutive_errors:
                    print(f"{self.name} service: too many consecutive errors, restarting...")
                    self.consecutive_errors = 0
                    await asyncio.sleep(5)
                else:
                    await asyncio.sleep(min(self.consecutive_errors, 5))
    
    def start(self):
        """启动服务"""
        def run_async_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.worker_loop())
        
        thread = threading.Thread(target=run_async_loop, daemon=True)
        thread.start()
        return thread
    
    def stop(self):
        """停止服务"""
        self.is_running = False


def safe_get_item(queue, timeout: float = 0.1) -> Optional[Any]:
    """安全获取队列项目
    
    Args:
        queue: 队列对象
        timeout: 超时时间
        
    Returns:
        获取的项目或None
    """
    try:
        if hasattr(queue, 'get_broadcast_item'):
            return queue.get_broadcast_item(timeout=timeout)
        else:
            return queue.get(timeout=timeout)
    except Exception:
        return None


def safe_put_item(queue, item: Any, timeout: float = 2.0) -> bool:
    """安全放入队列项目
    
    Args:
        queue: 队列对象
        item: 要放入的项目
        timeout: 超时时间
        
    Returns:
        bool: 是否成功放入
    """
    try:
        if hasattr(queue, 'put_broadcast_item'):
            return queue.put_broadcast_item(item, timeout=timeout)
        else:
            queue.put(item, timeout=timeout)
            return True
    except Exception as e:
        print(f"Error putting item to queue: {e}")
        return False


def print_processing_info(service_name: str, broadcast_id: str, info: str):
    """打印处理信息
    
    Args:
        service_name: 服务名称
        broadcast_id: 广播ID
        info: 信息内容
    """
    short_id = broadcast_id[:8] if broadcast_id else "unknown"
    print(f"[{service_name}] {short_id}: {info}")