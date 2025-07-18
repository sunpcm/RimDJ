"""语义感知队列管理模块

提供广播稿语义一致性支持的队列管理功能
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import uuid
import time
import queue
import threading


@dataclass
class BroadcastItem:
    """广播稿数据包装器
    
    用于在队列中传递带有语义标识的数据项目
    """
    broadcast_id: str  # 广播稿唯一标识
    content: Any  # 实际数据内容（文本、WAV、MP3块等）
    item_type: str  # 数据类型：'script', 'sentence', 'wav', 'mp3_chunk'
    sequence_number: int  # 在广播稿中的序号
    total_items: Optional[int] = None  # 该广播稿的总项目数
    is_last: bool = False  # 是否为最后一项
    timestamp: float = None  # 创建时间戳
    priority: int = 0  # 优先级（数字越小优先级越高）
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def __str__(self):
        return f"BroadcastItem(id={self.broadcast_id[:8]}, type={self.item_type}, seq={self.sequence_number}, last={self.is_last})"


class BroadcastTracker:
    """广播稿跟踪器
    
    跟踪每个广播稿的状态和完整性
    """
    def __init__(self):
        self.broadcasts: Dict[str, Dict] = {}
        self.lock = threading.Lock()
    
    def register_item(self, item: BroadcastItem):
        """注册广播项目"""
        with self.lock:
            if item.broadcast_id not in self.broadcasts:
                self.broadcasts[item.broadcast_id] = {
                    'total_items': item.total_items,
                    'received_items': 0,
                    'start_time': item.timestamp,
                    'last_update': item.timestamp,
                    'item_type': item.item_type,
                    'priority': item.priority,
                    'is_complete': False
                }
            
            broadcast_info = self.broadcasts[item.broadcast_id]
            broadcast_info['received_items'] += 1
            broadcast_info['last_update'] = item.timestamp
            
            # 检查是否完成
            if item.is_last or (
                broadcast_info['total_items'] is not None and 
                broadcast_info['received_items'] >= broadcast_info['total_items']
            ):
                broadcast_info['is_complete'] = True
    
    def get_incomplete_broadcasts(self, max_age_seconds: float = 30) -> List[str]:
        """获取不完整的过期广播稿ID列表"""
        current_time = time.time()
        incomplete = []
        
        with self.lock:
            for broadcast_id, info in self.broadcasts.items():
                if not info['is_complete'] and (current_time - info['start_time']) > max_age_seconds:
                    incomplete.append(broadcast_id)
        
        return incomplete
    
    def remove_broadcasts(self, broadcast_ids: List[str]):
        """移除指定的广播稿"""
        with self.lock:
            for broadcast_id in broadcast_ids:
                if broadcast_id in self.broadcasts:
                    del self.broadcasts[broadcast_id]
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        with self.lock:
            total = len(self.broadcasts)
            complete = sum(1 for info in self.broadcasts.values() if info['is_complete'])
            incomplete = total - complete
            
            return {
                'total_broadcasts': total,
                'complete_broadcasts': complete,
                'incomplete_broadcasts': incomplete,
                'oldest_incomplete_age': self._get_oldest_incomplete_age()
            }
    
    def _get_oldest_incomplete_age(self) -> Optional[float]:
        """获取最老的不完整广播稿的年龄"""
        current_time = time.time()
        oldest_age = None
        
        for info in self.broadcasts.values():
            if not info['is_complete']:
                age = current_time - info['start_time']
                if oldest_age is None or age > oldest_age:
                    oldest_age = age
        
        return oldest_age


class SemanticAwareQueue:
    """语义感知的队列管理器
    
    支持广播稿语义一致性的队列，提供智能背压处理
    """
    def __init__(self, maxsize: int = 100, name: str = "SemanticQueue"):
        self.queue = queue.Queue(maxsize=maxsize)
        self.maxsize = maxsize
        self.name = name
        self.tracker = BroadcastTracker()
        self.stats = {
            'total_put': 0,
            'total_get': 0,
            'dropped_items': 0,
            'dropped_broadcasts': 0
        }
        self.lock = threading.Lock()
    
    def put_broadcast_item(self, item: BroadcastItem, timeout: Optional[float] = None) -> bool:
        """放入广播项目
        
        Returns:
            bool: 是否成功放入
        """
        try:
            self.queue.put(item, timeout=timeout)
            self.tracker.register_item(item)
            
            with self.lock:
                self.stats['total_put'] += 1
            
            return True
            
        except queue.Full:
            with self.lock:
                self.stats['dropped_items'] += 1
            return False
    
    def get_broadcast_item(self, timeout: Optional[float] = None) -> Optional[BroadcastItem]:
        """获取广播项目"""
        try:
            item = self.queue.get(timeout=timeout)
            with self.lock:
                self.stats['total_get'] += 1
            return item
        except queue.Empty:
            return None
    
    def drop_incomplete_broadcasts(self, max_age_seconds: float = 30) -> List[str]:
        """丢弃不完整的过期广播稿
        
        Returns:
            List[str]: 被丢弃的广播稿ID列表
        """
        incomplete_ids = self.tracker.get_incomplete_broadcasts(max_age_seconds)
        
        if incomplete_ids:
            # 从队列中移除这些广播稿的项目
            self._remove_broadcasts_from_queue(incomplete_ids)
            self.tracker.remove_broadcasts(incomplete_ids)
            
            with self.lock:
                self.stats['dropped_broadcasts'] += len(incomplete_ids)
        
        return incomplete_ids
    
    def _remove_broadcasts_from_queue(self, broadcast_ids: List[str]):
        """从队列中移除指定广播稿的所有项目"""
        # 创建临时队列来过滤项目
        temp_items = []
        removed_count = 0
        
        # 取出所有项目
        while True:
            try:
                item = self.queue.get_nowait()
                if item.broadcast_id not in broadcast_ids:
                    temp_items.append(item)
                else:
                    removed_count += 1
            except queue.Empty:
                break
        
        # 将保留的项目放回队列
        for item in temp_items:
            try:
                self.queue.put_nowait(item)
            except queue.Full:
                # 如果队列满了，丢弃一些项目
                break
        
        if removed_count > 0:
            print(f"Removed {removed_count} items from {self.name} queue for {len(broadcast_ids)} broadcasts")
    
    def get_pressure_info(self) -> Dict:
        """获取队列压力信息"""
        current_size = self.queue.qsize()
        pressure_ratio = current_size / self.maxsize if self.maxsize > 0 else 0
        
        tracker_stats = self.tracker.get_stats()
        
        with self.lock:
            stats_copy = self.stats.copy()
        
        return {
            'name': self.name,
            'queue_size': current_size,
            'queue_maxsize': self.maxsize,
            'pressure_ratio': pressure_ratio,
            'pressure_level': self._get_pressure_level(pressure_ratio),
            'tracker_stats': tracker_stats,
            'queue_stats': stats_copy
        }
    
    def _get_pressure_level(self, ratio: float) -> str:
        """获取压力等级描述"""
        if ratio < 0.3:
            return "low"
        elif ratio < 0.6:
            return "medium"
        elif ratio < 0.8:
            return "high"
        else:
            return "critical"
    
    # 移除了压力检测和丢弃逻辑
    
    # 移除了丢弃完整广播稿的方法
    
    # 移除了获取完整广播稿的辅助方法
    
    def get_pressure_level(self) -> float:
        """获取队列压力级别（0.0-1.0）"""
        current_size = self.queue.qsize()
        return current_size / self.maxsize if self.maxsize > 0 else 0.0
    
    def get_active_broadcasts(self) -> List[str]:
        """获取活跃广播稿ID列表"""
        with self.tracker.lock:
            return list(self.tracker.broadcasts.keys())
    
    def get_semantic_status(self) -> Dict:
        """获取语义状态信息"""
        return self.tracker.get_stats()
    
    def cleanup_expired_items(self, max_age_seconds: float = 60):
        """清理过期项目"""
        dropped_broadcasts = self.drop_incomplete_broadcasts(max_age_seconds)
        if dropped_broadcasts:
            print(f"Cleaned up {len(dropped_broadcasts)} expired broadcasts from {self.name}")
        return dropped_broadcasts
    
    def qsize(self) -> int:
        """获取队列大小"""
        return self.queue.qsize()
    
    def empty(self) -> bool:
        """检查队列是否为空"""
        return self.queue.empty()
    
    def full(self) -> bool:
        """检查队列是否已满"""
        return self.queue.full()


def create_broadcast_id() -> str:
    """创建新的广播稿ID"""
    return str(uuid.uuid4())


def create_script_item(broadcast_id: str, content: str, priority: int = 0) -> BroadcastItem:
    """创建脚本项目"""
    return BroadcastItem(
        broadcast_id=broadcast_id,
        content=content,
        item_type='script',
        sequence_number=0,
        total_items=1,
        is_last=True,
        priority=priority
    )


def create_wav_item(broadcast_id: str, wav_data, sequence_number: int, 
                   total_items: Optional[int] = None, is_last: bool = False,
                   priority: int = 0) -> BroadcastItem:
    """创建WAV项目"""
    return BroadcastItem(
        broadcast_id=broadcast_id,
        content=wav_data,
        item_type='wav',
        sequence_number=sequence_number,
        total_items=total_items,
        is_last=is_last,
        priority=priority
    )


def create_mp3_item(broadcast_id: str, mp3_data: bytes, sequence_number: int,
                   total_items: Optional[int] = None, is_last: bool = False, 
                   priority: int = 0) -> BroadcastItem:
    """创建MP3项目"""
    return BroadcastItem(
        broadcast_id=broadcast_id,
        content=mp3_data,
        item_type='mp3_chunk',
        sequence_number=sequence_number,
        total_items=total_items,
        is_last=is_last,
        priority=priority
    )