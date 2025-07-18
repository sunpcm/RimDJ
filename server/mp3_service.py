import asyncio
import numpy as np
import lameenc
import config
from semantic_queue import create_mp3_item
from base_service import AsyncServiceBase, safe_get_item, safe_put_item, print_processing_info


class MP3Service(AsyncServiceBase):
    """MP3编码服务"""
    
    def __init__(self, mixed_wav_queue, broadcast_mp3_queue):
        super().__init__("MP3")
        self.mixed_wav_queue = mixed_wav_queue  # 混合后的WAV队列
        self.broadcast_mp3_queue = broadcast_mp3_queue  # 最终广播的MP3队列
        
        # 注意：lameenc编码器在flush后无法重复使用，因此每次都创建新实例
    
    async def process_item(self) -> bool:
        """处理混合WAV项目并编码为MP3"""
        # 检查输入队列状态
        queue_size = self.mixed_wav_queue.qsize()
        queue_maxsize = self.mixed_wav_queue.maxsize
        # print(f"MP3Service: 检查输入队列 - 当前大小: {queue_size}/{queue_maxsize}")
        
        # 获取混合WAV项目
        wav_item = safe_get_item(self.mixed_wav_queue)
        if wav_item is None:
            # print(f"MP3Service: 输入队列为空，等待数据...")
            return False
        
        wav_np = wav_item.content
        broadcast_id = wav_item.broadcast_id
        
        print(f"MP3Service: 获取到WAV项目:")
        print(f"  - 广播ID: {broadcast_id}")
        print(f"  - 序号: {wav_item.sequence_number}")
        print(f"  - 是否最后: {wav_item.is_last}")
        print(f"  - 优先级: {wav_item.priority}")
        
        if wav_np is None:
            print(f"MP3Service: WAV数据为None，跳过处理")
            return True
        
        if wav_np.size == 0:
            print(f"MP3Service: WAV数据为空数组，跳过处理")
            return True
        
        # 彻底关闭BGM相关的日志输出
        if "bgm_chunk_" not in broadcast_id:
            # 只为TTS和其他音频输出日志，完全跳过BGM日志
            print_processing_info("MP3", broadcast_id, f"encoding sequence {wav_item.sequence_number}")
        
        try:
            # 在线程池中运行MP3编码
            loop = asyncio.get_event_loop()
            mp3_chunks = await asyncio.wait_for(
                loop.run_in_executor(None, self._encode_wav_to_mp3, wav_np, wav_item.is_last),
                timeout=config.MP3_TIMEOUT
            )
            
            # 处理编码结果
            if mp3_chunks:
                await self._queue_mp3_chunks(wav_item, mp3_chunks)
            else:
                print_processing_info("MP3", broadcast_id, "encoding returned empty chunks")
                
        except asyncio.TimeoutError:
            print_processing_info("MP3", broadcast_id, "encoding timeout")
        except Exception as e:
            print_processing_info("MP3", broadcast_id, f"encoding failed: {e}")
        
        return True
    
    async def _queue_mp3_chunks(self, wav_item, mp3_chunks):
        """将MP3块放入队列"""
        total_chunks = len(mp3_chunks)
        base_sequence = wav_item.sequence_number * 1000  # 为MP3块预留序号空间
        
        print(f"MP3Service: 开始队列化MP3块:")
        print(f"  - 总块数: {total_chunks}")
        print(f"  - 基础序号: {base_sequence}")
        print(f"  - 广播ID: {wav_item.broadcast_id}")
        print(f"  - 是否最后WAV: {wav_item.is_last}")
        print(f"  - 当前队列大小: {self.broadcast_mp3_queue.qsize()}/{self.broadcast_mp3_queue.maxsize}")
        
        successful_chunks = 0
        for i, mp3_data in enumerate(mp3_chunks):
            if not mp3_data or len(mp3_data) == 0:
                print(f"  - 跳过空块#{i}")
                continue
            
            mp3_item = create_mp3_item(
                broadcast_id=wav_item.broadcast_id,
                mp3_data=mp3_data,
                sequence_number=base_sequence + i,
                is_last=wav_item.is_last and (i == total_chunks - 1),
                priority=wav_item.priority
            )
            
            # print(f"  - 队列化块#{i}: 大小={len(mp3_data)}字节, 序号={base_sequence + i}, 最后块={mp3_item.is_last}")
            
            success = safe_put_item(self.broadcast_mp3_queue, mp3_item, timeout=5.0)
            if success:
                successful_chunks += 1
                # print(f"    ✓ 成功入队")
            else:
                # print(f"    ✗ 入队失败 - 队列可能已满或超时")
                print_processing_info("MP3", wav_item.broadcast_id, f"failed to queue chunk {i}")
                break
        
        print(f"MP3Service: 队列化完成 - 成功: {successful_chunks}/{total_chunks}")
        
        # 检查队列状态
        queue_pressure = self.broadcast_mp3_queue.get_pressure_level() if hasattr(self.broadcast_mp3_queue, 'get_pressure_level') else "未知"
        print(f"  - 队列压力等级: {queue_pressure}")
        print(f"  - 队列使用率: {self.broadcast_mp3_queue.qsize()}/{self.broadcast_mp3_queue.maxsize} ({self.broadcast_mp3_queue.qsize()/self.broadcast_mp3_queue.maxsize*100:.1f}%)")
    
    def _encode_wav_to_mp3(self, wav_np, is_last_wav=False):
        """编码WAV数据为MP3块 - 在线程池中运行"""
        try:
            # 添加调试信息：检查输入数据
            if hasattr(self, '_debug_counter'):
                self._debug_counter += 1
            else:
                self._debug_counter = 1
            
            # 增加更详细的调试信息，每次都输出关键参数
            input_stats = {
                'shape': wav_np.shape,
                'dtype': wav_np.dtype,
                'min': np.min(wav_np),
                'max': np.max(wav_np),
                'mean': np.mean(wav_np),
                'std': np.std(wav_np),
                'zero_count': np.sum(wav_np == 0),
                'total_samples': len(wav_np),
                'non_zero_count': np.sum(wav_np != 0)
            }
            print(f"MP3Service: [编码#{self._debug_counter}] 输入WAV数据统计:")
            print(f"  - 形状: {input_stats['shape']}, 类型: {input_stats['dtype']}")
            print(f"  - 数值范围: [{input_stats['min']:.6f}, {input_stats['max']:.6f}]")
            print(f"  - 均值: {input_stats['mean']:.6f}, 标准差: {input_stats['std']:.6f}")
            print(f"  - 零值样本: {input_stats['zero_count']}/{input_stats['total_samples']} ({input_stats['zero_count']/input_stats['total_samples']*100:.1f}%)")
            print(f"  - 非零样本: {input_stats['non_zero_count']} ({input_stats['non_zero_count']/input_stats['total_samples']*100:.1f}%)")
            
            # 检查是否为静音数据
            if input_stats['non_zero_count'] == 0:
                print(f"MP3Service: 警告 - 输入数据全为零值（静音）！")
            elif input_stats['std'] < 1e-6:
                print(f"MP3Service: 警告 - 输入数据标准差极小，可能为静音或常数！")
            
            # 确保数据类型为int16
            if wav_np.dtype == np.float32 or wav_np.dtype == np.float64:
                # 浮点数据，范围应该在[-1.0, 1.0]
                wav_np = np.clip(wav_np, -1.0, 1.0)
                data = (wav_np * 32767).astype(np.int16)
            elif wav_np.dtype == np.int16:
                # 已经是int16格式
                data = wav_np
            else:
                # 其他格式，先转换为float再处理
                wav_np = wav_np.astype(np.float32)
                wav_np = np.clip(wav_np, -1.0, 1.0)
                data = (wav_np * 32767).astype(np.int16)
            
            # 保持单声道格式，不进行立体声转换
            print(f"MP3Service: 单声道数据统计:")
            print(f"  - 数据形状: {data.shape}, 类型: {data.dtype}")
            print(f"  - 数值范围: [{np.min(data)}, {np.max(data)}]")
            print(f"  - 预期chunk大小: {config.CHUNK_SIZE} (单声道)")
            
            mp3_chunks = self._create_mp3_chunks(data, is_last_wav)
            
            # 添加详细的输出数据检查
            total_mp3_size = sum(len(chunk) for chunk in mp3_chunks)
            print(f"MP3Service: 输出MP3数据统计:")
            print(f"  - MP3块数: {len(mp3_chunks)}, 总大小: {total_mp3_size} 字节")
            
            if mp3_chunks:
                for i, chunk in enumerate(mp3_chunks[:3]):  # 显示前3个块的信息
                    if len(chunk) > 0:
                        hex_preview = ' '.join(f'{b:02X}' for b in chunk[:16])
                        print(f"  - 块#{i}: 大小={len(chunk)}字节, 前16字节: {hex_preview}")
                        
                        # 检查MP3帧头
                        if len(chunk) >= 4:
                            frame_sync = (chunk[0] << 8) | chunk[1]
                            if (frame_sync & 0xFFE0) == 0xFFE0:
                                print(f"    ✓ 有效MP3帧同步头: 0x{frame_sync:04X}")
                            else:
                                print(f"    ✗ 无效MP3帧同步头: 0x{frame_sync:04X}")
                    else:
                        print(f"  - 块#{i}: 空块！")
                        
                if len(mp3_chunks) > 3:
                    print(f"  - ... 还有 {len(mp3_chunks) - 3} 个块")
            else:
                print(f"MP3Service: 严重错误 - 生成的MP3数据为空！")
            
            return mp3_chunks
            
        except Exception as e:
            print(f"MP3 encoding error: {e}")
            return []
    
    def _get_encoder(self):
        """获取新的编码器实例"""
        # 每次都创建新编码器，因为lameenc编码器在flush后无法重复使用
        try:
            encoder = lameenc.Encoder()
            
            # 设置编码器参数并记录 - 使用单声道配置
            bit_rate = 128
            sample_rate = config.SAMPLE_RATE
            channels = 1  # 改为单声道
            quality = 2
            
            encoder.set_bit_rate(bit_rate)
            encoder.set_in_sample_rate(sample_rate)
            encoder.set_channels(channels)
            encoder.set_quality(quality)
            
            print(f"MP3Service: 编码器配置:")
            print(f"  - 比特率: {bit_rate} kbps")
            print(f"  - 采样率: {sample_rate} Hz")
            print(f"  - 声道数: {channels} (单声道)")
            print(f"  - 质量等级: {quality} (0=最高质量, 9=最低质量)")
            
            return encoder
        except Exception as e:
            print(f"MP3Service: 创建编码器失败: {e}")
            return None
    
    def _return_encoder(self, encoder):
        """清理编码器资源（不再复用）"""
        # 不再复用编码器，直接丢弃
        pass
    
    def _create_mp3_chunks(self, data, is_last_wav=False):
        """创建MP3块 - 一次性编码然后分块输出"""
        mp3_chunks = []
        encoder = self._get_encoder()
        
        if not encoder:
            print(f"MP3Service: 编码器创建失败，无法处理数据")
            return []
        
        try:
            # 一次性编码所有数据
            print(f"MP3Service: 开始一次性编码:")
            print(f"  - 输入数据长度: {len(data)} 样本")
            print(f"  - PCM字节数: {len(data) * 2}")
            
            # 将所有数据编码为MP3
            pcm_data = data.tobytes()
            mp3_data = encoder.encode(pcm_data)
            
            # 刷新编码器获取剩余数据
            final_data = encoder.flush()
            
            # 合并编码数据和flush数据
            complete_mp3_data = b''
            if mp3_data:
                complete_mp3_data += mp3_data
            if final_data:
                complete_mp3_data += final_data
            
            print(f"  - 编码完成: 总MP3字节数={len(complete_mp3_data)}")
            
            if not complete_mp3_data:
                print(f"  ✗ 编码失败: 无MP3数据生成")
                return []
            
            # 过滤和验证完整的MP3数据
            filtered_mp3_data = self._filter_mp3_data(complete_mp3_data)
            if not filtered_mp3_data:
                print(f"  ✗ MP3数据过滤后为空")
                return []
            
            print(f"  ✓ 过滤后MP3数据: {len(filtered_mp3_data)} 字节")
            
            # 将MP3数据分块以适应流式传输
            # 每个chunk大约1秒的音频数据（约2KB）
            chunk_size = 2048  # 2KB per chunk
            total_chunks = (len(filtered_mp3_data) + chunk_size - 1) // chunk_size
            
            print(f"MP3Service: 开始分块输出:")
            print(f"  - MP3数据长度: {len(filtered_mp3_data)} 字节")
            print(f"  - 每块大小: {chunk_size} 字节")
            print(f"  - 预计块数: {total_chunks}")
            
            for i in range(0, len(filtered_mp3_data), chunk_size):
                chunk = filtered_mp3_data[i:i + chunk_size]
                chunk_num = i // chunk_size
                
                if len(chunk) > 0:
                    mp3_chunks.append(chunk)
                    # print(f"  - 输出块#{chunk_num}: {len(chunk)} 字节")
                    
                    # 检查前几个块的MP3帧头
                    if chunk_num < 3 and len(chunk) >= 4:
                        frame_sync = (chunk[0] << 8) | chunk[1]
                        if (frame_sync & 0xFFE0) == 0xFFE0:
                            print(f"    ✓ 有效MP3帧: 0x{frame_sync:04X}")
                        else:
                            print(f"    - 数据块: 0x{frame_sync:04X} (可能是帧中间部分)")
            
            print(f"MP3Service: 分块完成 - 总块数: {len(mp3_chunks)}")
                
        except Exception as e:
            print(f"Error in MP3 encoding: {e}")
        finally:
            # 将编码器返回池中复用
            self._return_encoder(encoder)
        
        return mp3_chunks
    
    def _filter_mp3_data(self, mp3_data):
        """过滤和验证MP3数据，移除编码器元数据"""
        if not mp3_data or len(mp3_data) < 4:
            return mp3_data
        
        try:
            # 检查是否为有效的MP3帧
            frame_sync = (mp3_data[0] << 8) | mp3_data[1]
            
            # MP3帧同步头应该是0xFFE0到0xFFFF之间
            if (frame_sync & 0xFFE0) == 0xFFE0:
                print(f"    ✓ 有效MP3帧: 0x{frame_sync:04X}")
                return mp3_data
            
            # 检查是否为纯LAME元数据头（只有4字节的LAME标识）
            if len(mp3_data) == 4 and mp3_data == b'LAME':
                print(f"    - 过滤纯LAME标识: {mp3_data.hex()}")
                return b''  # 过滤掉纯元数据
            
            # 检查是否包含LAME标识但可能有有效MP3数据
            if len(mp3_data) >= 4:
                header_bytes = mp3_data[:4]
                # 如果是4C414D45（LAME的十六进制），但数据长度大于4字节，可能包含有效MP3数据
                if (header_bytes[0] == 0x4C and header_bytes[1] == 0x41 and 
                    header_bytes[2] == 0x4D and header_bytes[3] == 0x45):
                    if len(mp3_data) > 4:
                        # 查找LAME标识后的有效MP3帧
                        for i in range(4, min(len(mp3_data) - 1, 64)):
                            if i + 1 < len(mp3_data):
                                sync = (mp3_data[i] << 8) | mp3_data[i + 1]
                                if (sync & 0xFFE0) == 0xFFE0:
                                    print(f"    ✓ LAME后找到MP3帧: 0x{sync:04X}，偏移{i}")
                                    return mp3_data[i:]  # 返回从有效帧开始的数据
                        print(f"    - LAME数据无有效帧: {header_bytes.hex()}")
                        return b''  # 没找到有效帧，过滤掉
                    else:
                        print(f"    - 过滤LAME元数据: {header_bytes.hex()}")
                        return b''  # 过滤掉元数据
            
            # 检查是否为其他编码器标识或损坏的数据
            if len(mp3_data) >= 8:
                # 查找可能的MP3帧开始位置
                for i in range(min(len(mp3_data) - 1, 64)):  # 只检查前64字节
                    if i + 1 < len(mp3_data):
                        sync = (mp3_data[i] << 8) | mp3_data[i + 1]
                        if (sync & 0xFFE0) == 0xFFE0:
                            print(f"    ✓ 在偏移{i}找到MP3帧: 0x{sync:04X}")
                            return mp3_data[i:]  # 返回从有效帧开始的数据
            
            print(f"    ✗ 无效MP3数据: 0x{frame_sync:04X}")
            return b''  # 过滤掉无效数据
            
        except Exception as e:
            print(f"    ✗ MP3数据过滤错误: {e}")
            return mp3_data  # 出错时返回原数据


def start_mp3_service(mixed_wav_queue, broadcast_mp3_queue):
    """启动MP3服务"""
    service = MP3Service(mixed_wav_queue, broadcast_mp3_queue)
    return service.start()