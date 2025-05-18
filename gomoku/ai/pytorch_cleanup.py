"""
PyTorch清理辅助模块，提供安全终止训练和清理资源的函数
"""

import gc
import os
import sys
import time
import threading
import traceback

def cleanup_pytorch_resources(verbose=True):
    """清理PyTorch资源，包括模型、优化器、CUDA缓存等
    
    Args:
        verbose: 是否输出详细信息
    """
    try:
        # 触发垃圾收集
        if verbose:
            print("正在运行垃圾回收...")
        gc.collect()
        
        # 清理PyTorch资源
        try:
            import torch
            
            # 清空CUDA缓存
            if torch.cuda.is_available():
                if verbose:
                    print("正在清空CUDA缓存...")
                torch.cuda.empty_cache()
                
                # 重置设备
                for device_id in range(torch.cuda.device_count()):
                    try:
                        torch.cuda.set_device(device_id)
                        torch.cuda.empty_cache()
                        if verbose:
                            print(f"已重置并清空GPU设备 {device_id}")
                    except Exception as e:
                        if verbose:
                            print(f"重置设备 {device_id} 出错: {e}")
                
                # 分离所有CUDA tensor
                with torch.no_grad():
                    for obj in gc.get_objects():
                        try:
                            if torch.is_tensor(obj) and obj.is_cuda:
                                obj.detach_()
                        except:
                            pass
                
                if verbose:
                    print("已分离所有CUDA张量")
        except ImportError:
            if verbose:
                print("未检测到PyTorch，跳过CUDA清理")
        except Exception as e:
            if verbose:
                print(f"清理PyTorch资源时出错: {e}")
                if verbose > 1:
                    traceback.print_exc()
    
    except Exception as e:
        if verbose:
            print(f"全局资源清理时出错: {e}")
            if verbose > 1:
                traceback.print_exc()

def interrupt_training_threads(timeout=5.0, verbose=True):
    """中断所有训练线程并等待它们完成
    
    Args:
        timeout: 等待线程终止的最大时间（秒）
        verbose: 是否输出详细信息
    """
    try:
        from mainWindow.training_interface import stop_all_training_threads
        
        if verbose:
            print(f"正在停止所有训练线程，超时时间: {timeout}秒...")
        
        # 停止所有训练线程
        stop_all_training_threads()
        
        # 等待指定时间
        start_time = time.time()
        while time.time() - start_time < timeout:
            # 检查是否还有活跃的TrainingThread
            active_threads = 0
            for thread in threading.enumerate():
                if thread.name.startswith('TrainingThread') or 'Training' in thread.__class__.__name__:
                    active_threads += 1
            
            if active_threads == 0:
                if verbose:
                    print("所有训练线程已终止")
                break
                
            if verbose and active_threads > 0:
                print(f"仍有 {active_threads} 个训练线程活跃，继续等待...")
            
            # 短暂休眠
            time.sleep(0.5)
        
        # 检查最终状态
        active_count = sum(1 for t in threading.enumerate() 
                         if t.name.startswith('TrainingThread') or 'Training' in t.__class__.__name__)
        
        if active_count > 0 and verbose:
            print(f"警告: 在超时后仍有 {active_count} 个训练线程活跃")
            
        # 再次尝试清理PyTorch资源
        cleanup_pytorch_resources(verbose)
            
        return active_count == 0
    
    except Exception as e:
        if verbose:
            print(f"中断训练线程时出错: {e}")
            if verbose > 1:
                traceback.print_exc()
        return False

if __name__ == "__main__":
    # 如果直接运行，则执行清理
    print("执行PyTorch资源清理...")
    cleanup_pytorch_resources(verbose=True)
    print("清理完成")
