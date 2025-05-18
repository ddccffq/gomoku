import os
import torch
import pickle
import tempfile
import datetime
import shutil

class ModelSaver:
    """模型保存辅助工具，提供多种保存方法和错误处理"""
    
    def __init__(self, primary_dir=None, log_func=print):
        """初始化保存工具
        
        Args:
            primary_dir: 主要保存目录
            log_func: 日志记录函数
        """
        self.primary_dir = primary_dir
        self.log_func = log_func
        
        # 创建临时保存目录
        self.temp_dir = os.path.join(
            tempfile.gettempdir(), 
            f"model_save_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # 确保目录存在
        if primary_dir:
            os.makedirs(primary_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        self.log_func(f"ModelSaver初始化 - 主目录: {primary_dir}, 临时目录: {self.temp_dir}")
    
    def save_model(self, model, filename, metadata=None):
        """保存模型，尝试多种方法
        
        Args:
            model: 要保存的PyTorch模型
            filename: 文件名（不含路径）
            metadata: 可选的元数据字典
        
        Returns:
            bool: 是否成功保存
            str: 成功保存的文件路径，如果失败则为None
        """
        # 保存目标路径
        primary_path = None
        if self.primary_dir:
            primary_path = os.path.join(self.primary_dir, filename)
        
        temp_path = os.path.join(self.temp_dir, filename)
        
        # 尝试保存方式1: 主目录torch.save
        if primary_path:
            try:
                torch.save(model.state_dict(), primary_path)
                self.log_func(f"? 模型已保存到主目录: {primary_path}")
                
                # 检查文件是否真的存在
                if os.path.exists(primary_path) and os.path.getsize(primary_path) > 0:
                    return True, primary_path
                else:
                    self.log_func(f"?? 文件已保存但大小为0或不存在: {primary_path}")
            except Exception as e:
                self.log_func(f"? 无法保存到主目录: {str(e)}")
        
        # 尝试保存方式2: 临时目录torch.save
        try:
            torch.save(model.state_dict(), temp_path)
            self.log_func(f"? 模型已保存到临时目录: {temp_path}")
            
            # 检查文件
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                # 如果主目录保存失败但临时目录成功，尝试复制
                if primary_path and not os.path.exists(primary_path):
                    try:
                        shutil.copy2(temp_path, primary_path)
                        self.log_func(f"? 已从临时目录复制到主目录")
                        return True, primary_path
                    except Exception as e:
                        self.log_func(f"?? 复制到主目录失败: {str(e)}")
                
                return True, temp_path
        except Exception as e:
            self.log_func(f"? 保存到临时目录失败: {str(e)}")
        
        # 尝试保存方式3: pickle
        pickle_path = temp_path + ".pickle"
        try:
            with open(pickle_path, 'wb') as f:
                pickle.dump(model, f)
            self.log_func(f"? 模型已通过pickle保存: {pickle_path}")
            return True, pickle_path
        except Exception as e:
            self.log_func(f"? pickle保存失败: {str(e)}")
        
        # 所有方法都失败了
        return False, None
    
    def cleanup(self):
        """清理临时文件"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                self.log_func(f"清理了临时目录: {self.temp_dir}")
        except Exception as e:
            self.log_func(f"清理临时目录失败: {str(e)}")
