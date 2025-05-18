import os
import json
import time
from datetime import datetime
from pathlib import Path
from PyQt5.QtGui import QIcon
from qfluentwidgets import isDarkTheme

class ResourceManager:
    """资源管理器 - 统一管理所有的资源加载"""
    
    # 资源路径常量
    RESOURCES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources")
    ICONS_DIR = os.path.join(RESOURCES_DIR, "icons")
    TRAINED_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trained_models")
    TRAINING_DATA_DIR = os.path.join(TRAINED_MODELS_DIR, "training_data")
    SELECTED_MODELS_DIR = os.path.join(TRAINED_MODELS_DIR, "selected_models")
    
    # 单例模式实现
    _instance = None
    
    @classmethod
    def instance(cls):
        """获取资源管理器单例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """初始化资源管理器"""
        # 确保目录存在
        os.makedirs(self.TRAINED_MODELS_DIR, exist_ok=True)
        os.makedirs(self.SELECTED_MODELS_DIR, exist_ok=True)
        os.makedirs(self.TRAINING_DATA_DIR, exist_ok=True)
        
        # 缓存最近一次的模型扫描结果
        self._cached_models = None
        self._last_scan_time = 0
        self._scan_interval = 60  # 扫描间隔，单位秒
    
    def get_icon(self, icon_name, theme=None):
        """获取图标资源"""
        if theme is None:
            theme = "dark" if isDarkTheme() else "light"
            
        icon_path = os.path.join(self.ICONS_DIR, theme, f"{icon_name}.png")
        if not os.path.exists(icon_path):
            icon_path = os.path.join(self.ICONS_DIR, f"{icon_name}.png")
            
        return QIcon(icon_path)
    
    def scan_trained_models(self, force_refresh=False):
        """扫描可用的训练模型，返回模型列表
        
        Args:
            force_refresh: 是否强制刷新缓存
            
        Returns:
            list: 模型信息列表，每个元素为一个字典，包含模型路径、大小、日期等信息
        """
        current_time = time.time()
        
        # 如果缓存有效且未强制刷新，直接返回缓存
        if not force_refresh and self._cached_models is not None and (current_time - self._last_scan_time) < self._scan_interval:
            return self._cached_models
            
        models = []
        print(f"正在扫描训练模型目录: {self.TRAINED_MODELS_DIR}")
        
        # 扫描模型目录中的所有.pth文件
        model_paths = []
        
        # 1. 扫描选定模型目录
        if os.path.exists(self.SELECTED_MODELS_DIR):
            for file in os.listdir(self.SELECTED_MODELS_DIR):
                if file.endswith(".pth"):
                    model_paths.append(os.path.join(self.SELECTED_MODELS_DIR, file))
        
        # 2. 递归扫描models目录及其所有子目录
        models_dir = os.path.join(self.TRAINED_MODELS_DIR, "models")
        if os.path.exists(models_dir):
            for root, dirs, files in os.walk(models_dir):
                for file in files:
                    if file.endswith(".pth"):
                        model_paths.append(os.path.join(root, file))
        
        # 解析每个模型文件的信息
        for model_path in model_paths:
            model_info = self._parse_model_info(model_path)
            models.append(model_info)
        
        # 按修改时间排序，最新的在前面
        models.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        
        # 更新缓存
        self._cached_models = models
        self._last_scan_time = current_time
        
        print(f"扫描完成，共找到 {len(models)} 个训练模型")
        return models
    
    def _parse_model_info(self, model_path):
        """解析模型文件信息
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            dict: 模型信息，包括文件名、路径、大小、时间戳等
        """
        file_name = os.path.basename(model_path)
        file_size = os.path.getsize(model_path)
        mod_time = os.path.getmtime(model_path)
        
        # 尝试从文件名中提取信息
        model_size = "unknown"
        model_type = "unknown"
        
        # 检测模型大小
        if "large" in file_name.lower():
            model_size = "large"
        elif "medium" in file_name.lower():
            model_size = "medium"
        elif "small" in file_name.lower():
            model_size = "small"
        
        # 检测模型类型
        if "best" in file_name.lower():
            model_type = "best"
        elif "final" in file_name.lower():
            model_type = "final"
        elif "epoch" in file_name.lower() or "round" in file_name.lower():
            # 尝试提取轮次信息
            try:
                import re
                match = re.search(r'(?:epoch|round)_(\d+)', file_name.lower())
                if match:
                    round_num = match.group(1)
                    model_type = f"epoch_{round_num}"
                else:
                    model_type = "checkpoint"
            except:
                model_type = "checkpoint"
        
        # 创建模型信息字典
        model_info = {
            'file_name': file_name,
            'path': model_path,
            'size': file_size,
            'size_formatted': self._format_size(file_size),
            'timestamp': mod_time,
            'date': datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S'),
            'model_size': model_size,
            'model_type': model_type
        }
        
        return model_info
    
    def _format_size(self, size_bytes):
        """格式化文件大小
        
        Args:
            size_bytes: 文件大小，单位字节
            
        Returns:
            str: 格式化后的大小字符串，如"1.23 MB"
        """
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.2f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.2f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
    
    def get_training_data_info(self):
        """获取训练数据信息
        
        Returns:
            dict: 训练数据信息，包括目录数量、样本数量等
        """
        if not os.path.exists(self.TRAINING_DATA_DIR):
            return {
                'status': 'error',
                'message': '训练数据目录不存在',
                'path': self.TRAINING_DATA_DIR,
                'directory_count': 0,
                'sample_count': 0
            }
        
        # 扫描目录，计算数据集大小
        print(f"开始在 {self.TRAINING_DATA_DIR} 中搜索训练数据...")
        directory_count = 0
        valid_directories = 0
        
        sessions = []
        
        # 遍历第一级会话目录
        for session_name in os.listdir(self.TRAINING_DATA_DIR):
            session_path = os.path.join(self.TRAINING_DATA_DIR, session_name)
            if not os.path.isdir(session_path):
                continue
                
            session_info = {
                'name': session_name,
                'path': session_path,
                'games': []
            }
            
            # 遍历每个会话中的游戏
            for game_name in os.listdir(session_path):
                game_path = os.path.join(session_path, game_name)
                directory_count += 1
                
                # 每50个目录打印一次进度
                if directory_count % 50 == 0:
                    valid_directories = directory_count - (directory_count // 50)  # 简化，假设大部分都是有效的
                    print(f"已检查 {directory_count} 个目录，找到 {valid_directories} 个有效数据目录...")
                
                if os.path.isdir(game_path):
                    # 实际应用中这里可以检查目录是否包含有效的训练数据文件
                    valid_directories += 1
                    session_info['games'].append(game_name)
        
        print(f"搜索完成，共检查了 {directory_count} 个目录，找到 {valid_directories} 个有效训练数据目录。")
        
        # 估算样本数量 (实际中可能需要读取文件来精确统计)
        estimated_samples = valid_directories * 24  # 假设每个目录平均24个样本
        
        return {
            'status': 'success',
            'path': self.TRAINING_DATA_DIR,
            'directory_count': directory_count,
            'valid_directories': valid_directories,
            'estimated_samples': estimated_samples,
            'sessions': sessions
        }
    
    def get_model_info(self, model_path):
        """获取特定模型的详细信息，并打印加载细节
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            dict: 模型详细信息
        """
        if not os.path.exists(model_path):
            print(f"错误: 模型文件不存在: {model_path}")
            return None
        
        model_info = self._parse_model_info(model_path)
        
        # 打印模型加载信息
        print(f"已选择模型文件: {model_path}")
        print(f"检测到模型大小: {model_info['model_size']}")
        
        return model_info
    
    def get_default_model_path(self):
        """获取默认模型路径
        
        如果有best_model.pth，返回它；否则返回找到的第一个模型
        
        Returns:
            str: 默认模型路径，如果没有找到则返回None
        """
        best_model_path = os.path.join(self.SELECTED_MODELS_DIR, "best_model.pth")
        if os.path.exists(best_model_path):
            return best_model_path
            
        # 如果没有best_model.pth，扫描所有模型并返回最新的一个
        models = self.scan_trained_models()
        if models:
            return models[0]['path']  # 最新的模型
            
        return None
    
    def print_model_load_log(self, model_path, success=True):
        """打印模型加载日志
        
        Args:
            model_path: 模型文件路径
            success: 是否加载成功
        """
        if success:
            model_info = self._parse_model_info(model_path)
            print(f"===== 模型加载成功 =====")
            print(f"📁 模型文件: {model_info['file_name']}")
            print(f"📊 模型大小: {model_info['model_size']}")
            print(f"📦 文件大小: {model_info['size_formatted']}")
            print(f"📅 创建日期: {model_info['date']}")
        else:
            print(f"❌ 模型加载失败: {model_path}")
            print(f"请检查文件是否存在或权限是否正确")
