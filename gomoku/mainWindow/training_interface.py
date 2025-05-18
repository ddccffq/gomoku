from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer, QSize, QUrl, QWaitCondition, QMutex, QObject
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog,
                           QGroupBox, QProgressBar, QSpacerItem, QSizePolicy, QGridLayout, 
                           QMessageBox, QApplication, QStackedWidget, QTextEdit, QDoubleSpinBox)
from PyQt5.QtGui import QFont, QClipboard, QDesktopServices

from qfluentwidgets import (InfoBar, InfoBarPosition, FluentIcon as FIF, 
                          PushButton, LineEdit, ComboBox, SpinBox, CheckBox,
                          ProgressBar, CardWidget, ScrollArea, TitleLabel,
                          BodyLabel, StrongBodyLabel, SubtitleLabel, CaptionLabel,
                          PrimaryPushButton, TransparentToolButton, TextEdit, SearchLineEdit)

import os
import time
import json
import threading
import datetime
import glob
import shutil
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import weakref

# 全局训练线程注册表
_training_threads = []
_threads_mutex = QMutex()

def register_training_thread(thread):
    """注册训练线程以便全局管理"""
    global _training_threads
    _threads_mutex.lock()
    try:
        _training_threads.append(weakref.ref(thread))
        # 清理失效的引用
        _training_threads = [t for t in _training_threads if t() is not None]
    finally:
        _threads_mutex.unlock()

def stop_all_training_threads():
    """停止所有活动的训练线程"""
    global _training_threads
    _threads_mutex.lock()
    try:
        active_threads = [t() for t in _training_threads if t() is not None]
        for thread in active_threads:
            if thread.isRunning():
                print(f"正在停止训练线程: {thread}")
                thread.requestInterruption()  # 请求线程中断
                thread.stop_training()        # 调用自定义的停止方法
                
                # 减少等待时间，使主线程响应更快
                thread.wait(500)             # 等待最多500ms让线程自行终止
                
                # 如果线程仍然运行，将由线程内部的计时器处理强制终止
        
        # 清理失效的引用
        _training_threads = [t for t in _training_threads if t() is not None and t().isRunning()]
    finally:
        _threads_mutex.unlock()


class EnhancedLogWidget(QWidget):
    """增强型日志显示组件，支持滚动、复制和搜索"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 主布局
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(10)
        
        # 创建搜索和控制按钮布局
        self.control_layout = QHBoxLayout()
        
        # 搜索框
        self.search_edit = SearchLineEdit(self)
        self.search_edit.setPlaceholderText("搜索日志内容...")
        self.search_edit.textChanged.connect(self.search_log)
        
        # 控制按钮布局
        self.buttons_layout = QHBoxLayout()
        self.copy_button = PushButton("复制全部", self, FIF.COPY)
        self.clear_button = PushButton("清空", self, FIF.DELETE)
        self.export_button = PushButton("导出", self, FIF.SAVE_AS)
        
        # 调整按钮大小为更紧凑的样式
        for btn in [self.copy_button, self.clear_button, self.export_button]:
            btn.setFixedHeight(30)
        
        # 连接按钮信号
        self.copy_button.clicked.connect(self.copy_all)
        self.clear_button.clicked.connect(self.clear_log)
        self.export_button.clicked.connect(self.export_log)
        
        # 添加到按钮布局
        self.buttons_layout.addWidget(self.copy_button)
        self.buttons_layout.addWidget(self.clear_button)
        self.buttons_layout.addWidget(self.export_button)
        self.buttons_layout.addStretch(1)
        
        # 添加搜索和按钮到控制布局
        self.control_layout.addWidget(self.search_edit, 1)
        self.control_layout.addLayout(self.buttons_layout)
        
        # 创建日志显示区域 - 使用QTextEdit而不是QPlainTextEdit以支持富文本
        self.log_text = QTextEdit(self)
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QTextEdit.WidgetWidth)
        # 设置字体为等宽字体，提高日志可读性
        font = QFont("Consolas", 10)
        self.log_text.setFont(font)
        
        # 增加日志区域高度
        self.log_text.setMinimumHeight(400)
        
        # 添加到主布局
        self.main_layout.addLayout(self.control_layout)
        self.main_layout.addWidget(self.log_text, 1)
        
        # 自定义上下文菜单添加复制功能
        self.log_text.setContextMenuPolicy(Qt.CustomContextMenu)
        self.log_text.customContextMenuRequested.connect(self.show_context_menu)
    
    def append(self, text):
        """添加日志文本"""
        # 获取当前光标位置
        cursor = self.log_text.textCursor()
        
        # 移动到文档末尾
        cursor.movePosition(cursor.End)
        
        # 在末尾添加文本
        cursor.insertText(text + "\n")
        
        # 自动滚动到底部确保新内容可见
        self.log_text.setTextCursor(cursor)
        self.log_text.ensureCursorVisible()
    
    def copy_all(self):
        """复制所有日志文本"""
        text = self.log_text.toPlainText()
        if text:
            clipboard = QApplication.clipboard()
            clipboard.setText(text)
            
            InfoBar.success(
                title='已复制',
                content="日志内容已复制到剪贴板",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self
            )
    
    def clear_log(self):
        """清空日志"""
        self.log_text.clear()
    
    def export_log(self):
        """导出日志到文件"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出日志", "", "文本文件 (*.txt);;所有文件 (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.toPlainText())
                
                InfoBar.success(
                    title='导出成功',
                    content=f"日志已导出至: {file_path}",
                    orient=Qt.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.TOP,
                    duration=3000,
                    parent=self
                )
            except Exception as e:
                InfoBar.error(
                    title='导出失败',
                    content=f"导出日志失败: {str(e)}",
                    orient=Qt.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.TOP,
                    duration=3000,
                    parent=self
                )
    
    def search_log(self, text):
        """搜索日志内容"""
        if not text:
            # 清除搜索高亮
            cursor = self.log_text.textCursor()
            cursor.setPosition(0)
            self.log_text.setTextCursor(cursor)
            return
            
        # 从当前位置开始搜索
        cursor = self.log_text.textCursor()
        cursor.setPosition(0)  # 从头开始搜索
        
        # 使用默认搜索选项
        self.log_text.setTextCursor(cursor)
        result = self.log_text.find(text)
        
        if result:
            # 滚动到找到的位置
            self.log_text.ensureCursorVisible()
        else:
            InfoBar.warning(
                title='未找到',
                content=f"找不到: {text}",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP_RIGHT,
                duration=2000,
                parent=self
            )
    
    def show_context_menu(self, pos):
        """显示自定义上下文菜单"""
        menu = self.log_text.createStandardContextMenu()
        
        # 添加自定义项
        selected_text = self.log_text.textCursor().selectedText()
        if selected_text:
            copy_selection = menu.addAction("复制选中内容")
            copy_selection.triggered.connect(self.copy_selection)
        
        copy_all = menu.addAction("复制全部")
        copy_all.triggered.connect(self.copy_all)
        
        # 显示菜单
        menu.exec_(self.log_text.mapToGlobal(pos))
    
    def copy_selection(self):
        """复制选中文本"""
        selected_text = self.log_text.textCursor().selectedText()
        if selected_text:
            clipboard = QApplication.clipboard()
            clipboard.setText(selected_text)


class TrainingThread(QThread):
    """训练线程"""
    progress_updated = pyqtSignal(int, dict)  # 进度更新信号(进度百分比, 信息字典)
    log_message = pyqtSignal(str)  # 日志消息信号
    training_completed = pyqtSignal(bool, str)  # 训练完成信号(是否成功, 消息)
    board_updated = pyqtSignal(list, list, int)  # 棋盘数据, 历史记录, 当前玩家
    training_epoch_completed = pyqtSignal(str, int, dict)  # 模型名称, 轮次, 指标字典
    safe_info_bar_signal = pyqtSignal(str, str, str)  # (title, content, type)
    status_update_signal = pyqtSignal(str, int)  # 状态更新信号(状态文本, 进度值)
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.is_running = True
        self.is_paused = False
        self.pause_condition = QWaitCondition()
        self.mutex = QMutex()
        self.current_epoch = 0
        self.current_batch = 0
        self.current_game = 0
        self.stats = {}
        self._stop_flag = False
        self._condition = QWaitCondition()
        self._mutex = QMutex()
        self._stop_event = threading.Event()  # 使用事件对象来控制停止
        
        # 注册线程
        register_training_thread(self)
        
        # 连接安全的InfoBar显示
        self.safe_info_bar_signal.connect(self._show_safe_info_bar)
        
        # 确保输出目录存在
        if not os.path.exists(self.config['output_dir']):
            os.makedirs(self.config['output_dir'], exist_ok=True)
    
    def _show_safe_info_bar(self, title, content, info_type):
        """安全地在主线程中显示InfoBar"""
        try:
            from qfluentwidgets import InfoBar, InfoBarPosition
            
            # 获取主窗口
            main_window = None
            for widget in QApplication.topLevelWidgets():
                if widget.isVisible():
                    main_window = widget
                    break
            
            if main_window:
                # 在UI线程中安全显示InfoBar
                if info_type == 'success':
                    InfoBar.success(
                        title=title,
                        content=content,
                        orient=Qt.Horizontal,
                        isClosable=True,
                        position=InfoBarPosition.TOP,
                        duration=4000,
                        parent=main_window
                    )
                elif info_type == 'error':
                    InfoBar.error(
                        title=title,
                        content=content,
                        orient=Qt.Horizontal,
                        isClosable=True,
                        position=InfoBarPosition.TOP,
                        duration=5000,
                        parent=main_window
                    )
                else:
                    InfoBar.info(
                        title=title,
                        content=content,
                        orient=Qt.Horizontal,
                        isClosable=True,
                        position=InfoBarPosition.TOP,
                        duration=4000,
                        parent=main_window
                    )
        except Exception as e:
            print(f"显示InfoBar出错: {e}")
    
    def stop_training(self):
        """停止训练的安全方法"""
        self._mutex.lock()
        try:
            self._stop_flag = True
            self._stop_event.set()  # 设置停止事件
            self._condition.wakeAll()  # 唤醒所有等待的线程
            
            # 确保训练完成后正确清理资源
            self.log_message.emit("正在中断训练过程，请稍候...")
            
            # 创建强制退出计时器 - 如果3秒后线程仍在运行则强制终止
            QTimer.singleShot(3000, self._force_stop_if_running)
        finally:
            self._mutex.unlock()
    
    def _force_stop_if_running(self):
        """如果线程仍在运行，强制终止"""
        if self.isRunning():
            self.log_message.emit("训练未能正常终止，强制结束线程...")
            self.terminate()  # 强制终止线程
            self.wait(1000)   # 等待最多1秒
            
            # 强制回调训练完成信号以更新UI
            self.training_completed.emit(False, "训练已被强制终止")
    
    def should_stop(self):
        """检查是否应该停止训练"""
        # 检查事件对象，这比简单的标志更可靠
        if self._stop_event.is_set():
            if not hasattr(self, '_stop_logged'):
                self._stop_logged = True
                print("训练接收到停止信号，准备终止")
            return True
            
        # 除了内部标志，还检查线程中断状态
        interrupted = self._stop_flag or self.isInterruptionRequested()
        if interrupted and not hasattr(self, '_stop_logged'):
            self._stop_logged = True
            print(f"训练线程标记为中断: _stop_flag={self._stop_flag}, isInterruptionRequested={self.isInterruptionRequested()}")
        return interrupted
    
    def should_pause(self):
        """检查是否应该暂停"""
        return self.is_paused
    
    def run(self):
        """执行训练"""
        self.log_message.emit("开始训练过程...")
        self.log_message.emit(f"数据源: {self.config['data_source']}")
        
        try:
            if self.config['data_source'] == 'self_play':
                self.train_from_self_play()
            else:
                self.train_from_local_data()
        except Exception as e:
            self.log_message.emit(f"训练过程中出错: {str(e)}")
            self.log_message.emit(traceback.format_exc())
            self.training_completed.emit(False, f"训练失败: {str(e)}")
    
    def train_from_self_play(self):
        """从自我对弈生成数据并训练"""
        self._stop_flag = False  # 重置停止标志
        
        self.log_message.emit("准备自我对弈训练...")
        try:
            from ai.models import create_gomoku_model
            from ai.trainer import GomokuTrainer
            from ai.selfplay import SelfPlayManager
            from ai.data_handler import get_data_loaders
            import numpy as np

            # 使用信号更新状态，而不是直接访问UI组件
            self.status_update_signal.emit("正在训练中...", 0)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.log_message.emit(f"使用设备: {device}")

            self.log_message.emit("初始化模型...")
            model_size = self.config.get('model_size', 'tiny')
            model = create_gomoku_model(board_size=15, device=device, model_size=model_size)

            # 新增：加载预训练模型（如果提供）
            if self.config.get('pretrained_model'):
                try:
                    model.load_state_dict(torch.load(self.config['pretrained_model'], map_location=device))
                    self.log_message.emit(f"✅ 已加载预训练模型: {self.config['pretrained_model']}")
                    print(f"✅ 已加载预训练模型: {self.config['pretrained_model']}")
                except Exception as e:
                    self.log_message.emit(f"⚠️ 预训练模型加载失败: {e}")
                    print(f"⚠️ 预训练模型加载失败: {e}")

            iterations = self.config.get('selfplay_iterations', 1)
            num_games = self.config['num_games']
            mcts_sim = self.config['mcts_simulations']
            
            # 验证输出目录
            if not os.path.exists(self.config['output_dir']):
                os.makedirs(self.config['output_dir'], exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_size = self.config.get('model_size', 'tiny')
            # 新建唯一模型保存目录
            models_dir = os.path.join(self.config['output_dir'], 'models', f"{timestamp}_{model_size}")
            os.makedirs(models_dir, exist_ok=True)
            self.log_message.emit(f"主要模型保存目录: {models_dir}")
            
            # 创建训练数据保存目录
            training_data_dir = os.path.join(self.config['output_dir'], 'training_data', f"session_{timestamp}")
            os.makedirs(training_data_dir, exist_ok=True)
            self.log_message.emit(f"训练数据将保存到: {training_data_dir}")
            
            # 保存初始模型
            try:
                init_model_path = os.path.join(models_dir, f"model_init_{timestamp}.pth")
                torch.save(model.state_dict(), init_model_path)
                self.log_message.emit(f"✅ 初始模型已保存到主目录: {init_model_path}")
            except Exception as e:
                self.log_message.emit(f"⚠️ 初始模型保存失败: {str(e)}")
            
            self.log_message.emit("开始训练模型，将在每轮结束后保存...")
            
            # 验证配置中的保存间隔
            save_interval = self.config.get('save_interval', 10)

            # 全局变量用于存储数据加载器
            global_train_loader = None
            global_val_loader = None

            for itr in range(iterations):
                if self.should_stop():
                    self.log_message.emit("训练被用户中断")
                    return
                    
                self.log_message.emit(f"===== 迭代 {itr+1}/{iterations}：自我对弈生成数据 =====")
                selfplay_manager = SelfPlayManager(
                    model=model,
                    board_size=15,
                    mcts_simulations=mcts_sim,
                    device=device,
                    exploration_temp=self.config.get('exploration_temp', 1.0)
                )

                states, policies, values = [], [], []
                for game_idx in range(num_games):
                    if self.should_stop():
                        self.log_message.emit("训练被用户中断")
                        return
                    
                    # 更新进度 - 使用信号
                    progress = int((game_idx / num_games) * 100)
                    status_text = f"正在训练中: {progress}% | 第 {game_idx+1}/{num_games} 局"
                    self.status_update_signal.emit(status_text, progress)
                    
                    self.log_message.emit(f"第 {game_idx+1}/{num_games} 局自我对弈...")
                    
                    # 修改这部分代码以处理多种返回值情况
                    result = selfplay_manager.play_game(
                        board_callback=self.update_board_state,
                        log_patterns=True,  # 启用棋型评估日志
                        check_interrupt=self.should_stop  # 传递中断检查函数
                    )
                    
                    # 如果返回空数据，说明已被中断
                    if not result or len(result) == 0:
                        self.log_message.emit("自我对弈被中断")
                        return
                    
                    # 根据返回结果的长度判断如何解包
                    if len(result) == 3:
                        s, p, v = result
                    elif len(result) == 4:
                        s, p, v, pattern_scores = result
                        # 可以添加对pattern_scores的处理
                        if pattern_scores:
                            max_score = max([score for _, _, score in pattern_scores])
                            self.log_message.emit(f"对局 {game_idx+1} 的最高棋型评分: {max_score}")
                    else:
                        self.log_message.emit(f"警告: play_game返回了意外数量的值 {len(result)}")
                        if len(result) > 2:  # 确保至少有需要的数据
                            s, p, v = result[:3]
                        else:
                            continue  # 跳过这次游戏
                    
                    if not s:  # 检查数据有效性
                        continue
                        
                    states.extend(s)
                    policies.extend(p)
                    values.extend(v)
                    
                    # 保存每一局的数据，以便中断后恢复
                    try:
                        game_dir = os.path.join(training_data_dir, f"game_{itr+1}_{game_idx+1}")
                        os.makedirs(game_dir, exist_ok=True)
                        
                        # 将NumPy数组转换为标准格式
                        game_states = np.array(s)
                        game_policies = np.array(p)
                        game_values = np.array(v)
                        
                        # 保存为NumPy文件格式
                        np.save(os.path.join(game_dir, "states.npy"), game_states)
                        np.save(os.path.join(game_dir, "policies.npy"), game_policies)
                        np.save(os.path.join(game_dir, "values.npy"), game_values)
                        
                        # 保存元数据（可选）
                        metadata = {
                            'timestamp': datetime.now().isoformat(),
                            'game_index': game_idx + 1,
                            'iteration': itr + 1,
                            'states_shape': game_states.shape,
                            'policies_shape': game_policies.shape,
                            'values_shape': game_values.shape
                        }
                        
                        with open(os.path.join(game_dir, "metadata.json"), 'w') as f:
                            json.dump(metadata, f, indent=2)
                            
                    except Exception as e:
                        self.log_message.emit(f"⚠️ 保存训练数据失败: {str(e)}")
                
                # 保存整个迭代的数据
                try:
                    iter_dir = os.path.join(training_data_dir, f"iteration_{itr+1}")
                    os.makedirs(iter_dir, exist_ok=True)
                    
                    # 将列表转换为NumPy数组
                    all_states = np.array(states)
                    all_policies = np.array(policies)
                    all_values = np.array(values)
                    
                    # 保存为NumPy文件格式
                    np.save(os.path.join(iter_dir, "states.npy"), all_states)
                    np.save(os.path.join(iter_dir, "policies.npy"), all_policies)
                    np.save(os.path.join(iter_dir, "values.npy"), all_values)
                    
                    self.log_message.emit(f"✅ 迭代 {itr+1} 的训练数据已保存到: {iter_dir}")
                except Exception as e:
                    self.log_message.emit(f"❌ 保存整个迭代数据失败: {str(e)}")
                
                # 尝试将数据转换为加载器，并检查是否成功
                try:
                    self.log_message.emit(f"开始创建数据加载器，批次大小: {int(self.config['batch_size'])}")
                    
                    # 确保有足够数据进行训练/验证集分割
                    min_samples_needed = max(2 * int(self.config['batch_size']), 10)
                    if len(states) < min_samples_needed:
                        self.log_message.emit(f"⚠️ 警告: 训练样本数量较少 ({len(states)}), 需要至少 {min_samples_needed} 个样本进行有效训练")
                        # 如果样本不足，复制样本以确保有足够数据
                        multiply_factor = (min_samples_needed // len(states)) + 1
                        self.log_message.emit(f"复制现有样本 {multiply_factor} 次以确保有足够训练数据")
                        states = states * multiply_factor
                        policies = policies * multiply_factor
                        values = values * multiply_factor
                        self.log_message.emit(f"扩充后的训练数据 - states: {len(states)}")
                    
                    # 创建数据加载器并确保变量被正确定义
                    global_train_loader, global_val_loader = get_data_loaders(states, policies, values,
                                                                batch_size=int(self.config['batch_size']))
                    
                    # 检查数据加载器是否有效
                    self.log_message.emit(f"数据加载器已创建 - 训练批次: {len(global_train_loader)}, 验证批次: {len(global_val_loader)}")
                    
                    if len(global_train_loader) == 0 or len(global_val_loader) == 0:
                        self.log_message.emit("❌ 警告: 数据加载器为空，可能导致训练失败")
                except Exception as e:
                    self.log_message.emit(f"❌ 创建数据加载器失败: {str(e)}")
                    self.log_message.emit(traceback.format_exc())
                    self.training_completed.emit(False, f"创建数据加载器失败: {str(e)}")
                    return

                self.log_message.emit(f"===== 迭代 {itr+1}/{iterations}：开始训练模型 =====")
                
                # 输出更明确的调试信息
                self.log_message.emit(f"📊 当前迭代: {itr+1}/{iterations}, 保存间隔: {save_interval}")
                
                # 创建带有保存目录参数的训练器配置
                trainer_config = {'save_interval': save_interval}
                try:
                    trainer = GomokuTrainer(model, device=device, learning_rate=float(self.config['learning_rate']), config=trainer_config)
                    
                    # 注意：为trainer.train传递保存目录参数
                    train_save_dir = models_dir
                    self.log_message.emit(f"📂 训练保存目录: {train_save_dir}")
                    
                    # 确保使用已创建的数据加载器
                    if global_train_loader is None or global_val_loader is None:
                        raise ValueError("数据加载器未创建成功")
                    
                    # 开始训练，使用全局数据加载器变量
                    trainer.train(global_train_loader, global_val_loader,
                                num_epochs=int(self.config['epochs']),
                                save_dir=train_save_dir,
                                callback=self._training_callback)
                    self.log_message.emit(f"✅ 模型训练成功完成")
                except Exception as e:
                    self.log_message.emit(f"❌ 模型训练异常: {str(e)}")
                    self.log_message.emit(traceback.format_exc())
                    
                    # 即使训练失败，也尝试保存最终模型
                    self.log_message.emit("尝试保存当前模型状态...")
                
                model = trainer.model if hasattr(trainer, 'model') else model
                self.log_message.emit(f"第 {itr+1} 轮训练完成，模型已更新")

                # 无论配置间隔如何，记录每轮的保存检查
                self.log_message.emit(f"⏱️ 检查保存条件: 当前迭代 {itr+1} % 保存间隔 {save_interval} = {(itr+1) % save_interval}")
                
                # 保存模型逻辑 - 增加额外检查
                if (itr + 1) % save_interval == 0:
                    self.log_message.emit(f"🔄 符合保存条件，准备保存模型...")
                    try:
                        save_path = os.path.join(models_dir, f"model_itr_{itr+1}_{timestamp}.pth")
                        torch.save(model.state_dict(), save_path)
                        self.log_message.emit(f"✅ 模型已保存到主目录: {save_path}")
                    except Exception as e:
                        self.log_message.emit(f"❌ 保存模型失败: {str(e)}")
                        self.log_message.emit(traceback.format_exc())
                else:
                    self.log_message.emit(f"⏭️ 不符合保存条件，跳过保存")

            # 保存最终模型到唯一位置
            try:
                final_path = os.path.join(models_dir, f"model_final_{timestamp}.pth")
                torch.save(model.state_dict(), final_path)
                self.log_message.emit(f"✅ 最终模型已保存到主目录: {final_path}")
                self.log_message.emit(f"📚 所有模型文件位置:\n1. 主目录: {models_dir}")
                self.training_completed.emit(True, "自我对弈迭代训练完成")
            except Exception as e:
                self.log_message.emit(f"❌ 保存最终模型失败: {str(e)}")
                self.log_message.emit(traceback.format_exc())
                self.training_completed.emit(False, f"训练完成但保存模型失败: {str(e)}")

            # 更新最终进度
            self.status_update_signal.emit("训练完成 (100%)", 100)

        except Exception as e:
            self.log_message.emit(f"训练过程中出错: {str(e)}")
            self.log_message.emit(traceback.format_exc())
            self.training_completed.emit(False, f"训练失败: {str(e)}")
    
    def train_from_local_data(self):
        """从本地数据训练模型"""
        self._stop_flag = False  # 重置停止标志
        
        # 获取根目录
        root_dir = self.config.get('local_data_path', '')
        if not root_dir or not os.path.exists(root_dir):
            self.log_message.emit("错误: 无效的本地数据路径")
            self.training_completed.emit(False, "无效的本地数据路径")
            return
            
        self.log_message.emit(f"开始从本地数据训练模型，数据根目录: {root_dir}")
        
        try:
            from ai.models import create_gomoku_model
            from ai.trainer import GomokuTrainer
            from ai.data_handler import get_data_loaders
            import numpy as np
            
            # 使用信号更新状态，而不是直接访问UI组件
            self.status_update_signal.emit("正在搜索训练数据...", 0)

            # 搜索所有包含训练数据的目录
            self.log_message.emit(f"开始在 {root_dir} 中搜索训练数据...")
            
            required_files = ['states.npy', 'policies.npy', 'values.npy']
            data_dirs = []
            
            # 计数器，用于显示搜索进度
            dirs_checked = 0
            
            # 使用os.walk递归遍历目录
            for dirpath, dirnames, filenames in os.walk(root_dir):
                dirs_checked += 1
                
                # 每检查50个目录显示一次进度
                if dirs_checked % 50 == 0:
                    self.log_message.emit(f"已检查 {dirs_checked} 个目录，找到 {len(data_dirs)} 个有效数据目录...")
                
                # 检查当前目录是否包含所有必需的文件
                has_all_files = all(filename in filenames for filename in required_files)
                
                if has_all_files:
                    data_dirs.append(dirpath)
            
            self.log_message.emit(f"搜索完成，共检查了 {dirs_checked} 个目录，找到 {len(data_dirs)} 个有效训练数据目录。")
            
            if not data_dirs:
                self.log_message.emit("错误: 未找到有效训练数据")
                self.training_completed.emit(False, "未找到有效训练数据")
                return
                
            self.log_message.emit(f"找到 {len(data_dirs)} 个训练数据目录，开始加载数据...")
            
            # 统计所有数据的总样本数，以便预分配内存
            total_samples = 0
            sample_counts = []
            
            # 检查数据规模，决定加载策略
            for i, data_dir in enumerate(data_dirs[:min(10, len(data_dirs))]):  # 只检查前10个目录
                try:
                    states_path = os.path.join(data_dir, 'states.npy')
                    if os.path.exists(states_path):
                        # 只加载状态数组的形状信息，不加载全部数据
                        states = np.load(states_path, mmap_mode='r')
                        samples = states.shape[0]
                        sample_counts.append(samples)
                except Exception as e:
                    self.log_message.emit(f"检查样本数量时出错: {str(e)}")
            
            # 如果找到了样本，计算平均样本数
            if sample_counts:
                avg_samples = int(sum(sample_counts) / len(sample_counts))
                estimated_total = avg_samples * len(data_dirs)
                self.log_message.emit(f"估计总样本数: 约 {estimated_total} 个样本")
                
                # 根据总样本数决定加载策略
                use_batch_loading = estimated_total > 50000  # 如果样本超过5万，使用批量加载
                
                if use_batch_loading:
                    self.log_message.emit(f"数据量较大，将使用批量加载以减少内存使用")
                    return self._train_with_batch_loading(data_dirs, root_dir)
            
            # 默认使用全量加载
            self.status_update_signal.emit(f"正在加载数据 (0%)...", 0)
            
            # 加载并合并所有训练数据
            all_states = []
            all_policies = []
            all_values = []
            
            # 加载每个目录中的数据
            for i, data_dir in enumerate(data_dirs):
                # 检查是否应该停止
                if self.should_stop():
                    self.log_message.emit("加载数据已中断")
                    self.training_completed.emit(False, "用户已中断")
                    return
                
                # 更新进度
                progress = int((i / len(data_dirs)) * 50)  # 数据加载占整个过程的50%
                self.status_update_signal.emit(f"正在加载数据 ({progress}%)...", progress)
                
                try:
                    # 加载数据
                    states_path = os.path.join(data_dir, 'states.npy')
                    policies_path = os.path.join(data_dir, 'policies.npy')
                    values_path = os.path.join(data_dir, 'values.npy')
                    
                    if os.path.exists(states_path) and os.path.exists(policies_path) and os.path.exists(values_path):
                        states = np.load(states_path)
                        policies = np.load(policies_path)
                        values = np.load(values_path)
                        
                        # 检查数据形状是否匹配
                        if len(states) == len(policies) == len(values):
                            # 将数据添加到全局列表
                            all_states.append(states)
                            all_policies.append(policies)
                            all_values.append(values)
                            
                            # 添加详细日志
                            rel_path = os.path.relpath(data_dir, root_dir)
                            session_name = os.path.basename(os.path.dirname(data_dir))
                            is_user_data = 'user_session' in session_name
                            session_label = "用户贡献" if is_user_data else "自我对弈"
                            self.log_message.emit(f"加载数据: {rel_path} - {len(states)} 个样本 [{session_label}]")
                        else:
                            self.log_message.emit(f"警告: 数据目录 {data_dir} 中的数据形状不匹配，已跳过")
                    else:
                        self.log_message.emit(f"警告: 数据目录 {data_dir} 缺少必要文件，已跳过")
                        
                except Exception as e:
                    self.log_message.emit(f"加载 {data_dir} 时出错: {str(e)}")
            
            # 清理内存
            import gc
            gc.collect()
            
            # 合并所有数据
            if all_states:
                try:
                    # 合并前记录一下总样本数
                    total_samples = sum(len(s) for s in all_states)
                    self.log_message.emit(f"准备合并总计 {total_samples} 个样本...")
                    
                    states = np.concatenate(all_states)
                    policies = np.concatenate(all_policies)
                    values = np.concatenate(all_values)
                    
                    # 释放原始数据列表内存
                    del all_states, all_policies, all_values
                    gc.collect()
                    
                    self.log_message.emit(f"数据加载完成，共 {len(states)} 个训练样本")
                except Exception as e:
                    self.log_message.emit(f"合并数据出错: {str(e)}")
                    self.log_message.emit(traceback.format_exc())
                    self.training_completed.emit(False, f"合并数据出错: {str(e)}")
                    return
            else:
                self.log_message.emit("错误: 未能加载任何有效数据")
                self.training_completed.emit(False, "未能加载任何有效数据")
                return
            
            # 开始训练模型
            self.status_update_signal.emit("正在初始化模型...", 50)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.log_message.emit(f"使用设备: {device}")

            self.log_message.emit("初始化模型...")
            model_size = self.config.get('model_size', 'tiny')
            model = create_gomoku_model(board_size=15, device=device, model_size=model_size)

            # 新增：加载预训练模型（如果提供）
            if self.config.get('pretrained_model'):
                try:
                    model.load_state_dict(torch.load(self.config['pretrained_model'], map_location=device))
                    self.log_message.emit(f"✅ 已加载预训练模型: {self.config['pretrained_model']}")
                    print(f"✅ 已加载预训练模型: {self.config['pretrained_model']}")
                except Exception as e:
                    self.log_message.emit(f"⚠️ 预训练模型加载失败: {e}")
                    print(f"⚠️ 预训练模型加载失败: {e}")

            # 验证输出目录
            if not os.path.exists(self.config['output_dir']):
                os.makedirs(self.config['output_dir'], exist_ok=True)
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_size = self.config.get('model_size', 'tiny')
            # 新建唯一模型保存目录
            models_dir = os.path.join(self.config['output_dir'], 'models', f"{timestamp}_{model_size}")
            os.makedirs(models_dir, exist_ok=True)
            self.log_message.emit(f"主要模型保存目录: {models_dir}")
            
            # 创建数据加载器
            self.log_message.emit(f"创建数据加载器，批次大小: {int(self.config['batch_size'])}")
            try:
                self.log_message.emit(f"创建数据加载器 - 样本数: {len(states)}")
                
                # 确保有足够数据进行训练/验证集分割
                min_samples_needed = max(2 * int(self.config['batch_size']), 10)
                if len(states) < min_samples_needed:
                    self.log_message.emit(f"⚠️ 警告: 训练样本数量较少 ({len(states)}), 需要至少 {min_samples_needed} 个样本进行有效训练")
                    # 如果样本不足，复制样本以确保有足够数据
                    multiply_factor = (min_samples_needed // len(states)) + 1
                    self.log_message.emit(f"复制现有样本 {multiply_factor} 次以确保有足够训练数据")
                    states = np.tile(states, (multiply_factor, 1, 1, 1))
                    policies = np.tile(policies, (multiply_factor, 1))
                    values = np.tile(values, (multiply_factor, 1))
                    self.log_message.emit(f"扩充后的训练数据 - states: {len(states)}")
                
                train_loader, val_loader = get_data_loaders(
                    states, policies, values,
                    batch_size=int(self.config['batch_size'])
                )
                
                self.log_message.emit(f"创建了数据加载器 - 训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}")
                
                if len(train_loader) == 0 or len(val_loader) == 0:
                    self.log_message.emit("❌ 警告: 数据加载器为空，可能导致训练失败")
            except Exception as e:
                self.log_message.emit(f"创建数据加载器失败: {str(e)}")
                self.log_message.emit(traceback.format_exc())
                self.training_completed.emit(False, f"创建数据加载器失败: {str(e)}")
                return
            
            # 开始训练
            self.log_message.emit("===== 开始训练模型 =====")
            self.status_update_signal.emit("正在训练模型 (0%)...", 50)
            
            # 输出更明确的调试信息
            save_interval = self.config.get('save_interval', 10)
            self.log_message.emit(f"📊 保存间隔: {save_interval} 轮")
            
            # 创建带有保存目录参数的训练器配置
            trainer_config = {
                'save_interval': save_interval,
                'weight_decay': float(self.config.get('weight_decay', 0.0005)),
                'dropout': float(self.config.get('dropout', 0.3)),
                'optimizer': self.config.get('optimizer', 'Adam')
            }
            
            try:
                trainer = GomokuTrainer(model, device=device, learning_rate=float(self.config['learning_rate']), config=trainer_config)
                
                # 开始训练
                train_save_dir = models_dir
                self.log_message.emit(f"📂 训练保存目录: {train_save_dir}")
                
                epochs = int(self.config['epochs'])
                trainer.train(train_loader, val_loader,
                            num_epochs=epochs,
                            save_dir=train_save_dir,
                            callback=self._training_callback)
                
                self.log_message.emit(f"✅ 模型训练成功完成")
            except Exception as e:
                self.log_message.emit(f"❌ 模型训练异常: {str(e)}")
                self.log_message.emit(traceback.format_exc())
                
                # 即使训练失败，也尝试保存最终模型
                self.log_message.emit("尝试保存当前模型状态...")
            
            # 保存最终模型到唯一位置
            try:
                final_path = os.path.join(models_dir, f"model_final_{timestamp}.pth")
                torch.save(model.state_dict(), final_path)
                self.log_message.emit(f"✅ 最终模型已保存到主目录: {final_path}")
                self.log_message.emit(f"📚 所有模型文件位置:\n1. 主目录: {models_dir}")
                self.training_completed.emit(True, "本地数据训练完成")
            except Exception as e:
                self.log_message.emit(f"❌ 保存最终模型失败: {str(e)}")
                self.log_message.emit(traceback.format_exc())
                self.training_completed.emit(False, f"训练完成但保存模型失败: {str(e)}")

            # 更新最终进度
            self.status_update_signal.emit("训练完成 (100%)", 100)
            
        except Exception as e:
            self.log_message.emit(f"训练过程中出错: {str(e)}")
            self.log_message.emit(traceback.format_exc())
            self.training_completed.emit(False, f"训练失败: {str(e)}")

    def _train_with_batch_loading(self, data_dirs, root_dir):
        """使用批量加载方式训练，适用于超大数据集
        
        Args:
            data_dirs: 包含训练数据的目录列表
            root_dir: 数据根目录
        """
        self.log_widget.append("使用批量加载模式进行训练 - 适用于大型数据集")
        
        try:
            from ai.models import create_gomoku_model
            from ai.trainer import GomokuTrainer
            import numpy as np
            
            # 初始化模型
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.log_message.emit(f"使用设备: {device}")
            model_size = self.config.get('model_size', 'tiny')
            self.log_message.emit(f"模型大小: {model_size}")
            model = create_gomoku_model(board_size=15, device=device, model_size=model_size)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_size = self.config.get('model_size', 'tiny')
            # 新建唯一模型保存目录
            models_dir = os.path.join(self.config['output_dir'], 'models', f"{timestamp}_{model_size}")
            os.makedirs(models_dir, exist_ok=True)
            self.log_message.emit(f"主要模型保存目录: {models_dir}")
            
            # 创建训练器
            trainer_config = {
                'save_interval': self.config.get('save_interval', 10),
                'weight_decay': float(self.config.get('weight_decay', 0.0005)),
                'dropout': float(self.config.get('dropout', 0.3)),
                'optimizer': self.config.get('optimizer', 'Adam')
            }
            
            trainer = GomokuTrainer(model, device=device, learning_rate=float(self.config['learning_rate']), config=trainer_config)
            
            # 分批次处理数据目录
            batch_size = min(50, len(data_dirs))  # 每批最多50个目录
            num_batches = (len(data_dirs) + batch_size - 1) // batch_size
            
            self.log_message.emit(f"将 {len(data_dirs)} 个数据目录分为 {num_batches} 批进行处理")
            
            # 分批训练
            epochs_per_batch = max(1, int(self.config['epochs']) // num_batches)
            self.log_message.emit(f"每批数据训练 {epochs_per_batch} 轮")
            
            for batch_idx in range(num_batches):
                if self.should_stop():
                    self.log_message.emit("训练已中断")
                    self.training_completed.emit(False, "用户已中断")
                    return
                
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(data_dirs))
                current_dirs = data_dirs[start_idx:end_idx]
                
                self.log_message.emit(f"处理第 {batch_idx+1}/{num_batches} 批数据目录，包含 {len(current_dirs)} 个目录")
                
                # 加载当前批次的数据
                all_states = []
                all_policies = []
                all_values = []
                
                for i, data_dir in enumerate(current_dirs):
                    # 检查是否应该停止
                    if self.should_stop():
                        self.log_message.emit("加载数据已中断")
                        return
                    
                    # 更新进度
                    batch_progress = (batch_idx * 100) / num_batches
                    dir_progress = (i * 100) / len(current_dirs) / num_batches
                    progress = int(batch_progress + dir_progress)
                    self.status_update_signal.emit(f"正在加载数据批次 {batch_idx+1}/{num_batches} ({progress}%)...", progress)
                    
                    try:
                        # 加载数据文件
                        states_path = os.path.join(data_dir, 'states.npy')
                        policies_path = os.path.join(data_dir, 'policies.npy')
                        values_path = os.path.join(data_dir, 'values.npy')
                        
                        if os.path.exists(states_path) and os.path.exists(policies_path) and os.path.exists(values_path):
                            states = np.load(states_path)
                            policies = np.load(policies_path)
                            values = np.load(values_path)
                            
                            if len(states) == len(policies) == len(values):
                                all_states.append(states)
                                all_policies.append(policies)
                                all_values.append(values)
                                
                                rel_path = os.path.relpath(data_dir, root_dir)
                                self.log_message.emit(f"加载数据: {rel_path} - {len(states)} 个样本")
                            else:
                                self.log_message.emit(f"警告: 数据形状不匹配，已跳过 {data_dir}")
                    except Exception as e:
                        self.log_message.emit(f"加载 {data_dir} 时出错: {str(e)}")
                
                # 如果没有加载到数据，跳过此批次
                if not all_states:
                    self.log_message.emit(f"批次 {batch_idx+1} 未加载到有效数据，跳过")
                    continue
                
                # 合并数据
                try:
                    states = np.concatenate(all_states)
                    policies = np.concatenate(all_policies)
                    values = np.concatenate(all_values)
                    
                    # 清理原始数据
                    del all_states, all_policies, all_values
                    import gc
                    gc.collect()
                    
                    self.log_message.emit(f"批次 {batch_idx+1} 数据已合并，共 {len(states)} 个样本")
                except Exception as e:
                    self.log_message.emit(f"合并批次 {batch_idx+1} 数据出错: {str(e)}")
                    continue
                
                # 创建数据加载器
                from ai.data_handler import get_data_loaders
                try:
                    train_loader, val_loader = get_data_loaders(
                        states, policies, values,
                        batch_size=int(self.config['batch_size'])
                    )
                    
                    self.log_message.emit(f"批次 {batch_idx+1} 数据加载器已创建 - 训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}")
                except Exception as e:
                    self.log_message.emit(f"创建批次 {batch_idx+1} 数据加载器失败: {str(e)}")
                    continue
                
                # 训练当前批次
                self.log_message.emit(f"开始训练批次 {batch_idx+1} 数据...")
                try:
                    trainer.train(
                        train_loader, val_loader,
                        num_epochs=epochs_per_batch,
                        save_dir=models_dir,
                        callback=self._training_callback
                    )
                    
                    # 每个批次结束后保存一个检查点
                    checkpoint_path = os.path.join(models_dir, f"model_batch_{batch_idx+1}_of_{num_batches}.pth")
                    torch.save(model.state_dict(), checkpoint_path)
                    self.log_message.emit(f"已保存批次 {batch_idx+1} 训练后的模型")
                    
                    # 释放内存
                    del states, policies, values, train_loader, val_loader
                    gc.collect()
                    
                except Exception as e:
                    self.log_message.emit(f"训练批次 {batch_idx+1} 数据时出错: {str(e)}")
            
            # 所有批次训练完成，保存最终模型
            try:
                final_path = os.path.join(models_dir, f"model_final_{timestamp}.pth")
                torch.save(model.state_dict(), final_path)
                self.log_message.emit(f"✅ 最终模型已保存: {final_path}")
                self.training_completed.emit(True, "批量训练完成")
            except Exception as e:
                self.log_message.emit(f"保存最终模型失败: {str(e)}")
                self.training_completed.emit(False, f"训练完成但保存失败: {str(e)}")
                
            self.status_update_signal.emit("训练完成 (100%)", 100)
            
        except Exception as e:
            self.log_message.emit(f"批量训练过程中出错: {str(e)}")
            self.log_message.emit(traceback.format_exc())
            self.training_completed.emit(False, f"批量训练失败: {str(e)}")

    def update_board_state(self, board, move_history, current_player, pattern_score=None):
        """更新棋盘状态，用于自我对弈可视化"""
        # 检查是否应该停止
        if self.should_stop():
            return
        
        # 如果没有历史记录，跳过
        if not move_history:
            return
        
        # 将NumPy数组转换为Python列表
        if hasattr(board, 'tolist'):
            board = board.tolist()
        
        # 确保move_history也是列表类型
        if hasattr(move_history, 'tolist'):
            move_history = move_history.tolist()
        
        # 创建唯一标识，使用move_history的长度和最后一步走法
        current_state = (len(move_history), move_history[-1] if move_history else None)
        
        # 检查是否与上一步相同，避免重复处理
        if hasattr(self, '_last_board_state') and self._last_board_state == current_state:
            return
        
        # 保存当前状态
        self._last_board_state = current_state
        
        # 通过信号发送更新
        try:
            self.board_updated.emit(board, move_history, current_player)
            
            # 构建日志消息，确保格式正确
            player_name = "黑棋" if current_player == 1 else "白棋" if current_player == 2 else "游戏结束"
            move_info = ""
            
            if move_history and len(move_history) > 0:
                last_move = move_history[-1]
                if isinstance(last_move, (list, tuple)) and len(last_move) >= 2:
                    row, col = last_move[0], last_move[1]
                    move_info = f"落子坐标：({row}, {col})"
            
            # 使用正确的回合计数，确保格式统一
            self.log_message.emit(f"棋盘状态已更新：回合 {len(move_history)}。{player_name}{move_info}。")
        except Exception as e:
            print(f"发送board_updated信号出错: {e}")
        
        # 处理UI更新
        QApplication.processEvents()
    
    def save_final_models(self):
        """保存最终训练好的模型"""
        if not self.config['output_dir']:
            self.log_message.emit("警告: 未指定输出目录，跳过保存最终模型")
            return
        
        self.log_message.emit("保存最终训练模型和统计数据...")
        
        try:
            output_dir = self.config['output_dir']
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            date_str = datetime.now().strftime("%Y%m%d")
            model_size = self.config.get('model_size', 'tiny')
            # 新建唯一模型保存目录
            models_dir = os.path.join(output_dir, 'models', f"{timestamp}_{model_size}")
            os.makedirs(models_dir, exist_ok=True)
            normalized_path = os.path.normpath(models_dir)
            self.log_message.emit(f"模型将保存在: {normalized_path}")
            
            # 保存训练统计数据
            stats_data = {
                'total_games': self.current_game if hasattr(self, 'current_game') else 0,
                'stats': getattr(self, 'stats', {}),
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }
            
            stats_path = os.path.join(models_dir, f'training_stats_{timestamp}.json')
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats_data, f, indent=2)
            
            self.log_message.emit(f"训练统计已保存: {os.path.normpath(stats_path)}")
            
            # 保存最终模型文件
            model1_path = os.path.join(models_dir, f'model1_final_{timestamp}.pth')
            model2_path = os.path.join(models_dir, f'model2_final_{timestamp}.pth')
            
            # 根据不同情况保存模型
            model_files = glob.glob(os.path.join(models_dir, "*_final_*.pth"))
            if model_files:
                model_files.sort(key=os.path.getmtime, reverse=True)
                newest_model = model_files[0]
                
                shutil.copy2(newest_model, model1_path)
                shutil.copy2(newest_model, model2_path)
                
                self.log_message.emit(f"模型1已保存: {os.path.normpath(model1_path)}")
                self.log_message.emit(f"模型2已保存: {os.path.normpath(model2_path)}")
            else:
                # 创建与训练时相同结构的模型
                try:
                    from ai.models import create_gomoku_model
                    
                    # 创建与训练使用的相同结构模型
                    device = torch.device("cpu")
                    model = create_gomoku_model(board_size=15, device=device, model_size=model_size)
                    
                    # 保存模型
                    torch.save(model.state_dict(), model1_path)
                    torch.save(model.state_dict(), model2_path)
                    
                    self.log_message.emit(f"模型1已保存: {os.path.normpath(model1_path)}")
                    self.log_message.emit(f"模型2已保存: {os.path.normpath(model2_path)}")
                except Exception as e:
                    # 如果导入失败，使用备选方案
                    self.log_message.emit(f"创建标准模型失败: {str(e)}，使用备选模型")
                    
                    # 备选方案：使用简单但更接近实际模型结构的网络
                    class SimplePolicyValueNet(nn.Module):
                        def __init__(self, board_size=15):
                            super().__init__()
                            self.conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
                            self.policy_head = nn.Sequential(
                                nn.Conv2d(32, 16, kernel_size=1),
                                nn.Flatten(),
                                nn.Linear(16 * board_size * board_size, board_size * board_size),
                                nn.LogSoftmax(dim=1)
                            )
                            self.value_head = nn.Sequential(
                                nn.Conv2d(32, 8, kernel_size=1),
                                nn.Flatten(),
                                nn.Linear(8 * board_size * board_size, 64),
                                nn.ReLU(),
                                nn.Linear(64, 1),
                                nn.Tanh()
                            )
                        
                        def forward(self, x):
                            x = F.relu(self.conv(x))
                            return self.policy_head(x), self.value_head(x)
                    
                    tiny_model = SimplePolicyValueNet(board_size=15)
                    torch.save(tiny_model.state_dict(), model1_path)
                    torch.save(tiny_model.state_dict(), model2_path)
            
            # 额外保存一个当前日期_类型命名的最佳模型在models目录下，方便快速访问
            master_model_name = f"best_{model_size}_{date_str}.pth"
            master_model_path = os.path.join(output_dir, 'models', master_model_name)
            try:
                shutil.copy2(model1_path, master_model_path)
                self.log_message.emit(f"最佳模型副本已保存: {os.path.normpath(master_model_path)}")
            except Exception as e:
                self.log_message.emit(f"保存最佳模型副本失败: {str(e)}")
                
            self.log_message.emit(f"提示: 请在文件夹 '{os.path.normpath(models_dir)}' 中查看保存的模型文件")
            
        except Exception as e:
            self.log_message.emit(f"保存模型失败: {str(e)}")
            self.log_message.emit(traceback.format_exc())
    
    def resume(self):
        """恢复训练"""
        self.mutex.lock()
        try:
            self.is_paused = False
            self.pause_condition.wakeAll()
            self.log_message.emit("训练已恢复")
        finally:
            self.mutex.unlock()
    
    def pause(self):
        """暂停训练"""
        self.mutex.lock()
        try:
            self.is_paused = True
            self.log_message.emit("训练已暂停")
        finally:
            self.mutex.unlock()
    
    def stop(self):
        """停止训练"""
        self.log_message.emit("正在停止训练...")
        self.is_running = False
        self.stop_training()  # 调用主要的停止方法
        self.resume()  # 如果处于暂停状态，唤醒以便能检测到停止信号

    # 添加回调函数来处理训练过程中的事件
    def _training_callback(self, event_type, event_data):
        """处理训练过程中的事件"""
        # 在每次回调中首先检查是否应该停止
        if self.should_stop():
            self.log_message.emit("训练过程已被用户中断，正在退出...")
            return False  # 返回False以停止训练
            
        if event_type == 'epoch_end':
            epoch = event_data.get('epoch', 0)
            train_loss = event_data.get('train_loss', 'N/A')
            val_loss = event_data.get('val_loss', 'N/A')
            
            # 更详细的进度信息
            self.log_message.emit(f"完成训练轮次 {epoch}, 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
            
            # 更新进度条
            total_epochs = self.config.get('epochs', 100)
            progress = int((epoch / total_epochs) * 100)
            self.status_update_signal.emit(f"训练进度: {progress}% (第 {epoch}/{total_epochs} 轮)", progress)
            
            # 检查是否处于暂停状态
            while self.is_paused and not self.should_stop():
                self.log_message.emit("训练已暂停，等待恢复...")
                time.sleep(0.5)  # 等待500毫秒后再次检查
            
            # 每个epoch结束后都检查是否应该停止
            if self.should_stop():
                return False  # 返回False以停止训练
                
            return True  # 继续训练
        elif event_type == 'batch_end':
            batch = event_data.get('batch', 0)
            total_batches = event_data.get('total_batches', 1)
            
            # 每10%更新一次进度
            if total_batches > 10 and batch % (total_batches // 10) == 0:
                epoch = event_data.get('epoch', 0)
                batch_progress = int((batch / total_batches) * 100)
                self.log_message.emit(f"轮次 {epoch} 进度: {batch_progress}% (批次 {batch}/{total_batches})")
            
            # 检查是否处于暂停状态
            if self.is_paused and batch % 10 == 0:  # 每10个批次检查一次，避免过于频繁
                while self.is_paused and not self.should_stop():
                    time.sleep(0.2)  # 等待200毫秒后再次检查
            
            # 每50个批次检查一次是否应该停止，避免过于频繁的检查
            if batch % 50 == 0:
                if self.should_stop():
                    return False
        
        return True  # 继续训练


class TrainingInterface(ScrollArea):
    """训练界面，用于模型训练和自我对弈"""
    
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("Training-Interface")
        
        self.view_widget = QWidget(self)
        self.setWidget(self.view_widget)
        self.setWidgetResizable(True)
        
        self.main_layout = QVBoxLayout(self.view_widget)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(15)
        
        self.title = TitleLabel("AI训练中心", self)
        self.main_layout.addWidget(self.title)
        
        self.content_layout = QHBoxLayout()
        self.main_layout.addLayout(self.content_layout, 1)
        
        self.left_widget = QWidget()
        self.left_layout = QVBoxLayout(self.left_widget)
        self.left_layout.setContentsMargins(0, 0, 10, 0)
        
        self.config_card = CardWidget(self)
        self.config_layout = QVBoxLayout(self.config_card)
        
        self.config_title = SubtitleLabel("训练配置", self)
        self.config_layout.addWidget(self.config_title)
        
        # 数据来源标签（修改为下拉框选择）
        self.data_layout = QHBoxLayout()
        self.data_label = BodyLabel("数据来源:", self)
        self.data_combobox = ComboBox(self)
        self.data_combobox.addItem("自我对弈生成数据")
        self.data_combobox.addItem("本地数据")
        self.data_combobox.setToolTip("选择训练数据来源")
        self.data_layout.addWidget(self.data_label)
        self.data_layout.addWidget(self.data_combobox, 1)
        self.config_layout.addLayout(self.data_layout)
        
        # 本地数据路径选择（开始时隐藏）
        self.local_data_widget = QWidget(self)
        self.local_data_layout = QHBoxLayout(self.local_data_widget)
        self.local_data_layout.setContentsMargins(0, 0, 0, 0)
        
        self.local_data_label = BodyLabel("数据路径:", self)
        self.local_data_path = LineEdit(self)
        self.local_data_path.setPlaceholderText("选择本地训练数据文件夹")
        self.local_data_browse = PushButton("浏览", self, FIF.FOLDER)
        
        self.local_data_layout.addWidget(self.local_data_label)
        self.local_data_layout.addWidget(self.local_data_path, 1)
        self.local_data_layout.addWidget(self.local_data_browse)
        
        self.config_layout.addWidget(self.local_data_widget)
        
        # 添加数据格式提示信息
        self.data_format_info = BodyLabel("", self)
        self.data_format_info.setWordWrap(True)
        self.data_format_info.setStyleSheet("color: #0078d4; font-size: 11px;")
        self.config_layout.addWidget(self.data_format_info)
        
        # 默认隐藏本地数据控件并显示自我对弈控件
        self.local_data_widget.hide()
        self.data_format_info.hide()
        
        # 监听数据来源变化，显示/隐藏相应控件
        def on_data_source_changed():
            is_local_data = self.data_combobox.currentText() == "本地数据"
            self.selfplay_widget.setVisible(not is_local_data)
            self.local_data_widget.setVisible(is_local_data)
            
            if is_local_data:
                format_text = "本地数据格式要求：文件夹需包含 'states.npy'(棋盘状态)、'policies.npy'(动作概率)和'values.npy'(价值评估)三个NumPy数组文件。"
                self.data_format_info.setText(format_text)
                self.data_format_info.show()
            else:
                self.data_format_info.hide()
        
        self.data_combobox.currentTextChanged.connect(on_data_source_changed)
        
        # 浏览本地数据目录
        self.local_data_browse.clicked.connect(self.browse_local_data_dir)
        
        # 自我对弈配置组件
        self.selfplay_widget = QWidget(self)
        self.selfplay_layout = QVBoxLayout(self.selfplay_widget)
        self.selfplay_layout.setContentsMargins(0, 10, 0, 10)
        
        self.games_layout = QHBoxLayout()
        self.games_label = BodyLabel("对弈局数:", self)
        self.games_spinbox = SpinBox(self)
        self.games_spinbox.setRange(10, 5000)
        self.games_spinbox.setValue(500)
        self.games_layout.addWidget(self.games_label)
        self.games_layout.addWidget(self.games_spinbox, 1)
        self.selfplay_layout.addLayout(self.games_layout)
        
        self.mcts_layout = QHBoxLayout()
        self.mcts_label = BodyLabel("MCTS模拟次数:", self)
        self.mcts_spinbox = SpinBox(self)
        self.mcts_spinbox.setRange(100, 10000)
        self.mcts_spinbox.setValue(1000)
        self.mcts_layout.addWidget(self.mcts_label)
        self.mcts_layout.addWidget(self.mcts_spinbox, 1)
        self.selfplay_layout.addLayout(self.mcts_layout)
        
        self.selfplay_desc = CaptionLabel("通过AI自我对弈生成训练数据，局数越多训练效果越好，但耗时更长", self)
        self.selfplay_layout.addWidget(self.selfplay_desc)

        self.config_layout.addWidget(self.selfplay_widget)
        
        self.models_layout = QGridLayout()
        self.model1_label = BodyLabel("模型1路径:", self)
        self.model1_path = LineEdit(self)
        self.model1_path.setPlaceholderText("选择或留空使用默认模型")
        self.model1_browse = PushButton("浏览", self, FIF.FOLDER)
        self.models_layout.addWidget(self.model1_label, 0, 0)
        self.models_layout.addWidget(self.model1_path, 0, 1)
        self.models_layout.addWidget(self.model1_browse, 0, 2)
        
        self.model2_label = BodyLabel("模型2路径:", self)
        self.model2_path = LineEdit(self)
        self.model2_path.setPlaceholderText("选择或留空使用默认模型")
        self.model2_browse = PushButton("浏览", self, FIF.FOLDER)
        self.models_layout.addWidget(self.model2_label, 1, 0)
        self.models_layout.addWidget(self.model2_path, 1, 1)
        self.models_layout.addWidget(self.model2_browse, 1, 2)
        
        self.output_label = BodyLabel("输出目录:", self)
        self.output_dir = LineEdit(self)
        self.output_dir.setPlaceholderText("选择模型保存目录")
        self.output_browse = PushButton("浏览", self, FIF.FOLDER)
        self.models_layout.addWidget(self.output_label, 2, 0)
        self.models_layout.addWidget(self.output_dir, 2, 1)
        self.models_layout.addWidget(self.output_browse, 2, 2)
        
        self.config_layout.addLayout(self.models_layout)
        
        self.advanced_title = SubtitleLabel("高级参数", self)
        self.config_layout.addWidget(self.advanced_title)
        
        self.common_advanced_widget = QWidget(self)
        self.common_advanced_layout = QGridLayout(self.common_advanced_widget)
        self.common_advanced_layout.setColumnStretch(1, 1)
        
        # 添加模型大小选择
        self.model_size_label = BodyLabel("模型大小:", self)
        self.model_size_combobox = ComboBox(self)
        self.model_size_combobox.addItems(["tiny", "small", "medium", "large"])
        self.model_size_combobox.setToolTip("模型大小会影响训练速度和性能，tiny最快但性能最弱，large最慢但潜在性能最强")
        self.common_advanced_layout.addWidget(self.model_size_label, 0, 0)
        self.common_advanced_layout.addWidget(self.model_size_combobox, 0, 1)
        
        # 根据模型大小添加对应的描述标签
        self.model_size_desc = CaptionLabel("", self)
        self.model_size_desc.setWordWrap(True)
        self.common_advanced_layout.addWidget(self.model_size_desc, 1, 0, 1, 2)
        
        # 更新大小描述的函数
        def update_model_size_desc(size):
            if size == "tiny":
                self.model_size_desc.setText("最小模型: 训练快，占用内存小，适合快速测试")
            elif size == "small":
                self.model_size_desc.setText("小型模型: 平衡速度和性能，适合一般训练")
            elif size == "medium":
                self.model_size_desc.setText("中型模型: 较好性能，需要更多训练时间和内存")
            else:  # large
                self.model_size_desc.setText("大型模型: 潜在最佳性能，但训练慢且需要大量内存")
        
        # 连接信号
        self.model_size_combobox.currentTextChanged.connect(update_model_size_desc)
        # 设置初始描述
        update_model_size_desc("tiny")
        
        self.batch_label = BodyLabel("批次大小:", self)
        self.batch_spinbox = SpinBox(self)
        self.batch_spinbox.setRange(1, 512)
        self.batch_spinbox.setValue(64)
        self.common_advanced_layout.addWidget(self.batch_label, 2, 0)
        self.common_advanced_layout.addWidget(self.batch_spinbox, 2, 1)
        
        self.lr_label = BodyLabel("学习率:", self)
        self.learning_rate = LineEdit(self)
        self.learning_rate.setText("0.001")
        self.common_advanced_layout.addWidget(self.lr_label, 3, 0)
        self.common_advanced_layout.addWidget(self.learning_rate, 3, 1)
        
        self.epochs_label = BodyLabel("训练轮次:", self)
        self.epochs_spinbox = SpinBox(self)
        self.epochs_spinbox.setRange(1, 10000)
        self.epochs_spinbox.setValue(50)
        self.common_advanced_layout.addWidget(self.epochs_label, 4, 0)
        self.common_advanced_layout.addWidget(self.epochs_spinbox, 4, 1)
        
        self.save_interval_label = BodyLabel("模型保存间隔(轮):", self)
        self.save_interval = SpinBox(self)
        self.save_interval.setRange(1, 50)
        self.save_interval.setValue(1)
        self.save_interval.setToolTip("每训练多少轮(epoch)保存一次模型检查点")
        self.common_advanced_layout.addWidget(self.save_interval_label, 5, 0)
        self.common_advanced_layout.addWidget(self.save_interval, 5, 1)
        
        # 添加缺失的自我对弈迭代次数控件
        self.iterations_label = BodyLabel("自我对弈迭代次数:", self)
        self.iterations_spinbox = SpinBox(self)
        self.iterations_spinbox.setRange(1, 10)
        self.iterations_spinbox.setValue(1)
        self.iterations_spinbox.setToolTip("执行多少轮自我对弈-训练循环")
        self.common_advanced_layout.addWidget(self.iterations_label, 6, 0)
        self.common_advanced_layout.addWidget(self.iterations_spinbox, 6, 1)
        
        # 添加缺失的探索温度控件
        self.temp_label = BodyLabel("探索温度:", self)
        self.temp_spinbox = QDoubleSpinBox(self)
        self.temp_spinbox.setRange(0.1, 2.0)
        self.temp_spinbox.setValue(1.0)
        self.temp_spinbox.setSingleStep(0.1)
        self.temp_spinbox.setToolTip("控制MCTS探索时的随机程度，值越大随机性越强")
        self.common_advanced_layout.addWidget(self.temp_label, 7, 0)
        self.common_advanced_layout.addWidget(self.temp_spinbox, 7, 1)
        
        # 添加权重衰减控件
        self.weight_decay_label = BodyLabel("权重衰减:", self)
        self.weight_decay_spinbox = QDoubleSpinBox(self)
        self.weight_decay_spinbox.setRange(0.0001, 0.01)
        self.weight_decay_spinbox.setValue(0.0005)
        self.weight_decay_spinbox.setSingleStep(0.0001)
        self.weight_decay_spinbox.setDecimals(4)
        self.common_advanced_layout.addWidget(self.weight_decay_label, 8, 0)
        self.common_advanced_layout.addWidget(self.weight_decay_spinbox, 8, 1)
        
        # 添加Dropout控件
        self.dropout_label = BodyLabel("Dropout率:", self)
        self.dropout_spinbox = QDoubleSpinBox(self)
        self.dropout_spinbox.setRange(0.0, 0.5)
        self.dropout_spinbox.setValue(0.3)
        self.dropout_spinbox.setSingleStep(0.05)
        self.common_advanced_layout.addWidget(self.dropout_label, 9, 0)
        self.common_advanced_layout.addWidget(self.dropout_spinbox, 9, 1)
        
        # 添加优化器选择控件
        self.optimizer_label = BodyLabel("优化器:", self)
        self.optimizer_type = ComboBox(self)
        self.optimizer_type.addItems(["Adam", "SGD", "RMSprop"])
        self.optimizer_type.setCurrentText("Adam")
        self.common_advanced_layout.addWidget(self.optimizer_label, 10, 0)
        self.common_advanced_layout.addWidget(self.optimizer_type, 10, 1)
        
        # 添加预训练模型选项
        self.use_pretrain_label = BodyLabel("使用预训练:", self)
        self.use_pretrain = CheckBox(self)
        self.use_pretrain.setChecked(True)
        self.common_advanced_layout.addWidget(self.use_pretrain_label, 11, 0)
        self.common_advanced_layout.addWidget(self.use_pretrain, 11, 1)
        
        self.config_layout.addWidget(self.common_advanced_widget)
        
        self.button_layout = QHBoxLayout()
        self.start_button = PrimaryPushButton("开始训练", self, FIF.PLAY)
        self.pause_button = PushButton("暂停训练", self, FIF.PAUSE)
        self.stop_button = PushButton("停止训练", self, FIF.CANCEL)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.button_layout.addWidget(self.start_button)
        self.button_layout.addWidget(self.pause_button)
        self.button_layout.addWidget(self.stop_button)
        self.config_layout.addLayout(self.button_layout)
        
        self.progress_card = CardWidget(self)
        self.progress_layout = QVBoxLayout(self.progress_card)
        
        self.progress_title = SubtitleLabel("训练进度", self)
        self.progress_layout.addWidget(self.progress_title)
        
        self.progress_bar = ProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_layout.addWidget(self.progress_bar)
        
        self.progress_info = BodyLabel("训练未开始", self)
        self.progress_layout.addWidget(self.progress_info)
        
        self.left_layout.addWidget(self.config_card)
        self.left_layout.addWidget(self.progress_card)
        self.left_layout.addStretch(1)
        
        self.log_card = CardWidget(self)
        self.log_layout = QVBoxLayout(self.log_card)
        self.log_layout.setContentsMargins(15, 15, 15, 15)
        self.log_layout.setSpacing(10)
        
        self.log_title = SubtitleLabel("训练日志", self)
        self.log_layout.addWidget(self.log_title)
        
        self.log_widget = EnhancedLogWidget(self)
        self.log_layout.addWidget(self.log_widget, 1)
        
        self.content_layout.addWidget(self.left_widget, 1)
        self.content_layout.addWidget(self.log_card, 2)
        
        self.output_browse.clicked.connect(self.browse_output_dir)
        self.model1_browse.clicked.connect(lambda: self.browse_model_path(self.model1_path))
        self.model2_browse.clicked.connect(lambda: self.browse_model_path(self.model2_path))
        self.start_button.clicked.connect(self.start_training)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.stop_button.clicked.connect(self.stop_training)
        
        self.training_thread = None
        self.is_paused = False

        default_output = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trained_models")
        os.makedirs(default_output, exist_ok=True)
        self.output_dir.setText(default_output)
        
        self.output_browse.setIcon(FIF.FOLDER)
        self.model1_browse.setIcon(FIF.FOLDER) 
        self.model2_browse.setIcon(FIF.FOLDER)
        
        self.parameters_interface = getattr(parent, "parametersInterface", None)
    
    def on_epoch_completed(self, model_name, epoch, metrics):
        """训练每轮回调，用于更新参数趋势折线图"""
        # 确保参数界面存在
        if hasattr(self, 'parameters_interface') and self.parameters_interface:
            # 初始化数据存储
            if not hasattr(self, '_trend_epochs'):
                self._trend_epochs = []
                self._trend_values_loss = []
                self._trend_values_accuracy = []
            
            # 添加新的epoch数据点
            self._trend_epochs.append(epoch)
            
            # 添加损失数据
            val_loss = metrics.get('val_loss', 0.0)
            self._trend_values_loss.append(val_loss)
            
            # 添加精度数据(如果有)
            val_accuracy = metrics.get('val_accuracy', metrics.get('accuracy', 0.0))
            self._trend_values_accuracy.append(val_accuracy)
            
            # 向参数界面传递损失数据
            self.parameters_interface.update_parameters(
                self._trend_epochs, 
                self._trend_values_loss,
                'loss'  # 明确指定param_type为loss
            )
            
            # 向参数界面传递精度数据
            if any(v != 0.0 for v in self._trend_values_accuracy):  # 只有当精度数据有效时才传递
                self.parameters_interface.update_parameters(
                    self._trend_epochs,
                    self._trend_values_accuracy,
                    'accuracy'
                )
            
            # 记录日志
            self.log_widget.append(
                f"轮次 {epoch} 完成: 损失={val_loss:.4f}, 精度={val_accuracy:.4f}"
            )
    
    def on_status_update(self, status_text, progress_value):
        """处理状态更新信号"""
        self.progress_info.setText(status_text)
        self.progress_bar.setValue(progress_value)
    
    def _get_training_config(self):
        data_source = 'self_play' if self.data_combobox.currentText() == "自我对弈生成数据" else 'local'
        try:
            learning_rate = float(self.learning_rate.text() or '0.001')
        except ValueError:
            learning_rate = 0.001
        
        config = {
            'data_source': data_source,
            'epochs': self.epochs_spinbox.value(),
            'batch_size': self.batch_spinbox.value(),
            'learning_rate': learning_rate,
            'output_dir': self.output_dir.text(),
            'save_interval': self.save_interval.value(),
            'num_games': self.games_spinbox.value(),
            'mcts_simulations': self.mcts_spinbox.value(),
            'model_size': self.model_size_combobox.currentText(),
            # 添加新参数到配置中
            'selfplay_iterations': self.iterations_spinbox.value(),
            'exploration_temp': self.temp_spinbox.value(),
            'weight_decay': self.weight_decay_spinbox.value(),
            'dropout': self.dropout_spinbox.value(),
            'local_data_path': self.local_data_path.text() if data_source == 'local' else None,
            'pretrained_model': self.model1_path.text() if self.use_pretrain.isChecked() else None
        }
        
        return config
    
    def _validate_config(self):
        config = self._get_training_config()
        
        if not config['output_dir']:
            try:
                default_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trained_models")
                os.makedirs(default_dir, exist_ok=True)
                self.output_dir.setText(default_dir)
                self.log_widget.append(f"已使用默认输出目录: {default_dir}")
            except Exception as e:
                InfoBar.error(
                    title='配置错误',
                    content=f"无法创建默认输出目录: {str(e)}",
                    orient=Qt.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.TOP,
                    duration=3000,
                    parent=self
                )
                return False
        
        try:
            learning_rate = float(config['learning_rate'])
            if learning_rate <= 0 or learning_rate > 1.0:
                self.learning_rate.setText("0.001")
                InfoBar.warning(
                    title='参数重置',
                    content="学习率应在0-1之间，已重置为默认值0.001",
                    orient=Qt.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.TOP,
                    duration=3000,
                    parent=self
                )
        except ValueError:
            self.learning_rate.setText("0.001")
            InfoBar.warning(
                title='参数重置',
                content="学习率格式无效，已重置为默认值0.001",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self
            )
        
        return True
    
    def browse_model_path(self, line_edit):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", 
            "模型文件 (*.pth *.pt *.bin);;所有文件 (*)"
        )
        
        if file_path:
            line_edit.setText(file_path)
            self.log_widget.append(f"已选择模型文件: {file_path}")
            
            # 新增代码：分析模型文件并显示重要信息
            self.analyze_and_show_model_info(file_path)

    def analyze_and_show_model_info(self, model_path):
        """分析模型文件并显示重要信息"""
        try:
            import torch
            import os
            
            # 获取文件大小信息
            file_size_bytes = os.path.getsize(model_path)
            file_size_mb = file_size_bytes / (1024 * 1024)
            self.log_widget.append(f"📊 模型文件大小: {file_size_mb:.2f} MB")
            
            # 加载模型来分析结构
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            try:
                # 尝试直接加载模型状态字典
                state_dict = torch.load(model_path, map_location=device)
                
                # 分析模型结构
                if isinstance(state_dict, dict):
                    num_layers = len(state_dict.keys())
                    self.log_widget.append(f"📊 模型层数: {num_layers}")
                    
                    # 寻找卷积层过滤器数量
                    for key, value in state_dict.items():
                        if 'conv' in key.lower() and '.weight' in key:
                            if len(value.shape) == 4:  # 卷积层权重通常是4维的
                                filters = value.shape[0]
                                self.log_widget.append(f"📊 检测到滤波器数量: {filters}")
                                
                                # 根据滤波器数量估计模型大小类别
                                model_size = "unknown"
                                if filters <= 32:
                                    model_size = "tiny"
                                elif filters <= 64:
                                    model_size = "small"
                                elif filters <= 128:
                                    model_size = "medium"
                                else:
                                    model_size = "large"
                                    
                                self.log_widget.append(f"📊 检测到模型大小: {model_size}")
                                break
                    
                    # 计算总参数量
                    total_params = sum(p.numel() for p in state_dict.values())
                    self.log_widget.append(f"📊 模型总参数: {total_params:,}")
                    
                    # 检测是否有批标准化层
                    has_bn = any('bn' in k.lower() or 'batch' in k.lower() for k in state_dict.keys())
                    if has_bn:
                        self.log_widget.append(f"📊 模型包含批归一化层")
                else:
                    self.log_widget.append("⚠️ 无法分析模型结构 - 不是标准PyTorch状态字典")
                    
            except Exception as e:
                self.log_widget.append(f"⚠️ 加载模型分析失败: {str(e)}")
                
                # 尝试使用更通用的方法 - 从AI工厂加载模型
                try:
                    from ai.ai_factory import load_model
                    self.log_widget.append(f"尝试使用AI工厂加载模型...")
                    
                    # 使用AI工厂加载模型，它会打印模型信息
                    model = load_model(model_path, "selected_model")
                    if model is not None:
                        self.log_widget.append(f"✅ 通过AI工厂成功加载模型")
                except Exception as e2:
                    self.log_widget.append(f"❌ AI工厂加载模型失败: {str(e2)}")
                    
        except Exception as e:
            self.log_widget.append(f"❌ 分析模型时出错: {str(e)}")
    
    def browse_output_dir(self):
        directory = QFileDialog.getExistingDirectory(
            self, "选择输出目录", 
            self.output_dir.text() if hasattr(self, 'output_dir') else "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if directory:
            self.output_dir.setText(directory)
            self.log_widget.append(f"已设置输出目录: {directory}")
            
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                self.log_widget.append(f"创建目录失败: {str(e)}")
                InfoBar.error(
                    title='目录错误',
                    content=f"创建目录失败: {str(e)}",
                    orient=Qt.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.TOP,
                    duration=3000,
                    parent=self
                )
    
    def browse_local_data_dir(self):
        """浏览选择本地训练数据目录"""
        directory = QFileDialog.getExistingDirectory(
            self, "选择训练数据根目录",
            self.local_data_path.text() if self.local_data_path.text() else "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if directory:
            # 设置为根目录，稍后会递归搜索其中的训练数据
            self.local_data_path.setText(directory)
            self.log_widget.append(f"已选择训练数据根目录: {directory}")
            
            # 开始搜索数据目录
            data_dirs = self.search_training_data_dirs(directory)
            
            # 显示找到的数据目录数量
            self.log_widget.append(f"在根目录下找到 {len(data_dirs)} 个有效训练数据目录")
            
            # 如果找到的目录超过5个，显示前5个作为示例
            if data_dirs:
                if len(data_dirs) > 5:
                    examples = data_dirs[:5]
                    self.log_widget.append(f"示例数据目录:")
                    for i, d in enumerate(examples):
                        self.log_widget.append(f"  {i+1}. {os.path.relpath(d, directory)}")
                    self.log_widget.append(f"  ... 等共 {len(data_dirs)} 个目录")
                else:
                    self.log_widget.append(f"数据目录列表:")
                    for i, d in enumerate(data_dirs):
                        self.log_widget.append(f"  {i+1}. {os.path.relpath(d, directory)}")

    def search_training_data_dirs(self, root_dir):
        """递归搜索包含训练数据的目录
        
        Args:
            root_dir: 根目录路径
            
        Returns:
            包含有效训练数据的目录路径列表
        """
        self.log_widget.append(f"开始在 {root_dir} 中搜索训练数据...")
        
        required_files = ['states.npy', 'policies.npy', 'values.npy']
        data_dirs = []
        
        # 计数器，用于显示搜索进度
        dirs_checked = 0
        
        # 使用os.walk递归遍历目录
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirs_checked += 1
            
            # 每检查50个目录显示一次进度
            if dirs_checked % 50 == 0:
                self.log_widget.append(f"已检查 {dirs_checked} 个目录，找到 {len(data_dirs)} 个有效数据目录...")
            
            # 检查当前目录是否包含所有必需的文件
            has_all_files = all(filename in filenames for filename in required_files)
            
            if has_all_files:
                data_dirs.append(dirpath)
        
        self.log_widget.append(f"搜索完成，共检查了 {dirs_checked} 个目录，找到 {len(data_dirs)} 个有效训练数据目录。")
        return data_dirs
        
    def _set_config_enabled(self, enabled):
        self.selfplay_widget.setEnabled(enabled)
        
        self.model1_path.setEnabled(enabled)
        self.model1_browse.setEnabled(enabled)
        self.model2_path.setEnabled(enabled)
        self.model2_browse.setEnabled(enabled)
        
        self.output_dir.setEnabled(enabled)
        self.output_browse.setEnabled(enabled)
        
        if hasattr(self, 'common_advanced_widget'):
            self.common_advanced_widget.setEnabled(enabled)
    
    def start_training(self):
        config = self._get_training_config()
        
        if self.parameters_interface:
            for key, edit in self.parameters_interface.param_edits.items():
                val = edit.text().strip()
                if val:
                    config[key] = type(config.get(key, val))(val)
        
        if not self._validate_config():
            return
        
        self.log_widget.append("===== 开始训练过程... =====")
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self._set_config_enabled(False)
        
        # 添加详细的训练参数记录
        self.log_widget.append("===== 训练参数配置 =====")
        self.log_widget.append(f"📊 数据来源: {'本地数据' if config['data_source'] == "local" else '自我对弈'}")
        if config['data_source'] == 'local':
            self.log_widget.append(f"📂 本地数据路径: {config['local_data_path']}")
        else:
            self.log_widget.append(f"🎮 对弈局数: {config['num_games']} 局")
            self.log_widget.append(f"🔄 自我对弈迭代次数: {config.get('selfplay_iterations', 1)} 次")
            self.log_widget.append(f"🧠 MCTS模拟次数: {config['mcts_simulations']} 次/步")
            self.log_widget.append(f"🌡️ 探索温度: {config.get('exploration_temp', 1.0)}")
        self.log_widget.append(f"🧩 模型大小: {config.get('model_size', 'tiny')}")  # 添加模型大小日志
        self.log_widget.append(f"📦 批次大小: {config['batch_size']}")
        self.log_widget.append(f"📈 学习率: {config['learning_rate']}")
        self.log_widget.append(f"🔄 训练轮次: {config['epochs']}")
        self.log_widget.append(f"💾 模型保存间隔: {config['save_interval']} 轮")
        self.log_widget.append(f"⚙️ 优化器: {self.optimizer_type.currentText()}")
        self.log_widget.append(f"📝 权重衰减: {config.get('weight_decay', '0.0005')}")
        self.log_widget.append(f"🎭 Dropout率: {config.get('dropout', '0.3')}")
        self.log_widget.append(f"🧩 使用预训练: {'是' if self.use_pretrain.isChecked() else '否'}")
        self.log_widget.append(f"📂 输出目录: {config['output_dir']}")
        self.log_widget.append("===== 训练详细日志 =====")
        
        self.training_thread = TrainingThread(config)
        self.training_thread.progress_updated.connect(self.on_progress_updated)
        self.training_thread.log_message.connect(self.on_log_message)
        self.training_thread.training_completed.connect(self.on_training_completed)
        self.training_thread.board_updated.connect(self.update_monitor_board)  # 连接新的信号
        
        # 关键修改：确保每次创建新的训练线程后立即连接信号
        self.training_thread.training_epoch_completed.connect(self.on_epoch_completed)
        self.training_thread.status_update_signal.connect(self.on_status_update)
        
        self.training_thread.start()
        self.is_paused = False
        
        self.log_widget.append("===== 训练开始 =====")
        self.log_widget.append(f"🔍 提示: 小样本量训练主要用于测试模型保存功能是否正常工作")
        
        # 更新状态显示
        self.progress_info.setText("训练已开始")
        self.progress_bar.setValue(0)
        QApplication.processEvents()  # 强制更新UI
        
        InfoBar.success(
            title='训练开始',
            content="AI模型训练已开始",
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=3000,
            parent=self
        )
    
    def toggle_pause(self):
        if not self.training_thread:
            return
        
        if self.is_paused:
            self.training_thread.resume()
            self.pause_button.setText("暂停训练")
            self.pause_button.setIcon(FIF.PAUSE)
            self.is_paused = False
            
            InfoBar.info(
                title='训练已恢复',
                content="训练进程已继续",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self
            )
            self.log_widget.append("训练已恢复")
            self.progress_info.setText("训练进行中...")
        else:
            self.training_thread.pause()
            self.pause_button.setText("继续训练")
            self.pause_button.setIcon(FIF.PLAY)
            self.is_paused = True
            
            InfoBar.info(
                title='训练已暂停',
                content="训练进程已暂停",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self
            )
            self.log_widget.append("训练已暂停")
            self.progress_info.setText("训练已暂停")
    
    def stop_training(self):
        if not self.training_thread:
            return
            
        reply = QMessageBox.question(
            self, '确认停止', 
            "确定要停止当前训练吗？进度将会丢失。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.training_thread.stop()
            self.log_widget.append("正在停止训练...")
            
            # 禁用各种按钮，防止重复点击
            self.start_button.setEnabled(False)
            self.pause_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            
            # 显示一个小提示，让用户知道正在停止训练
            InfoBar.info(
                title='正在停止训练',
                content="正在安全停止训练过程，请稍候...",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP_RIGHT,
                duration=2000,
                parent=self
            )
            
            # 启动定时器，等待一段时间后检查线程状态
            QTimer.singleShot(3000, self._check_training_stopped)

    def _check_training_stopped(self):
        """检查训练是否已停止，如果未停止则强制终止"""
        if self.training_thread and self.training_thread.isRunning():
            self.log_widget.append("训练未及时停止，正在强制终止...")
            
            # 尝试强制终止线程
            self.training_thread.terminate()
            self.training_thread.wait(1000)
            
            # 无论成功与否，都重置UI
            self.start_button.setEnabled(True)
            self.pause_button.setText("暂停训练")
            self.pause_button.setIcon(FIF.PAUSE)
            self.pause_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self._set_config_enabled(True)
            
            InfoBar.warning(
                title='训练已强制停止',
                content="训练过程已被强制终止，可能未正确保存模型",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self
            )
            
            # 重置线程对象
            self.training_thread = None
        else:
            # 如果已经停止，不需要做任何事情
            pass
    
    def on_progress_updated(self, progress, info):
        self.progress_bar.setValue(progress)
        epoch = info.get('epoch', 0)
        batch = info.get('batch', 0)
        loss = info.get('loss', 0.0)
        accuracy = info.get('accuracy', 0.0)
        if 'total_batches' in info:
            total_batches = info['total_batches']
            self.progress_info.setText(f"进度: {progress}% | Epoch: {epoch} | Batch: {batch}/{total_batches} | 损失: {loss:.4f} | 精度: {accuracy:.4f}")
        else:
            self.progress_info.setText(f"进度: {progress}% | Epoch: {epoch} | 损失: {loss:.4f} | 精度: {accuracy:.4f}")
    
    def update_progress_text(self, text):
        """更新进度文本"""
        self.progress_info.setText(text)
    
    def update_progress_value(self, value):
        """更新进度条值"""
        self.progress_bar.setValue(value)
    
    def on_log_message(self, message):
        self.log_widget.append(message)
    
    def on_training_completed(self, success, message):
        if success:
            self.log_widget.append("===== 训练已成功完成 =====")
            
            InfoBar.success(
                title='训练完成',
                content="AI模型训练已成功完成",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self
            )
            
            output_dir = self.output_dir.text().strip()
            models_dir = os.path.join(output_dir, 'models')
            if os.path.exists(models_dir):
                InfoBar.info(
                    title='模型已保存',
                    content=f"模型已成功保存到: {models_dir}",
                    orient=Qt.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.BOTTOM,
                    duration=5000,
                    parent=self
                )
                
                open_folder_btn = PushButton("在文件夹中查看", self, FIF.FOLDER)
                open_folder_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(models_dir)))
                self.progress_layout.addWidget(open_folder_btn)
        else:
            self.log_widget.append(f"===== 训练未完成 =====")
            self.log_widget.append(f"原因: {message}")
            
            InfoBar.error(
                title='训练中断',
                content=message,
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self
            )
        
        self._set_config_enabled(True)
        self.training_thread = None
        self.is_paused = False
        self.pause_button.setText("暂停训练")
        self.pause_button.setIcon(FIF.PAUSE)
        self.stop_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.start_button.setEnabled(True)
    
    def update_monitor_board(self, board_data, moves, current_player):
        """更新监控棋盘状态，仅用于可视化，不再写日志"""
        try:
            # 检查参数界面是否存在
            if hasattr(self, 'parameters_interface') and self.parameters_interface:
                # 检查是否有boardSignal信号
                if hasattr(self.parameters_interface, 'boardSignal'):
                    # 发送信号到参数界面更新棋盘
                    self.parameters_interface.boardSignal.emit(board_data, moves, current_player)
            else:
                main_window = self.window()
                if hasattr(main_window, 'parametersInterface'):
                    if hasattr(main_window.parametersInterface, 'boardSignal'):
                        main_window.parametersInterface.boardSignal.emit(board_data, moves, current_player)
        except Exception as e:
            self.log_widget.append(f"更新监控棋盘失败: {str(e)}")

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    win = TrainingInterface()
    win.setWindowTitle("AI训练中心 - 独立模式")
    win.resize(1600, 1200)
    win.show()
    sys.exit(app.exec_())