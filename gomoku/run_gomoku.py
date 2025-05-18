# coding:utf-8
import sys
import os
import signal
import atexit
import time
import threading

# 设置OpenMP环境变量，解决多个OpenMP运行时冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 设置PyTorch线程数，避免过多线程导致的性能问题
os.environ["OMP_NUM_THREADS"] = "4"  # OpenMP线程
os.environ["MKL_NUM_THREADS"] = "4"  # MKL线程

# 配置中文字体支持
import matplotlib
font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resource', 'fonts', 'msyh.ttf')
if os.path.exists(font_path):
    matplotlib.font_manager.fontManager.addfont(font_path)
    matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
else:
    try:
        from matplotlib.font_manager import FontProperties
        fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist
                if ('SimSun' in f.name or 'SimHei' in f.name or 'Microsoft YaHei' in f.name)]
        if fonts:
            matplotlib.rcParams['font.family'] = fonts[0]
    except:
        print("WARNING: 未能找到合适的中文字体，图表中的中文可能无法正确显示")

# 全局变量，用于存储应用实例
app = None
# 标记程序是否正在关闭
is_exiting = False
# 用于等待线程终止的最大时间（秒）
MAX_WAIT_TIME = 3

def safe_exit(wait_time=MAX_WAIT_TIME):
    """安全退出程序，确保所有资源都被正确释放
    
    Args:
        wait_time: 等待线程终止的最大时间（秒）
    """
    global is_exiting
    
    # 如果已经在退出过程中，避免重复执行
    if is_exiting:
        return
        
    is_exiting = True
    print("\n程序正在安全退出...")
    
    # 确保PyTorch相关资源得到清理
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("已清空CUDA缓存")
    except:
        pass
    
    # 尝试终止所有训练线程
    try:
        from mainWindow.training_interface import stop_all_training_threads
        print("正在停止所有训练线程...")
        stop_all_training_threads()
    except Exception as e:
        print(f"停止训练线程时出错: {e}")
    
    # 在单独的线程中等待一段时间，确保资源有时间释放
    def delayed_exit():
        time.sleep(0.5)  # 短暂等待让UI更新
        print(f"等待最多 {wait_time} 秒让资源释放...")
        
        # 强制释放PyTorch资源的最后尝试
        try:
            import gc
            gc.collect()
            
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("再次清空CUDA缓存")
        except:
            pass
        
        # 设置一个短暂的延迟，让线程有时间终止
        if app:
            app.quit()
        else:
            sys.exit(0)
    
    # 启动延迟退出线程
    if app:
        # 对于Qt应用，使用QTimer在主线程中执行退出
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(wait_time * 1000, app.quit)
    else:
        # 否则，使用普通线程
        exit_thread = threading.Thread(target=delayed_exit)
        exit_thread.daemon = True
        exit_thread.start()

def signal_handler(sig, frame):
    """处理中断信号（Ctrl+C）"""
    print("\n程序接收到中断信号，将安全退出...")
    # 使用单次计时器在主线程中执行退出操作
    if app:
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, safe_exit)
    else:
        safe_exit()

def setup_high_dpi():
    """设置高DPI支持"""
    # 确保高分辨率显示器上的缩放处理正确
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    # 启用高DPI缩放
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    # 使用高DPI图像
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

def setup_signal_handling():
    """设置信号处理"""
    # 注册SIGINT信号处理器(处理Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)
    
    # 在Windows上，增加SIGBREAK(Ctrl+Break)信号处理
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, signal_handler)
    
    # 注册退出处理
    atexit.register(safe_exit)

def setup_pytorch_optimizations():
    """设置PyTorch优化参数"""
    try:
        import torch
        
        # 打印PyTorch信息，有助于调试
        device_info = f"PyTorch {torch.__version__}, "
        device_info += f"CUDA {'可用' if torch.cuda.is_available() else '不可用'}"
        if torch.cuda.is_available():
            device_info += f" ({torch.cuda.get_device_name(0)})"
        print(device_info)
        
        # 设置较小的线程数
        torch.set_num_threads(4)
        
        # 使用TensorFloat32精度以提高性能
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # 启用cudnn基准
            torch.backends.cudnn.benchmark = True
    except ImportError:
        print("未安装PyTorch，跳过PyTorch优化")
    except Exception as e:
        print(f"设置PyTorch优化时出错: {e}")

def main():
    """程序主入口 - 简化并提高健壮性"""
    global app
    
    # 设置信号处理
    setup_signal_handling()
    
    # 设置高DPI支持
    setup_high_dpi()
    
    # 设置PyTorch优化
    setup_pytorch_optimizations()

    # 确保应用所有目录结构都存在且有正确的权限
    app_root = os.path.dirname(os.path.abspath(__file__))
    default_output = os.path.join(app_root, "trained_models")
    models_dir = os.path.join(default_output, "models")
    logs_dir = os.path.join(default_output, "logs")
    history_dir = os.path.join(default_output, "game_history")
    training_data_dir = os.path.join(default_output, "training_data")
    charts_dir = os.path.join(app_root, "charts")
    
    # 创建所有必要的目录结构
    for directory in [default_output, models_dir, logs_dir, history_dir, training_data_dir, charts_dir]:
        os.makedirs(directory, exist_ok=True)
        
    # 验证目录权限
    if not os.access(models_dir, os.W_OK):
        print(f"警告: 没有写入权限到模型目录 {models_dir}")
    
    # 确保训练线程使用正确的输出目录
    config = {}
    config['output_dir'] = models_dir
    
    # 使用一致的斜杠格式
    print(f"程序目录: {os.path.normpath(app_root)}")
    print(f"模型存储目录: {os.path.normpath(models_dir)}")
    print(f"历史记录目录: {os.path.normpath(history_dir)}")

    try:
        # 创建应用实例并保存到全局变量
        from PyQt5.QtWidgets import QApplication
        app = QApplication(sys.argv)
        
        # 设置应用属性
        app.setApplicationName("五子棋AI")
        app.setOrganizationName("BUPT")
        
        # 导入窗口类
        from mainWindow.main_window import Window
        w = Window()
        w.show()
        
        # 创建一个定时处理信号的对象，与主事件循环集成
        from PyQt5.QtCore import QTimer
        signal_timer = QTimer()
        signal_timer.setInterval(500)  # 每500毫秒检查一次信号
        signal_timer.timeout.connect(lambda: None)  # 空函数，允许信号处理
        signal_timer.start()
        
        # 确保在退出前清理资源
        app.aboutToQuit.connect(lambda: [signal_timer.stop()])
        
        # 进入应用程序主循环
        return app.exec_()
    except Exception as e:
        print(f"程序启动失败: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
