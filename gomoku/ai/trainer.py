import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime

class GomokuTrainer:
    """五子棋模型训练器"""
    
    def __init__(self, model, device=None, learning_rate=0.001, config=None):
        """初始化训练器
        
        Args:
            model: 策略价值网络模型
            device: 计算设备 (CPU/GPU)
            learning_rate: 学习率
            config: 额外配置参数
        """
        self.model = model
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # 损失函数
        self.value_criterion = nn.MSELoss()  # 价值网络使用均方误差
        self.policy_loss_fn = nn.KLDivLoss(reduction='batchmean')  # 改为 KL散度损失，用于 soft‑target
        
        # 训练统计
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # 保存额外配置参数
        self.learning_rate = learning_rate
        self.config = config or {}
        print(f"GomokuTrainer初始化，配置: {self.config}")
        
        # 记录保存间隔
        self.save_interval = self.config.get('save_interval', 5)
        print(f"模型保存间隔: {self.save_interval}")
    
    def train_batch(self, states, policies, values):
        """训练单个批次
        
        Args:
            states: 棋盘状态张量, [batch_size, 3, board_size, board_size]
            policies: 策略目标张量, [batch_size, board_size*board_size]
            values: 价值目标张量, [batch_size, 1]
            
        Returns:
            tuple: (总损失, 策略损失, 价值损失, 策略准确率)
        """
        # 确保输入在正确设备上
        states = states.to(self.device)
        policies = policies.to(self.device)
        values = values.to(self.device)
        
        # 确保values的形状一致 - 修复形状不匹配警告
        if values.dim() == 1:
            values = values.unsqueeze(1)  # 将 [batch_size] 转换为 [batch_size, 1]
        
        # 设置为训练模式
        self.model.train()
        
        # 前向传播
        log_ps, vs = self.model(states)  # log softmax 已在模型里
        
        # 计算损失
        policy_loss = self.policy_loss_fn(log_ps, policies)  # 使用 KLDiv 计算策略损失（输入为 log_probs, target 为概率分布）
        value_loss = self.value_criterion(vs, values)  # 此处values现在已经是[batch_size, 1]形状
        
        # 总损失
        loss = policy_loss + value_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 计算策略准确率
        _, predicted = torch.max(log_ps, 1)
        _, targets = torch.max(policies, 1)
        accuracy = (predicted == targets).float().mean().item()
        
        return loss.item(), policy_loss.item(), value_loss.item(), accuracy
    
    def evaluate(self, loader):
        """评估模型
        
        Args:
            loader: 验证数据加载器
            
        Returns:
            tuple: (总损失, 策略损失, 价值损失, 策略准确率)
        """
        self.model.eval()
        total_loss = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        accuracy_sum = 0.0
        count = 0
        
        with torch.no_grad():
            for states, policies, values in loader:
                # 将数据移到设备上
                states = states.to(self.device)
                policies = policies.to(self.device)
                values = values.to(self.device)
                
                # 确保values的形状一致 - 修复形状不匹配警告
                if values.dim() == 1:
                    values = values.unsqueeze(1)  # 将 [batch_size] 转换为 [batch_size, 1]
                
                # 前向传播
                log_ps, vs = self.model(states)
                
                # 计算损失
                policy_loss = self.policy_loss_fn(log_ps, policies)
                value_loss = self.value_criterion(vs, values)
                loss = policy_loss + value_loss
                
                # 计算准确率
                _, predicted = torch.max(log_ps, 1)
                _, targets = torch.max(policies, 1)
                accuracy = (predicted == targets).float().mean().item()
                
                # 累加统计
                total_loss += loss.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                accuracy_sum += accuracy
                count += 1
        
        # 计算平均值
        return (
            total_loss / count,
            policy_loss_sum / count,
            value_loss_sum / count,
            accuracy_sum / count
        )
    
    def train(self, train_loader, val_loader, num_epochs=10, save_dir=None, callback=None, save_model_callback=None):
        """训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮次
            save_dir: 模型保存目录
            callback: 回调函数 - func(event_type, event_data) -> bool
            save_model_callback: 保存模型的回调函数 - func(model, epoch, metrics, is_best) -> bool
            
        Returns:
            dict: 训练统计信息
        """
        print(f"开始训练，总轮次: {num_epochs}, 保存目录: {save_dir}, 保存间隔: {self.save_interval}")
        
        # 从配置中获取保存间隔
        save_interval = self.config.get('save_interval', 5)
        # 新增：打印完整训练参数
        print("训练参数:",
              f"学习率={self.learning_rate},",
              f"保存间隔={save_interval},",
              f"权重衰减={self.config.get('weight_decay')},",
              f"Dropout={self.config.get('dropout')},",
              f"优化器={self.config.get('optimizer')},",
              f"模型大小={self.config.get('model_size')}"
        )
        
        # 确认输入数据加载器是否有效
        if len(train_loader) == 0:
            print("⚠️ 警告: 训练数据加载器为空")
            if callback:
                callback('error', {'message': '训练数据加载器为空'})
            return {'error': 'empty_train_loader'}
            
        # 验证保存目录
        if save_dir:
            try:
                os.makedirs(save_dir, exist_ok=True)
                print(f"确保保存目录存在: {save_dir}")
                
                # 检查目录权限
                if os.access(save_dir, os.W_OK):
                    print(f"✅ 保存目录有写权限: {save_dir}")
                    
                    # 保存初始模型作为安全保障
                    init_path = os.path.join(save_dir, f'initial_model.pth')
                    try:
                        torch.save(self.model.state_dict(), init_path)
                        print(f"已保存初始模型: {init_path}")
                    except Exception as e:
                        print(f"保存初始模型失败: {str(e)}")
                else:
                    print(f"❌ 警告: 保存目录没有写权限: {save_dir}")
                    # 尝试使用临时目录
                    import tempfile
                    temp_dir = tempfile.gettempdir()
                    print(f"尝试使用临时目录作为备选: {temp_dir}")
                    save_dir = temp_dir
                    
                    # 再次测试临时目录的权限
                    if os.access(save_dir, os.W_OK):
                        print(f"✅ 临时目录有写权限")
                    else:
                        print(f"❌ 警告: 临时目录也没有写权限")
            except Exception as e:
                print(f"❌ 创建保存目录出错: {e}")
                import tempfile
                save_dir = tempfile.gettempdir()
                print(f"使用临时目录: {save_dir}")
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        # 提前创建目录
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            # 测试写入权限
            try:
                test_file = os.path.join(save_dir, "test_write.tmp")
                with open(test_file, 'w') as f:
                    f.write("测试写入权限")
                os.remove(test_file)
                print(f"✅ 写入测试成功: {save_dir}")
            except Exception as e:
                print(f"❌ 写入测试失败: {e}")
        
        # 使用配置中的save_interval，而不是硬编码值
        print(f"使用配置的保存间隔: {save_interval}")
        
        try:
            for epoch in range(1, num_epochs + 1):
                print(f"开始训练第 {epoch}/{num_epochs} 轮...")
                epoch_start_time = time.time()
                
                # 训练前检查是否需要中断
                if callback and not callback('epoch_start', {'epoch': epoch}):
                    print("训练被外部中断")
                    return {
                        'train_losses': self.train_losses,
                        'val_losses': self.val_losses,
                        'train_accuracies': self.train_accuracies,
                        'val_accuracies': self.val_accuracies,
                        'epochs_completed': epoch - 1
                    }
                
                # 进入训练模式
                self.model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                # 训练一个epoch
                for batch_idx, (states, policies, values) in enumerate(train_loader):
                    # 检查是否需要中断
                    if callback and not callback('batch_start', {
                        'epoch': epoch, 
                        'batch': batch_idx,
                        'total_batches': len(train_loader)
                    }):
                        print("训练被批次回调中断")
                        return {
                            'train_losses': self.train_losses,
                            'val_losses': self.val_losses,
                            'train_accuracies': self.train_accuracies,
                            'val_accuracies': self.val_accuracies,
                            'epochs_completed': epoch - 1,
                            'batch_completed': batch_idx
                        }
                    
                    # 将数据发送到设备
                    states = states.to(self.device)
                    policies = policies.to(self.device)
                    values = values.to(self.device)
                    
                    # 清除梯度
                    self.optimizer.zero_grad()
                    
                    # 前向传播
                    log_ps, vs = self.model(states)
                    
                    # 计算策略损失 (交叉熵)
                    policy_loss = F.nll_loss(log_ps, policies.argmax(dim=1))
                    
                    # 计算价值损失 (均方误差)
                    value_loss = F.mse_loss(vs.view(-1), values.view(-1))
                    
                    # 总损失 = 策略损失 + 价值损失
                    loss = policy_loss + value_loss
                    
                    # 反向传播
                    loss.backward()
                    
                    # 更新参数
                    self.optimizer.step()
                    
                    # 累加训练统计
                    train_loss += loss.item()
                    _, predicted = log_ps.max(1)
                    train_total += policies.size(0)
                    train_correct += (predicted == policies.argmax(dim=1)).sum().item()
                    
                    # 批次结束回调
                    if callback and not callback('batch_end', {
                        'epoch': epoch, 
                        'batch': batch_idx + 1,
                        'total_batches': len(train_loader),
                        'loss': loss.item(),
                        'policy_loss': policy_loss.item(),
                        'value_loss': value_loss.item()
                    }):
                        print("训练在批次结束时被中断")
                        return {
                            'train_losses': self.train_losses,
                            'val_losses': self.val_losses,
                            'train_accuracies': self.train_accuracies,
                            'val_accuracies': self.val_accuracies,
                            'epochs_completed': epoch - 1,
                            'batch_completed': batch_idx + 1
                        }
                
                # 计算训练指标
                train_loss /= len(train_loader)
                train_accuracy = 100.0 * train_correct / train_total if train_total > 0 else 0
                
                # 进入评估模式
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                # 评估模型
                with torch.no_grad():
                    for states, policies, values in val_loader:
                        states = states.to(self.device)
                        policies = policies.to(self.device)
                        values = values.to(self.device)
                        
                        log_ps, vs = self.model(states)
                        
                        # 计算策略损失
                        policy_loss = F.nll_loss(log_ps, policies.argmax(dim=1))
                        
                        # 计算价值损失
                        value_loss = F.mse_loss(vs.view(-1), values.view(-1))
                        
                        # 总损失
                        loss = policy_loss + value_loss
                        val_loss += loss.item()
                        
                        # 统计准确率
                        _, predicted = log_ps.max(1)
                        val_total += policies.size(0)
                        val_correct += (predicted == policies.argmax(dim=1)).sum().item()
                
                # 计算验证指标
                val_loss /= len(val_loader)
                val_accuracy = 100.0 * val_correct / val_total if val_total > 0 else 0
                
                # 记录指标
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.train_accuracies.append(train_accuracy)
                self.val_accuracies.append(val_accuracy)
                
                # 计算epoch用时
                epoch_time = time.time() - epoch_start_time
                
                # 打印训练信息
                print(f"Epoch {epoch}/{num_epochs}, "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%, "
                      f"Time: {epoch_time:.2f}s")
                      
                # epoch结束回调
                if callback:
                    continue_training = callback('epoch_end', {
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'train_accuracy': train_accuracy,
                        'val_accuracy': val_accuracy,
                        'time': epoch_time
                    })
                    
                    if not continue_training:
                        print(f"训练在第 {epoch} 轮结束时被中断")
                        break
                
                # 模型保存 - 使用配置的save_interval
                try:
                    if save_dir:
                        # 创建子目录，基于日期和模型大小
                        model_size = self.config.get('model_size', 'tiny') if hasattr(self, 'config') else 'default'
                        date_str = datetime.now().strftime("%Y%m%d")
                        model_subdir = f"{date_str}_{model_size}"
                        save_subdir = os.path.join(save_dir, model_subdir)
                        os.makedirs(save_subdir, exist_ok=True)
                        
                        # 根据配置的间隔保存，或者是最后一轮
                        if epoch % save_interval == 0 or epoch == num_epochs:
                            epoch_path = os.path.join(save_subdir, f'model_epoch_{epoch}.pth')
                            print(f"正在保存第 {epoch} 轮模型...")
                            torch.save(self.model.state_dict(), epoch_path)
                            print(f"✅ 模型已保存: {epoch_path}")
                        
                        # 如果是最佳模型，也保存为best_model
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_path = os.path.join(save_subdir, 'best_model.pth')
                            torch.save(self.model.state_dict(), best_path)
                            print(f"✅ 最佳模型已保存: {best_path}")
                            
                            # 额外在根目录保存一个副本，方便引用
                            root_best_path = os.path.join(save_dir, f'best_{model_size}_{date_str}.pth')
                            torch.save(self.model.state_dict(), root_best_path)
                            print(f"✅ 根目录最佳模型已保存: {root_best_path}")
                            
                            # 调用保存模型回调
                            if save_model_callback:
                                save_model_callback(self.model, epoch, 
                                                   {'val_loss': val_loss, 'train_loss': train_loss}, 
                                                   is_best=True)
                        
                        # 使用自定义保存回调
                        if save_model_callback and (epoch % save_interval == 0 or epoch == num_epochs):
                            save_model_callback(self.model, epoch, 
                                               {'val_loss': val_loss, 'train_loss': train_loss},
                                               is_best=False)
                except Exception as e:
                    print(f"❌ 保存模型时出错: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
            
            # 计算总用时
            total_time = time.time() - start_time
            print(f"训练完成，总用时: {total_time:.2f}秒")
            
            # 返回训练统计
            return {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accuracies': self.train_accuracies,
                'val_accuracies': self.val_accuracies,
                'total_time': total_time,
                'epochs_completed': num_epochs
            }
            
        except Exception as e:
            print(f"❌ 训练过程发生错误: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
            return {
                'error': str(e),
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accuracies': self.train_accuracies,
                'val_accuracies': self.val_accuracies,
                'epochs_completed': epoch if 'epoch' in locals() else 0
            }
