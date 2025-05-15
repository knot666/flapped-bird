import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
import pickle
import os
import time

# 改进的神经网络结构，增加网络深度和节点数
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        
        # 更深的网络结构，但保持简单
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.output = nn.Linear(64, output_size)
        
        # 使用较小的dropout
        self.dropout = nn.Dropout(0.05)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        return self.output(x)

class Transition:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.priority = 1.0  # 初始优先级设为1.0

# 优先级经验回放
class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.7, beta=0.5, beta_increment=0.002, init_memory=None):
        """
        alpha: 控制优先级使用程度 (0 - 纯随机采样, 1 - 仅用优先级)
        beta: 控制重要性采样权重 (0 - 无补偿, 1 - 完全补偿)
        """
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.memory = deque(maxlen=capacity)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.capacity = capacity
        
        # 如果提供了初始记忆，加载它
        if init_memory:
            for item in init_memory:
                if len(self.memory) < self.capacity:
                    if isinstance(item, Transition):
                        self.push(item.state, item.action, item.reward, item.next_state, item.done)
                    elif isinstance(item, tuple) and len(item) == 5:
                        self.push(*item)

    def push(self, state, action, reward, next_state, done):
        """添加新的经验，对高奖励和得分高的经验增加优先级"""
        max_priority = np.max(self.priorities) if self.memory else 1.0
        
        # 检查是否包含得分信息（增强状态）
        score_bonus = 0.0
        if len(state) > 0 and state[-1] >= 0:  # 最后一个元素是得分
            score = state[-1] * 10  # 还原归一化的得分
            # 为高得分经验增加优先级
            score_bonus = score * 0.5  # 得分越高，优先级越高
        
        # 为高奖励增加优先级
        reward_bonus = 0.0
        if reward > 10:  # 高奖励（通过管道）
            reward_bonus = 2.0
            
        # 最终优先级
        priority = max_priority + score_bonus + reward_bonus
        
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(state, action, reward, next_state, done))
        else:
            self.memory[self.position] = Transition(state, action, reward, next_state, done)
        
        # 设置优先级
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """优先级采样"""
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.memory)]
        
        # 计算采样概率
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 抽样和计算重要性权重
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        
        # 增加beta值
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 计算重要性权重
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # 归一化权重
        
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        """更新采样经验的优先级"""
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority

    def __len__(self):
        return len(self.memory)
        
    def __iter__(self):
        return iter(self.memory)

class DQNAgent:
    """改进版深度Q网络代理类"""
    def save_model(self, filename):
        """保存模型参数到文件"""
        try:
            # 获取当前全局的episode_rewards列表长度作为回合数
            # 由于episode_rewards是在主文件中定义的，我们需要从外部获取
            import sys
            episode_count = 0
            # 尝试从主模块获取episode_rewards
            if 'episode_rewards' in sys.modules['__main__'].__dict__:
                episode_count = len(sys.modules['__main__'].__dict__['episode_rewards'])
            
            torch.save({
                'policy_net': self.policy_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'steps': self.steps,
                'epsilon': self.epsilon,
                'training_stats': self.training_stats,
                'high_scores': getattr(self, 'high_scores', []),
                'episode_count': episode_count  # 添加回合数保存
            }, filename)
            print(f"成功保存模型到 {filename}")
        except Exception as e:
            print(f"保存模型失败：{e}")

    def __init__(self, state_size, action_size, load_memory=None, load_model=None):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 初始化网络
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        
        # 使用更小的学习率
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=0.0001, weight_decay=0.001)
        
        # 训练统计数据
        self.training_stats = {
            'rewards': [],
            'total_rewards': [],
            'scores': [],
            'losses': [],
            'epsilons': [],
            'avg_q_values': []
        }
        
        # 高分记录
        self.high_scores = []
        
        # 加载已有模型参数
        if load_model:
            try:
                checkpoint = torch.load(load_model, map_location=self.device)
                self.policy_net.load_state_dict(checkpoint['policy_net'])
                self.target_net.load_state_dict(checkpoint['target_net'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.steps = checkpoint.get('steps', 0)
                self.epsilon = checkpoint.get('epsilon', 1.0)
                self.training_stats = checkpoint.get('training_stats', self.training_stats)
                self.high_scores = checkpoint.get('high_scores', [])
                print(f"成功加载模型参数：{load_model}")
            except FileNotFoundError:
                print(f"模型文件 {load_model} 不存在，使用初始参数")
                self.steps = 0
                self.epsilon = 1.0
            except Exception as e:
                print(f"加载模型失败：{e}，使用初始参数")
                self.steps = 0
                self.epsilon = 1.0
        else:
            self.steps = 0
            self.epsilon = 1.0
            
        # 同步目标网络参数
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 使用优先级经验回放
        self.memory = PrioritizedReplayMemory(capacity=100000, 
                                             alpha=0.6,  # 降低优先级影响
                                             beta=0.4,   # 降低初始beta值
                                             beta_increment=0.001,  # 更慢地提高beta
                                             init_memory=load_memory.memory if load_memory else None)
        if load_memory is not None:
            print(f"成功加载记忆池，共{len(self.memory)}条经验")

        # 超参数
        self.batch_size = 64  # 减小批量大小
        self.gamma = 0.99     # 保持折扣因子
        self.target_update_freq = 1  # 更频繁地更新目标网络
        
        # 确保日志目录存在
        os.makedirs('logs', exist_ok=True)
        
        # 创建训练日志文件
        self.log_file = os.path.join('logs', f'training_log_{time.strftime("%Y%m%d_%H%M%S")}.csv')
        with open(self.log_file, 'w') as f:
            f.write("step,loss,reward,epsilon,avg_q_value,memory_size,score\n")

    def select_action(self, state, epsilon):
        """选择动作（带ε-贪婪策略）"""
        self.steps += 1
        if random.random() < epsilon:
            return 0
        else:
            with torch.no_grad():
                state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
                q_values = self.policy_net(state)
                # 记录平均Q值
                if len(self.training_stats['avg_q_values']) < 1000:
                    self.training_stats['avg_q_values'].append(q_values.mean().item())
                else:
                    self.training_stats['avg_q_values'] = self.training_stats['avg_q_values'][1:] + [q_values.mean().item()]
                return q_values.argmax().item()

    def optimize_model(self):
        """训练模型（使用优先级回放）"""
        if len(self.memory) < self.batch_size:
            return None

        # 使用优先级采样
        transitions, indices, weights = self.memory.sample(self.batch_size)
        weights = torch.tensor(weights, device=self.device, dtype=torch.float32)
        
        # 解包并转换数据为张量
        state_batch = torch.tensor([t.state for t in transitions], device=self.device, dtype=torch.float32)
        action_batch = torch.tensor([t.action for t in transitions], device=self.device, dtype=torch.long)
        reward_batch = torch.tensor([t.reward for t in transitions], device=self.device, dtype=torch.float32)
        next_state_batch = torch.tensor([t.next_state for t in transitions], device=self.device, dtype=torch.float32)
        done_batch = torch.tensor([t.done for t in transitions], device=self.device, dtype=torch.float32)

        # 计算当前Q值
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值（Double DQN）
        next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
        next_q_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze(1).detach()
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        # 计算时序差分误差（用于更新优先级）
        td_errors = torch.abs(current_q_values - expected_q_values).detach().cpu().numpy()
        
        # 使用Huber损失（对异常值更鲁棒）
        loss = F.smooth_l1_loss(current_q_values, expected_q_values, reduction='none')
        # 应用重要性采样权重
        weighted_loss = (loss * weights).mean()
        
        # 反向传播
        self.optimizer.zero_grad()
        weighted_loss.backward()
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.optimizer.step()

        # 更新优先级
        self.memory.update_priorities(indices, td_errors + 1e-5)  # 添加小常数防止零优先级

        # 目标网络更新逻辑
        if self.steps % self.target_update_freq == 0:
            # 软更新：部分更新目标网络参数
            tau = 0.1  # 软更新系数
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)

        # 记录训练统计
        loss_value = weighted_loss.item()
        if len(self.training_stats['losses']) < 1000:
            self.training_stats['losses'].append(loss_value)
        else:
            self.training_stats['losses'] = self.training_stats['losses'][1:] + [loss_value]

        return loss_value

    def record_score(self, score, total_reward):
        """记录游戏得分和总奖励"""
        if len(self.training_stats['scores']) < 1000:
            self.training_stats['scores'].append(score)
            self.training_stats['total_rewards'].append(total_reward)
        else:
            self.training_stats['scores'] = self.training_stats['scores'][1:] + [score]
            self.training_stats['total_rewards'] = self.training_stats['total_rewards'][1:] + [total_reward]
            
        # 记录高分
        if score > 0 and (not self.high_scores or score > max(self.high_scores)):
            self.high_scores.append(score)
            self.high_scores.sort(reverse=True)
            self.high_scores = self.high_scores[:10]  # 只保留前10个高分 

    def get_avg_q_value(self):
        """获取最近的平均Q值"""
        if not self.training_stats['avg_q_values']:
            return 0.0
        return sum(self.training_stats['avg_q_values']) / len(self.training_stats['avg_q_values']) 