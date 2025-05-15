import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import argparse
from matplotlib.font_manager import FontProperties

# 配置中文字体支持
def setup_chinese_font():
    try:
        # 尝试使用系统上的中文字体
        fonts = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'Arial Unicode MS', 'SimSun']
        for font in fonts:
            try:
                font_prop = FontProperties(fname=f"C:\\Windows\\Fonts\\{font}.ttf")
                return font_prop
            except:
                continue
        
        print("警告: 未找到中文字体，可能导致中文显示为方块")
        return None
    except:
        print("警告: 配置中文字体失败")
        return None

def plot_training_curves(file_path, window_size=10):
    """
    绘制训练曲线，使用滑动窗口平滑曲线
    
    参数:
    file_path: 训练数据CSV文件路径
    window_size: 平滑窗口大小
    """
    # 配置中文字体
    chinese_font = setup_chinese_font()
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 '{file_path}' 不存在")
        return
    
    # 读取数据
    try:
        data = pd.read_csv(file_path)
        print(f"成功读取数据: {len(data)} 条记录")
    except Exception as e:
        print(f"读取文件错误: {e}")
        return
    
    # 创建图表
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    if chinese_font:
        fig.suptitle('Flappy Bird DQN 训练曲线', fontsize=16, fontproperties=chinese_font)
    else:
        fig.suptitle('Flappy Bird DQN Training Curves', fontsize=16)
    
    # 平滑函数
    def smooth(y, window_size):
        box = np.ones(window_size) / window_size
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
    
    # 中文标签字典
    labels = {
        'score': '得分',
        'avg_reward': '平均奖励',
        'length': '回合长度',
        'total_reward': '总奖励',
        'episodes': '回合',
        'steps': '步数',
        'reward': '奖励',
        'Q值': 'Q值',
        'epsilon': '探索率',
        'origin': '原始数据',
        'smooth': '平滑',
    }
    
    # 1. 绘制得分曲线
    ax1 = axs[0, 0]
    if 'score' in data.columns:
        scores = data['score'].values
        episodes = data['episode'].values
        ax1.plot(episodes, scores, 'b-', alpha=0.3, label=labels['origin'] if chinese_font else 'Original Data')
        if len(scores) > window_size:
            ax1.plot(episodes, smooth(scores, window_size), 'r-', 
                    label=f"{labels['smooth']} (窗口={window_size})" if chinese_font else f"Smoothed (window={window_size})")
        
        if chinese_font:
            ax1.set_title(labels['score'], fontproperties=chinese_font)
            ax1.set_xlabel(labels['episodes'], fontproperties=chinese_font)
            ax1.set_ylabel(labels['score'], fontproperties=chinese_font)
        else:
            ax1.set_title('Score')
            ax1.set_xlabel('Episodes')
            ax1.set_ylabel('Score')
            
        ax1.legend(prop=chinese_font)
        ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. 绘制平均奖励曲线
    ax2 = axs[0, 1]
    if 'avg_reward' in data.columns:
        rewards = data['avg_reward'].values
        episodes = data['episode'].values
        ax2.plot(episodes, rewards, 'g-', alpha=0.3, label=labels['origin'] if chinese_font else 'Original Data')
        if len(rewards) > window_size:
            ax2.plot(episodes, smooth(rewards, window_size), 'r-', 
                    label=f"{labels['smooth']} (窗口={window_size})" if chinese_font else f"Smoothed (window={window_size})")
        
        if chinese_font:
            ax2.set_title(labels['avg_reward'], fontproperties=chinese_font)
            ax2.set_xlabel(labels['episodes'], fontproperties=chinese_font)
            ax2.set_ylabel(labels['reward'], fontproperties=chinese_font)
        else:
            ax2.set_title('Average Reward')
            ax2.set_xlabel('Episodes')
            ax2.set_ylabel('Reward')
            
        ax2.legend(prop=chinese_font)
        ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3. 绘制回合长度曲线
    ax3 = axs[1, 0]
    if 'length' in data.columns:
        lengths = data['length'].values
        episodes = data['episode'].values
        ax3.plot(episodes, lengths, 'y-', alpha=0.3, label=labels['origin'] if chinese_font else 'Original Data')
        if len(lengths) > window_size:
            ax3.plot(episodes, smooth(lengths, window_size), 'r-', 
                    label=f"{labels['smooth']} (窗口={window_size})" if chinese_font else f"Smoothed (window={window_size})")
        
        if chinese_font:
            ax3.set_title(labels['length'], fontproperties=chinese_font)
            ax3.set_xlabel(labels['episodes'], fontproperties=chinese_font)
            ax3.set_ylabel(labels['steps'], fontproperties=chinese_font)
        else:
            ax3.set_title('Episode Length')
            ax3.set_xlabel('Episodes')
            ax3.set_ylabel('Steps')
            
        ax3.legend(prop=chinese_font)
        ax3.grid(True, linestyle='--', alpha=0.7)
    
    # 4. 绘制总奖励曲线
    ax4 = axs[1, 1]
    if 'total_reward' in data.columns:
        total_rewards = data['total_reward'].values
        episodes = data['episode'].values
        ax4.plot(episodes, total_rewards, 'm-', alpha=0.3, label=labels['origin'] if chinese_font else 'Original Data')
        if len(total_rewards) > window_size:
            ax4.plot(episodes, smooth(total_rewards, window_size), 'r-', 
                    label=f"{labels['smooth']} (窗口={window_size})" if chinese_font else f"Smoothed (window={window_size})")
        
        if chinese_font:
            ax4.set_title(labels['total_reward'], fontproperties=chinese_font)
            ax4.set_xlabel(labels['episodes'], fontproperties=chinese_font)
            ax4.set_ylabel(labels['reward'], fontproperties=chinese_font)
        else:
            ax4.set_title('Total Reward')
            ax4.set_xlabel('Episodes')
            ax4.set_ylabel('Reward')
            
        ax4.legend(prop=chinese_font)
        ax4.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图表 - 使用更高DPI和PNG格式保证中文正常显示
    output_path = os.path.join(os.path.dirname(file_path), 'training_curves_cn.png')
    plt.savefig(output_path, dpi=300, format='png')
    print(f"图表已保存至: {output_path}")
    
    # 显示图表
    plt.show()

def plot_model_learning(log_file, window_size=100):
    """
    绘制模型学习曲线
    
    参数:
    log_file: 训练日志文件路径
    window_size: 平滑窗口大小
    """
    # 配置中文字体
    chinese_font = setup_chinese_font()
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 检查文件是否存在
    if not os.path.exists(log_file):
        print(f"错误: 文件 '{log_file}' 不存在")
        return
    
    # 读取数据
    try:
        data = pd.read_csv(log_file)
        print(f"成功读取训练日志: {len(data)} 条记录")
    except Exception as e:
        print(f"读取文件错误: {e}")
        return
    
    # 创建图表
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    if chinese_font:
        fig.suptitle('DQN 学习曲线', fontsize=16, fontproperties=chinese_font)
    else:
        fig.suptitle('DQN Learning Curves', fontsize=16)
    
    # 平滑函数
    def smooth(y, window_size):
        box = np.ones(window_size) / window_size
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
    
    # 中文标签字典
    labels = {
        'loss': '损失',
        'reward': '奖励',
        'epsilon': '探索率(ε)',
        'avg_q_value': '平均Q值',
        'steps': '步数',
        'origin': '原始数据',
        'smooth': '平滑',
    }
    
    # 1. 绘制损失曲线
    ax1 = axs[0, 0]
    if 'loss' in data.columns:
        losses = data['loss'].values
        steps = data['step'].values
        ax1.plot(steps, losses, 'b-', alpha=0.3, label=labels['origin'] if chinese_font else 'Original Data')
        if len(losses) > window_size:
            ax1.plot(steps, smooth(losses, window_size), 'r-', 
                    label=f"{labels['smooth']} (窗口={window_size})" if chinese_font else f"Smoothed (window={window_size})")
        
        if chinese_font:
            ax1.set_title(labels['loss'], fontproperties=chinese_font)
            ax1.set_xlabel(labels['steps'], fontproperties=chinese_font)
            ax1.set_ylabel(labels['loss'], fontproperties=chinese_font)
        else:
            ax1.set_title('Training Loss')
            ax1.set_xlabel('Steps')
            ax1.set_ylabel('Loss')
            
        ax1.legend(prop=chinese_font)
        ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. 绘制奖励曲线
    ax2 = axs[0, 1]
    if 'reward' in data.columns:
        rewards = data['reward'].values
        steps = data['step'].values
        ax2.plot(steps, rewards, 'g-', alpha=0.3, label=labels['origin'] if chinese_font else 'Original Data')
        if len(rewards) > window_size:
            ax2.plot(steps, smooth(rewards, window_size), 'r-', 
                    label=f"{labels['smooth']} (窗口={window_size})" if chinese_font else f"Smoothed (window={window_size})")
        
        if chinese_font:
            ax2.set_title(labels['reward'], fontproperties=chinese_font)
            ax2.set_xlabel(labels['steps'], fontproperties=chinese_font)
            ax2.set_ylabel(labels['reward'], fontproperties=chinese_font)
        else:
            ax2.set_title('Batch Reward')
            ax2.set_xlabel('Steps')
            ax2.set_ylabel('Reward')
            
        ax2.legend(prop=chinese_font)
        ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3. 绘制探索率曲线
    ax3 = axs[1, 0]
    if 'epsilon' in data.columns:
        epsilons = data['epsilon'].values
        steps = data['step'].values
        
        if chinese_font:
            ax3.plot(steps, epsilons, 'y-', label=labels['epsilon'])
            ax3.set_title(labels['epsilon'], fontproperties=chinese_font)
            ax3.set_xlabel(labels['steps'], fontproperties=chinese_font)
            ax3.set_ylabel('ε', fontproperties=chinese_font)
        else:
            ax3.plot(steps, epsilons, 'y-', label='Epsilon (ε)')
            ax3.set_title('Exploration Rate Change')
            ax3.set_xlabel('Steps')
            ax3.set_ylabel('ε')
            
        ax3.legend(prop=chinese_font)
        ax3.grid(True, linestyle='--', alpha=0.7)
    
    # 4. 绘制平均Q值曲线
    ax4 = axs[1, 1]
    if 'avg_q_value' in data.columns:
        q_values = data['avg_q_value'].values
        steps = data['step'].values
        ax4.plot(steps, q_values, 'm-', alpha=0.3, label=labels['origin'] if chinese_font else 'Original Data')
        if len(q_values) > window_size:
            ax4.plot(steps, smooth(q_values, window_size), 'r-', 
                    label=f"{labels['smooth']} (窗口={window_size})" if chinese_font else f"Smoothed (window={window_size})")
        
        if chinese_font:
            ax4.set_title(labels['avg_q_value'], fontproperties=chinese_font)
            ax4.set_xlabel(labels['steps'], fontproperties=chinese_font)
            ax4.set_ylabel('Q值', fontproperties=chinese_font)
        else:
            ax4.set_title('Average Q Value')
            ax4.set_xlabel('Steps')
            ax4.set_ylabel('Q Value')
            
        ax4.legend(prop=chinese_font)
        ax4.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图表 - 使用更高DPI和PNG格式保证中文正常显示
    output_path = os.path.join(os.path.dirname(log_file), 'learning_curves_cn.png')
    plt.savefig(output_path, dpi=300, format='png')
    print(f"图表已保存至: {output_path}")
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练曲线可视化工具')
    parser.add_argument('--episodes', type=str, default='models/training_curves.csv',
                      help='回合训练数据文件路径')
    parser.add_argument('--logs', type=str, default='logs/training_log.csv',
                      help='步级训练日志文件路径')
    parser.add_argument('--smooth', type=int, default=10,
                      help='平滑窗口大小')
    
    args = parser.parse_args()
    
    print("Flappy Bird DQN 训练可视化工具 (中文版)")
    print("-" * 40)
    
    # 绘制回合级训练曲线
    print("\n绘制回合训练曲线...")
    plot_training_curves(args.episodes, args.smooth)
    
    # 绘制步级学习曲线
    print("\n绘制步级学习曲线...")
    # 查找最新的训练日志文件
    if args.logs == 'logs/training_log.csv' and os.path.exists('logs'):
        log_files = [f for f in os.listdir('logs') if f.startswith('training_log_') and f.endswith('.csv')]
        if log_files:
            # 按文件名排序（包含时间戳）
            log_files.sort(reverse=True)
            latest_log = os.path.join('logs', log_files[0])
            print(f"找到最新日志文件: {latest_log}")
            plot_model_learning(latest_log, args.smooth * 10)  # 步级数据更多，使用更大窗口
        else:
            print("未找到训练日志文件")
    else:
        plot_model_learning(args.logs, args.smooth * 10) 