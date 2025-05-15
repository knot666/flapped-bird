import random
import math
import torch
import pygame
import numpy as np
import os
import time
from dqn_improved import DQNAgent, PrioritizedReplayMemory
import pickle
from collections import deque
import glob
import re
import datetime

# 创建日志目录
os.makedirs("logs", exist_ok=True)

# 创建训练日志文件
def create_training_log():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/training_log_{timestamp}.csv"
    with open(log_file, "w") as f:
        f.write("step,loss,reward,epsilon,avg_q_value\n")
    return log_file

# 初始化pygame
pygame.init()

# 游戏窗口设置
WIDTH = 138 * 4
HEIGHT = 396
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy Bird - 强化学习版")

# 游戏变量
GAP = 150
SPEED = 3
GRAVITY = 0.25 # 减小重力，使飞行更平滑
FLAP_VELOCITY = -4.8  # 减小跳跃速度，使控制更精确
MAX_PIPES_AHEAD = 3
anim_counter = 0
score = 0
score_flag = False
started = True  # 自动启动游戏
auto_mode = True  # 自动设置为AI控制

# 探索率相关常量
OBSERVE = 500  # 进一步减少观察期
EXPLORE = 20000  # 减少探索期
FINAL_EPSILON = 0.1  # 提高最终探索率
INITIAL_EPSILON = 0.5  # 提高初始探索率
total_steps = 0  # 总步数计数器

# 记录训练信息
episode_rewards = []
episode_lengths = []
current_episode_reward = 0
current_episode_length = 0

# 背景和地面初始化
backgrounds = []
for i in range(5):
    backimage = pygame.image.load("images/flappybird_background.png")
    backimage = pygame.transform.scale(backimage, (138, 396))
    backgrounds.append(pygame.Rect(i * 138, 0, 138, 396))

ground = pygame.image.load("images/flappybird_ground.png")
ground = pygame.transform.scale(ground, (WIDTH, 50))
ground_rect = ground.get_rect(topleft=(0, HEIGHT - 50))

# 小鸟初始化
bird_img1 = pygame.image.load("images/flappybird1.png")
bird_img2 = pygame.image.load("images/flappybird2.png")
bird_img3 = pygame.image.load("images/flappybird3.png")
bird_imgs = [bird_img1, bird_img2, bird_img3, bird_img2]
bird = pygame.Rect(WIDTH // 2, HEIGHT // 2, 30, 30)
bird_status = {"dead": False, "vy": 0}

# 管道初始化
pipes = []

# GUI元素
gui_title = pygame.image.load("images/flappybird_title.png")
gui_ready = pygame.image.load("images/flappybird_get_ready.png")
gui_start = pygame.image.load("images/flappybird_start_button.png")
gui_auto = pygame.image.load("images/flappybird_start_button.png")
gui_auto = pygame.transform.scale(
    gui_auto, (int(gui_auto.get_width() * 1.2), int(gui_auto.get_height() * 1.2))
)
gui_over = pygame.image.load("images/flappybird_game_over.png")

# 记忆池函数
def save_memory(memory, filename):
    with open(filename, "wb") as f:
        pickle.dump(memory, f)


def load_memory(filename):
    try:
        with open(filename, "rb") as f:
            obj = pickle.load(f)
            if hasattr(obj, "memory"):  # 检查是否有memory属性
                return obj.memory  # 返回存储的deque数据
            return obj if isinstance(obj, deque) else deque(maxlen=10000)
    except (FileNotFoundError, Exception) as e:
        print(f"加载记忆池失败: {e}，创建新的记忆池")
        return deque(maxlen=10000)


# 初始化DQN代理
# 增强状态空间：
# 1. 小鸟Y坐标
# 2. 小鸟Y速度
# 3-8. 接下来3个管道的相对X距离和中心Y坐标（相对于小鸟）
# 9. 当前得分（归一化）
state_size = 2 + (2 * MAX_PIPES_AHEAD) + 1  # 鸟状态(2) + 每个管道状态(2) * 管道数量 + 得分(1)

action_size = 2
# 模式配置：
# 0 - 手动模式（玩家控制+经验记录）
# 1 - 自动模式（AI自主训练）
game_mode = 1  # 初始为自动模式
print("游戏模式 = ", game_mode)

# 创建记录目录
os.makedirs("models", exist_ok=True)

# 检查GPU可用性
if torch.cuda.is_available():
    print(f"GPU可用: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
else:
    print("GPU不可用，使用CPU训练")

# 加载/初始化代理
load_memory_path = "models/replay_memory.pkl"
load_model_path = "models/dqn_checkpoint.pth" if game_mode == 1 else None

# 加载记忆池
mem = load_memory(load_memory_path)

# 检查现有训练进度
def detect_previous_training():
    try:
        checkpoint_files = glob.glob("models/dqn_checkpoint_*.pth")
        if not checkpoint_files:
            return 0
        
        pattern = re.compile(r'dqn_checkpoint_(\d+)\.pth')
        max_episode = 0
        for file in checkpoint_files:
            match = pattern.search(file)
            if match:
                episode_num = int(match.group(1))
                max_episode = max(max_episode, episode_num)
        
        return max_episode
    except Exception as e:
        print(f"检测训练进度失败: {e}")
        return 0

# 检测并恢复训练进度
prev_episode_count = detect_previous_training()
if prev_episode_count > 0:
    print(f"检测到之前的训练回合数: {prev_episode_count}")
    # 预填充虚拟回合数据，这样新的保存会从正确的回合数继续
    episode_rewards = [0] * prev_episode_count
    episode_lengths = [0] * prev_episode_count

# 初始化改进版代理
agent = DQNAgent(
    state_size,
    action_size,
    load_memory=PrioritizedReplayMemory(capacity=100000, init_memory=mem),
    load_model=load_model_path,
)

# 在加载模型后添加这行代码
if load_model_path:
    # 恢复训练步数
    if hasattr(agent, 'steps'):
        total_steps = agent.steps
    
    # 尝试从模型文件中恢复回合数
    try:
        checkpoint = torch.load(load_model_path, map_location=agent.device)
        if 'episode_count' in checkpoint:
            # 清空当前列表
            episode_rewards = []
            episode_lengths = []
            
            # 填充虚拟数据，保持回合数一致
            for _ in range(checkpoint['episode_count']):
                episode_rewards.append(0)
                episode_lengths.append(0)
            
            print(f"已恢复训练回合计数: {len(episode_rewards)}")
    except Exception as e:
        print(f"恢复回合计数失败: {e}")

# 调整探索率 - 大幅提高初始探索率
epsilon = INITIAL_EPSILON

# 游戏声音
flap_sound = pygame.mixer.Sound("sounds/flap.wav")
collide_sound = pygame.mixer.Sound("sounds/collide.wav")
fall_sound = pygame.mixer.Sound("sounds/fall.wav")
pygame.mixer.music.load("music/flappybird.mp3")


def get_state():
    """增强版游戏状态：包含小鸟信息和多个前方管道信息"""
    # 基础状态：小鸟信息
    state = [bird.y / HEIGHT, bird_status["vy"] / 10]  # 归一化小鸟Y坐标  # 归一化垂直速度

    # 添加多个管道信息
    visible_pipes = []
    for pipe_top, pipe_bottom in pipes:
        # 只考虑在小鸟前方的管道
        if pipe_top.right > bird.x:
            pipe_center = (pipe_top.height + pipe_bottom.top) / 2
            visible_pipes.append(
                {
                    "distance": (pipe_top.centerx - bird.centerx) / WIDTH,  # 归一化水平距离
                    "center": pipe_center / HEIGHT,  # 归一化管道中心Y坐标
                }
            )

    # 确保我们有足够的管道信息（不足则用默认值填充）
    while len(visible_pipes) < MAX_PIPES_AHEAD:
        visible_pipes.append({"distance": 1.0, "center": 0.5})  # 最远距离  # 屏幕中心

    # 只取前MAX_PIPES_AHEAD个管道
    for i in range(min(MAX_PIPES_AHEAD, len(visible_pipes))):
        state.extend([visible_pipes[i]["distance"], visible_pipes[i]["center"]])

    return state


def perform_action(action):
    if action == 1:
        bird_status["vy"] = FLAP_VELOCITY
        flap_sound.play()


# 改进的奖励计算函数
def calculate_reward():
    """改进的奖励函数，更注重保持合适的高度"""
    global current_episode_reward

    # 基础存活奖励
    reward = 0.1

    # 获取最近的管道
    nearest_pipe = None
    min_distance = float('inf')
    for pipe_top, pipe_bottom in pipes:
        if pipe_top.right > bird.x:
            distance = pipe_top.centerx - bird.centerx
            if distance < min_distance:
                min_distance = distance
                nearest_pipe = (pipe_top, pipe_bottom)

    if nearest_pipe:
        pipe_top, pipe_bottom = nearest_pipe
        pipe_center = (pipe_top.height + pipe_bottom.top) / 2
        # 计算小鸟到管道中心的距离
        distance_to_center = abs(bird.centery - pipe_center)
        
        # 根据到管道中心的距离给予奖励
        if distance_to_center < GAP / 4:  # 在管道中心附近
            reward += 0.5
        elif distance_to_center < GAP / 2:  # 在安全范围内
            reward += 0.2
        else:  # 偏离太远
            reward -= 0.1

    # 对飞行高度的惩罚
    if bird.top < 50:  # 如果飞得太高
        reward -= 0.5
    elif bird.bottom > HEIGHT - 100:  # 如果飞得太低
        reward -= 0.5

    # 通过管道奖励
    if score_flag and bird.x > pipes[0][0].right:
        reward += 30.0

    # 死亡惩罚
    if bird_status["dead"]:
        # 根据得分调整死亡惩罚
        return -5.0 + min(20, score * 3)

    current_episode_reward += reward
    return reward


def reset_game():
    global score, score_flag, pipes, anim_counter, current_episode_reward, current_episode_length

    # 记录本轮训练统计
    if auto_mode and current_episode_length > 0:
        episode_rewards.append(current_episode_reward)
        episode_lengths.append(current_episode_length)

        # 记录得分到DQN代理
        agent.record_score(score, current_episode_reward)

        # 打印训练信息
        avg_reward = current_episode_reward / current_episode_length if current_episode_length > 0 else 0
        print(f"回合结束：得分={score}, 步数={current_episode_length}, 总奖励={current_episode_reward:.2f}, 平均奖励={avg_reward:.4f}")

        # 保存模型
        if len(episode_rewards) % 10 == 0:
            episode_count = len(episode_rewards)
            agent.save_model(f"models/dqn_checkpoint_{episode_count}.pth")
            agent.save_model("models/dqn_checkpoint.pth")
            save_memory(agent.memory, "models/replay_memory.pkl")

            # 保存训练曲线数据
            with open("models/training_curves.csv", "a") as f:
                f.write(f"{episode_count},{score},{current_episode_length},{current_episode_reward},{avg_reward}\n")

        # 有进步时保存
        if score > 0 and (not hasattr(reset_game, "high_score") or score > reset_game.high_score):
            reset_game.high_score = score
            agent.save_model(f"models/dqn_best_score_{score}.pth")
            print(f"新高分记录！保存模型 dqn_best_score_{score}.pth")

    # 重置游戏状态
    score = 0
    score_flag = False
    pipes = []
    # 修改初始位置，让小鸟从较低的位置开始
    bird.center = (WIDTH // 2, HEIGHT // 2 + 50)
    bird_status["dead"] = False
    bird_status["vy"] = 0
    anim_counter = 0
    current_episode_reward = 0
    current_episode_length = 0

    # 初始生成管道
    for i in range(MAX_PIPES_AHEAD):
        build_pipes()


# 初始化高分记录
reset_game.high_score = 0


def build_pipes():
    global score_flag
    score_flag = True
    # 改进的管道生成：更合理的随机高度
    pipe_height = random.randint(50, HEIGHT - GAP - 50)
    pipe_top = pygame.Rect(WIDTH + len(pipes) * 200, 0, 50, pipe_height)
    pipe_bottom = pygame.Rect(
        WIDTH + len(pipes) * 200, pipe_height + GAP, 50, HEIGHT - pipe_height - GAP
    )
    pipes.append((pipe_top, pipe_bottom))


def update_pipes():
    global pipes, score_flag

    # 检查并移除已通过的管道
    if pipes and pipes[0][0].right < bird.x:
        # 移除管道前保存其索引，用于判断是否真的移除了管道
        old_pipe_count = len(pipes)

        # 移动现有管道
        for pipe_top, pipe_bottom in pipes:
            pipe_top.x -= SPEED
            pipe_bottom.x -= SPEED

        # 移除已通过的管道
        pipes = [(top, bottom) for top, bottom in pipes if top.right > 0]

        # 只有当真正移除了管道时才重置score_flag
        if len(pipes) < old_pipe_count:
            score_flag = True  # 重置计分标志，使小鸟能够为下一个管道得分
    else:
        # 移动现有管道
        for pipe_top, pipe_bottom in pipes:
            pipe_top.x -= SPEED
            pipe_bottom.x -= SPEED

        # 移除已通过的管道（防御性检查）
        pipes = [(top, bottom) for top, bottom in pipes if top.right > 0]

    # 添加新管道，确保始终有MAX_PIPES_AHEAD个管道在前方
    while len(pipes) < MAX_PIPES_AHEAD:
        last_pipe_x = pipes[-1][0].x if pipes else WIDTH
        pipe_height = random.randint(50, HEIGHT - GAP - 50)
        pipe_top = pygame.Rect(last_pipe_x + 200, 0, 50, pipe_height)
        pipe_bottom = pygame.Rect(
            last_pipe_x + 200, pipe_height + GAP, 50, HEIGHT - pipe_height - GAP
        )
        pipes.append((pipe_top, pipe_bottom))


def fly():
    global score, score_flag
    if pipes and score_flag and bird.x > pipes[0][0].right:
        score += 1
        score_flag = False
    bird_status["vy"] += GRAVITY
    bird.y += bird_status["vy"]
    if bird.top < 0:
        bird.top = 0


def check_collision():
    for pipe_top, pipe_bottom in pipes:
        if bird.colliderect(pipe_top) or bird.colliderect(pipe_bottom):
            collide_sound.play()
            bird_status["dead"] = True
            return
    if bird.colliderect(ground_rect):
        fall_sound.play()
        bird_status["dead"] = True


def update():
    global state, epsilon, started, current_episode_length, total_steps

    if bird_status["dead"]:
        if auto_mode:
            reset_game()
            started = True
        else:
            if pygame.mouse.get_pressed()[0]:
                reset_game()
                started = True
        return

    # 更新AI控制
    state = get_state()

    # 只有在自动模式下才执行AI的决策
    if auto_mode:
        current_episode_length += 1
        total_steps += 1  # 更新总步数
        agent.steps = total_steps  # 同步步数

        # 增强状态，添加得分信息
        augmented_state = state.copy()
        augmented_state.append(score / 10.0)  # 添加归一化得分

        # 使用增强后的状态选择动作
        action = agent.select_action(augmented_state, epsilon)
        perform_action(action)
        # print(1, epsilon)
    # 更新背景和管道
    for i in range(len(backgrounds)):
        backgrounds[i].x -= SPEED
        if backgrounds[i].right <= 0:
            backgrounds[i].left = WIDTH

    update_pipes()
    fly()
    check_collision()

    # 只有在自动模式下才进行学习过程
    if auto_mode:
        reward = calculate_reward()
        next_state = get_state()
        done = bird_status["dead"]

        # 增强状态，添加得分信息
        augmented_state = state.copy()
        augmented_state.append(score / 10.0)  # 归一化得分添加到状态

        augmented_next_state = next_state.copy()
        augmented_next_state.append(score / 10.0)  # 对下一状态也这样处理

        # 存储经验 - 使用增强的状态表示
        agent.memory.push(augmented_state, action, reward, augmented_next_state, done)

        # 只有在观察期之后才训练模型
        if total_steps > OBSERVE:
            # 训练模型
            loss = agent.optimize_model()
            
            # 记录训练日志
            with open(log_file, "a") as f:
                f.write(f"{total_steps},{loss},{reward},{epsilon},{agent.get_avg_q_value()}\n")

            # 每4步再额外训练一次，增加样本利用率
            if current_episode_length % 4 == 0 and len(agent.memory) > agent.batch_size:
                loss = agent.optimize_model()
                # 记录额外训练日志
                with open(log_file, "a") as f:
                    f.write(f"{total_steps},{loss},{reward},{epsilon},{agent.get_avg_q_value()}\n")

        # 降低探索率 - 线性衰减
        if total_steps > OBSERVE and epsilon > FINAL_EPSILON:
            # 线性衰减公式
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
            # 确保epsilon不会低于FINAL_EPSILON
            epsilon = max(FINAL_EPSILON, epsilon)


def draw():
    screen.fill((255, 255, 255))

    # 绘制背景
    for i in range(len(backgrounds)):
        screen.blit(backimage, backgrounds[i])

    # 未开始状态显示
    if not started:
        screen.blit(gui_title, (WIDTH // 2 - gui_title.get_width() // 2, 50))
        screen.blit(gui_ready, (WIDTH // 2 - gui_ready.get_width() // 2, 150))
        if game_mode == 0:  # 手动模式
            screen.blit(gui_start, (WIDTH // 2 - gui_start.get_width() // 2, 345))
        elif game_mode == 1:  # 自动模式
            screen.blit(gui_auto, (WIDTH // 2 - gui_auto.get_width() // 2, 345))
        return

    # 绘制管道
    for pipe_top, pipe_bottom in pipes:
        pygame.draw.rect(screen, (0, 255, 0), pipe_top)
        pygame.draw.rect(screen, (0, 255, 0), pipe_bottom)

    # 绘制地面
    screen.blit(ground, ground_rect)

    # 绘制小鸟
    current_bird_img = bird_imgs[anim_counter // 2]
    screen.blit(current_bird_img, bird)

    # 绘制分数
    font = pygame.font.Font(None, 36)
    text = font.render(str(score), True, (0, 0, 0))
    screen.blit(text, (30, 30))

    # 自动模式时显示额外信息
    if auto_mode:
        # 显示探索率
        eps_text = font.render(f"ε: {epsilon:.4f}", True, (0, 0, 0))
        screen.blit(eps_text, (30, 70))

        # 显示训练步数
        steps_text = font.render(f"Steps: {total_steps}", True, (0, 0, 0))
        screen.blit(steps_text, (30, 110))

        # 显示阶段
        if total_steps <= OBSERVE:
            state_text = font.render("observe", True, (255, 0, 0))
        elif total_steps <= OBSERVE + EXPLORE:
            state_text = font.render("exolore", True, (0, 0, 255))
        else:
            state_text = font.render("train", True, (0, 255, 0))
        screen.blit(state_text, (30, 150))

    # 游戏结束显示
    if bird_status["dead"]:
        screen.blit(
            gui_over,
            (
                WIDTH // 2 - gui_over.get_width() // 2,
                HEIGHT // 2 - gui_over.get_height() // 2,
            ),
        )
        restart_text = font.render("Click to restart", True, (0, 0, 0))
        screen.blit(
            restart_text, (WIDTH // 2 - restart_text.get_width() // 2, HEIGHT // 2 + 50)
        )


def on_mouse_down(pos):
    global started, auto_mode, game_mode

    # 处理死亡状态
    if bird_status["dead"]:
        if not auto_mode:
            reset_game()
            started = True
        return

    # 未开始状态处理
    if not started:
        # 根据游戏模式处理
        if game_mode == 1:  # 自动模式
            started = True
            auto_mode = True
            print("已切换至自动模式 - AI控制")
            reset_game()
            pygame.mixer.music.play(-1)
            # agent.epsilon = 0.1  # 设置较低的探索率
        elif game_mode == 0:  # 手动模式
            started = True
            auto_mode = True  # 自动设置为AI控制
            print("已切换至手动模式 - 玩家控制")
            reset_game()
            pygame.mixer.music.play(-1)

    # 手动模式记录玩家操作
    if not auto_mode and started:
        state = get_state()
        action = 1  # 点击视为跳跃动作
        bird_status["vy"] = FLAP_VELOCITY
        flap_sound.play()

        # 记录玩家操作到记忆池
        reward = calculate_reward()
        next_state = get_state()
        done = bird_status["dead"]
        agent.memory.push(state, action, reward, next_state, done)


# 创建训练曲线记录文件
if not os.path.exists("models/training_curves.csv"):
    with open("models/training_curves.csv", "w") as f:
        f.write("episode,score,length,total_reward,avg_reward\n")

# 游戏主循环
running = True
clock = pygame.time.Clock()

# 在初始化部分添加
log_file = create_training_log()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            # 保存最终模型和记忆池
            # 获取当前回合数
            current_episode_count = len(episode_rewards)
            if prev_episode_count > current_episode_count:
                current_episode_count = prev_episode_count
            agent.save_model(f"models/dqn_checkpoint_{current_episode_count}_final.pth")
            agent.save_model("models/dqn_checkpoint_final.pth")
            save_memory(agent.memory, "models/replay_memory_final.pkl")
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            on_mouse_down(pygame.mouse.get_pos())
        elif event.type == pygame.KEYDOWN:
            if (
                event.key == pygame.K_SPACE
                and not auto_mode
                and started
                and not bird_status["dead"]
            ):
                # 空格键跳跃（手动模式）
                state = get_state()
                action = 1
                bird_status["vy"] = FLAP_VELOCITY
                flap_sound.play()

                # 记录操作
                reward = calculate_reward()
                next_state = get_state()
                done = bird_status["dead"]
                agent.memory.push(state, action, reward, next_state, done)

    update()
    draw()
    pygame.display.flip()

    # 控制帧率
    clock.tick(60)

# 退出前保存
print("游戏结束，正在保存模型和记忆池...")
# 获取当前回合数
current_episode_count = len(episode_rewards)
if prev_episode_count > current_episode_count:
    current_episode_count = prev_episode_count
agent.save_model(f"models/dqn_checkpoint_{current_episode_count}_final.pth")
agent.save_model("models/dqn_checkpoint_final.pth")
save_memory(agent.memory, "models/replay_memory_final.pkl")
pygame.quit()
