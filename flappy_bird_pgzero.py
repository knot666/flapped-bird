# -*- coding: utf-8 -*-
import random
import math
import torch
import numpy as np
import os
import time
from dqn_improved import DQNAgent, PrioritizedReplayMemory
import pickle
from collections import deque
import glob
import re
import datetime

# 游戏常量
WIDTH = 138 * 4
HEIGHT = 396
GAP = 150
SPEED = 3
GRAVITY = 0.25
FLAP_VELOCITY = -4.8
MAX_PIPES_AHEAD = 3

# 游戏变量
bird = Actor('flappybird1', center=(WIDTH//2, HEIGHT//2))
bird_status = {"dead": False, "vy": 0}
pipes = []
score = 0
score_flag = False
started = True
auto_mode = True
anim_counter = 0

# 添加背景滚动变量
background_x = 0
background_width = 138  # 背景图片的宽度

# 探索率相关常量
OBSERVE = 500
EXPLORE = 20000
FINAL_EPSILON = 0.1
INITIAL_EPSILON = 0.5
total_steps = 0

# 记录训练信息
episode_rewards = []
episode_lengths = []
current_episode_reward = 0
current_episode_length = 0

# 创建日志目录
os.makedirs("logs", exist_ok=True)

# 创建训练日志文件
def create_training_log():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/training_log_{timestamp}.csv"
    with open(log_file, "w") as f:
        f.write("step,loss,reward,epsilon,avg_q_value\n")
    return log_file

# 初始化DQN代理
state_size = 2 + (2 * MAX_PIPES_AHEAD) + 1
action_size = 2
game_mode = 1

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
def load_memory(filename):
    try:
        with open(filename, "rb") as f:
            obj = pickle.load(f)
            if hasattr(obj, "memory"):
                return obj.memory
            return obj if isinstance(obj, deque) else deque(maxlen=10000)
    except (FileNotFoundError, Exception) as e:
        print(f"加载记忆池失败: {e}，创建新的记忆池")
        return deque(maxlen=10000)

# 添加保存记忆池函数
def save_memory(memory, filename):
    try:
        with open(filename, "wb") as f:
            pickle.dump(memory, f)
    except Exception as e:
        print(f"保存记忆池失败: {e}")

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
    episode_rewards = [0] * prev_episode_count
    episode_lengths = [0] * prev_episode_count

# 初始化改进版代理
agent = DQNAgent(
    state_size,
    action_size,
    load_memory=PrioritizedReplayMemory(capacity=100000, init_memory=mem),
    load_model=load_model_path,
)

if load_model_path:
    if hasattr(agent, 'steps'):
        total_steps = agent.steps

    try:
        checkpoint = torch.load(load_model_path, map_location=agent.device)
        if 'episode_count' in checkpoint:
            episode_rewards = []
            episode_lengths = []

            for _ in range(checkpoint['episode_count']):
                episode_rewards.append(0)
                episode_lengths.append(0)

            print(f"已恢复训练回合计数: {len(episode_rewards)}")
    except Exception as e:
        print(f"恢复回合计数失败: {e}")

# 调整探索率
epsilon = INITIAL_EPSILON

def get_state():
    """增强版游戏状态：包含小鸟信息和多个前方管道信息"""
    state = [bird.y / HEIGHT, bird_status["vy"] / 10]

    visible_pipes = []
    for pipe_top, pipe_bottom in pipes:
        if pipe_top.right > bird.x:
            pipe_center = (pipe_top.height + pipe_bottom.top) / 2
            visible_pipes.append(
                {
                    "distance": (pipe_top.centerx - bird.centerx) / WIDTH,
                    "center": pipe_center / HEIGHT,
                }
            )

    while len(visible_pipes) < MAX_PIPES_AHEAD:
        visible_pipes.append({"distance": 1.0, "center": 0.5})

    for i in range(min(MAX_PIPES_AHEAD, len(visible_pipes))):
        state.extend([visible_pipes[i]["distance"], visible_pipes[i]["center"]])

    return state

def perform_action(action):
    if action == 1:
        bird_status["vy"] = FLAP_VELOCITY
        sounds.flap.play()

def calculate_reward():
    """改进的奖励函数，更注重保持合适的高度"""
    global current_episode_reward

    reward = 0.1

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
        distance_to_center = abs(bird.centery - pipe_center)

        if distance_to_center < GAP / 4:
            reward += 0.5
        elif distance_to_center < GAP / 2:
            reward += 0.2
        else:
            reward -= 0.1

    if bird.top < 50:
        reward -= 0.5
    elif bird.bottom > HEIGHT - 100:
        reward -= 0.5

    if score_flag and bird.x > pipes[0][0].right:
        reward += 30.0

    if bird_status["dead"]:
        return -5.0 + min(20, score * 3)

    current_episode_reward += reward
    return reward

def reset_game():
    global score, score_flag, pipes, anim_counter, current_episode_reward, current_episode_length

    if auto_mode and current_episode_length > 0:
        episode_rewards.append(current_episode_reward)
        episode_lengths.append(current_episode_length)

        agent.record_score(score, current_episode_reward)

        avg_reward = current_episode_reward / current_episode_length if current_episode_length > 0 else 0
        print(f"回合结束：得分={score}, 步数={current_episode_length}, 总奖励={current_episode_reward:.2f}, 平均奖励={avg_reward:.4f}")

        if len(episode_rewards) % 10 == 0:
            episode_count = len(episode_rewards)
            agent.save_model(f"models/dqn_checkpoint_{episode_count}.pth")
            agent.save_model("models/dqn_checkpoint.pth")
            save_memory(agent.memory, "models/replay_memory.pkl")

            with open("models/training_curves.csv", "a") as f:
                f.write(f"{episode_count},{score},{current_episode_length},{current_episode_reward},{avg_reward}\n")

        if score > 0 and (not hasattr(reset_game, "high_score") or score > reset_game.high_score):
            reset_game.high_score = score
            agent.save_model(f"models/dqn_best_score_{score}.pth")
            print(f"新高分记录！保存模型 dqn_best_score_{score}.pth")

    score = 0
    score_flag = False
    pipes = []
    bird.center = (WIDTH // 2, HEIGHT // 2 + 50)
    bird_status["dead"] = False
    bird_status["vy"] = 0
    anim_counter = 0
    current_episode_reward = 0
    current_episode_length = 0

    for i in range(MAX_PIPES_AHEAD):
        build_pipes()

reset_game.high_score = 0

def build_pipes():
    global score_flag
    score_flag = True
    pipe_height = random.randint(50, HEIGHT - GAP - 50)
    pipe_top = Rect(WIDTH + len(pipes) * 200, 0, 50, pipe_height)
    pipe_bottom = Rect(
        WIDTH + len(pipes) * 200, pipe_height + GAP, 50, HEIGHT - pipe_height - GAP
    )
    pipes.append((pipe_top, pipe_bottom))

def update_pipes():
    global pipes, score_flag

    if pipes and pipes[0][0].right < bird.x:
        old_pipe_count = len(pipes)

        for pipe_top, pipe_bottom in pipes:
            pipe_top.x -= SPEED
            pipe_bottom.x -= SPEED

        pipes = [(top, bottom) for top, bottom in pipes if top.right > 0]

        if len(pipes) < old_pipe_count:
            score_flag = True
    else:
        for pipe_top, pipe_bottom in pipes:
            pipe_top.x -= SPEED
            pipe_bottom.x -= SPEED

        pipes = [(top, bottom) for top, bottom in pipes if top.right > 0]

    while len(pipes) < MAX_PIPES_AHEAD:
        last_pipe_x = pipes[-1][0].x if pipes else WIDTH
        pipe_height = random.randint(50, HEIGHT - GAP - 50)
        pipe_top = Rect(last_pipe_x + 200, 0, 50, pipe_height)
        pipe_bottom = Rect(
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

def animation():
    global anim_counter
    anim_counter += 1
    if anim_counter == 2:
        bird.image = "flappybird1"
    elif anim_counter == 4:
        bird.image = "flappybird2"
    elif anim_counter == 6:
        bird.image = "flappybird3"
    elif anim_counter == 8:
        bird.image = "flappybird2"
        anim_counter = 0

def check_collision():
    for pipe_top, pipe_bottom in pipes:
        if bird.colliderect(pipe_top) or bird.colliderect(pipe_bottom):
            sounds.collide.play()
            bird_status["dead"] = True
            return
    if bird.colliderect(Rect(0, HEIGHT - 50, WIDTH, 50)):
        sounds.fall.play()
        bird_status["dead"] = True

def update():
    global state, epsilon, started, current_episode_length, total_steps, anim_counter, background_x

    if bird_status["dead"]:
        if auto_mode:
            reset_game()
            started = True
        return

    # 更新背景位置
    background_x -= SPEED
    if background_x <= -background_width:
        background_x = 0

    state = get_state()

    if auto_mode:
        current_episode_length += 1
        total_steps += 1
        agent.steps = total_steps

        augmented_state = state.copy()
        augmented_state.append(score / 10.0)

        action = agent.select_action(augmented_state, epsilon)
        perform_action(action)

    update_pipes()
    fly()
    check_collision()

    if auto_mode:
        reward = calculate_reward()
        next_state = get_state()
        done = bird_status["dead"]

        augmented_state = state.copy()
        augmented_state.append(score / 10.0)

        augmented_next_state = next_state.copy()
        augmented_next_state.append(score / 10.0)

        agent.memory.push(augmented_state, action, reward, augmented_next_state, done)

        if total_steps > OBSERVE:
            loss = agent.optimize_model()

            with open(log_file, "a") as f:
                f.write(f"{total_steps},{loss},{reward},{epsilon},{agent.get_avg_q_value()}\n")

            if current_episode_length % 4 == 0 and len(agent.memory) > agent.batch_size:
                loss = agent.optimize_model()
                with open(log_file, "a") as f:
                    f.write(f"{total_steps},{loss},{reward},{epsilon},{agent.get_avg_q_value()}\n")

        if total_steps > OBSERVE and epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
            epsilon = max(FINAL_EPSILON, epsilon)

    animation()

def draw():
    # 绘制滚动背景
    for i in range(5):  # 绘制多个背景图片以确保覆盖整个屏幕
        screen.blit('flappybird_background', (background_x + i * background_width, 0))

    if not started:
        screen.blit('flappybird_title', (WIDTH // 2 - 100, 50))
        screen.blit('flappybird_get_ready', (WIDTH // 2 - 100, 150))
        if game_mode == 0:
            screen.blit('flappybird_start_button', (WIDTH // 2 - 100, 345))
        elif game_mode == 1:
            screen.blit('flappybird_start_button', (WIDTH // 2 - 100, 345))
        return

    for pipe_top, pipe_bottom in pipes:
        screen.draw.filled_rect(pipe_top, (0, 255, 0))
        screen.draw.filled_rect(pipe_bottom, (0, 255, 0))

    screen.blit('flappybird_ground', (0, HEIGHT - 50))
    bird.draw()

    screen.draw.text(str(score), (30, 30), color="black", fontsize=36)

    if auto_mode:
        screen.draw.text(f"ε: {epsilon:.4f}", (30, 70), color="black", fontsize=36)
        screen.draw.text(f"Steps: {total_steps}", (30, 110), color="black", fontsize=36)

        if total_steps <= OBSERVE:
            screen.draw.text("observe", (30, 150), color="red", fontsize=36)
        elif total_steps <= OBSERVE + EXPLORE:
            screen.draw.text("explore", (30, 150), color="blue", fontsize=36)
        else:
            screen.draw.text("train", (30, 150), color="green", fontsize=36)

    if bird_status["dead"]:
        screen.blit('flappybird_game_over', (WIDTH // 2 - 100, HEIGHT // 2 - 50))
        screen.draw.text("Click to restart", (WIDTH // 2 - 100, HEIGHT // 2 + 50), color="black", fontsize=36)

def on_mouse_down(pos):
    global started, auto_mode, game_mode

    if bird_status["dead"]:
        if not auto_mode:
            reset_game()
            started = True
        return

    if not started:
        if game_mode == 1:
            started = True
            auto_mode = True
            print("已切换至自动模式 - AI控制")
            reset_game()
            music.play('flappybird')
        elif game_mode == 0:
            started = True
            auto_mode = True
            print("已切换至手动模式 - 玩家控制")
            reset_game()
            music.play('flappybird')

    if not auto_mode and started:
        state = get_state()
        action = 1
        bird_status["vy"] = FLAP_VELOCITY
        sounds.flap.play()

        reward = calculate_reward()
        next_state = get_state()
        done = bird_status["dead"]
        agent.memory.push(state, action, reward, next_state, done)

def on_key_down(key):
    if key == keys.SPACE and not auto_mode and started and not bird_status["dead"]:
        state = get_state()
        action = 1
        bird_status["vy"] = FLAP_VELOCITY
        sounds.flap.play()

        reward = calculate_reward()
        next_state = get_state()
        done = bird_status["dead"]
        agent.memory.push(state, action, reward, next_state, done)

# 创建训练曲线记录文件
if not os.path.exists("models/training_curves.csv"):
    with open("models/training_curves.csv", "w") as f:
        f.write("episode,score,length,total_reward,avg_reward\n")

# 创建训练日志
log_file = create_training_log()
