# flapped-bird

## 项目简介

本项目是基于 Pygame Zero 和 DQN（深度Q网络）的 Flappy Bird 智能体训练与演示。AI 通过深度强化学习自动玩 Flappy Bird，并支持人机切换。

## 主要特性
- Pygame Zero 简化游戏开发
- DQN 智能体自动学习
- 支持训练日志与模型保存
- 可视化训练曲线
- 支持自动与手动两种模式

## 依赖环境
- Python 3.7 及以上
- Pygame Zero
- torch (PyTorch)
- numpy

安装依赖：
```bash
pip install pgzero torch numpy
```

## 运行方法
1. **准备资源**
   - 将所有图片放入 `images/` 文件夹（如 flappybird1.png、flappybird_background.png 等）
   - 将音效放入 `sounds/` 文件夹（如 flap.wav 等）
2. **运行游戏**
   ```bash
   pgzrun flappy_bird_pgzero.py
   ```
3. **训练与测试**
   - 默认自动模式（AI控制），可在代码中切换 `game_mode` 变量
   - 训练日志和模型会自动保存在 `logs/` 和 `models/` 目录

## 主要文件说明
- `flappy_bird_pgzero.py`：主程序，Pygame Zero + DQN
- `dqn_improved.py`：DQN 智能体实现
- `images/`：游戏图片资源
- `sounds/`：游戏音效资源
- `models/`：训练模型与经验池
- `logs/`：训练日志与曲线

## GitHub 仓库
[https://github.com/knot666/flapped-bird](https://github.com/knot666/flapped-bird)

---

如有问题欢迎提 issue 或 PR！
