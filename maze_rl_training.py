"""
迷宫强化学习训练脚本
使用Gymnasium创建自定义迷宫环境，使用Ray RLlib的PPO算法训练智能体找到目标
并记录训练过程中的奖励变化曲线

本实现使用RLlib的默认模型配置（无需自定义TorchModelV2），这是官方推荐的成熟方案。
通过model配置参数（如fcnet_hiddens）来指定网络结构，简化了代码并提高了可维护性。
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

from ray.tune.registry import register_env
import ray
from ray.rllib.algorithms.ppo import PPOConfig


class SimpleMazeEnv(gym.Env):
    """
    简单的网格迷宫环境
    智能体需要从起点(0,0)移动到目标点
    """
    
    # 动作定义: 0=上, 1=下, 2=左, 3=右
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    
    def __init__(self, config=None):
        super(SimpleMazeEnv, self).__init__()
        self.config = config or {}
        
        # 迷宫大小
        self.size = self.config.get("size", 5)
        
        # 定义迷宫：0=可通行, 1=墙壁
        # 创建一个简单的5x5迷宫，有障碍物
        self.maze = np.array([
            [0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0]
        ], dtype=np.int32)
        
        # 起点和终点
        self.start_pos = np.array([0, 0])
        self.goal_pos = np.array([self.size - 1, self.size - 1])
        
        # 当前位置
        self.agent_pos = None
        
        # 观测空间：智能体的位置坐标 (x, y)
        # 也可以选择使用one-hot编码或者周围环境的观察
        self.observation_space = spaces.Box(
            low=0, 
            high=self.size - 1, 
            shape=(2,), 
            dtype=np.float32
        )
        
        # 动作空间：4个离散动作（上下左右）
        self.action_space = spaces.Discrete(4)
        
        # 最大步数限制，防止无限循环
        self.max_steps = self.size * self.size * 2
        self.steps = 0
        
    def reset(self, *, seed=None, options=None):
        """重置环境到初始状态"""
        super().reset(seed=seed)
        
        # 重置智能体位置到起点
        self.agent_pos = self.start_pos.copy()
        self.steps = 0
        
        return self.agent_pos.astype(np.float32), {}
    
    def step(self, action):
        """执行一步动作"""
        self.steps += 1
        
        # 计算新位置
        new_pos = self.agent_pos.copy()
        
        if action == self.ACTION_UP:
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == self.ACTION_DOWN:
            new_pos[0] = min(self.size - 1, new_pos[0] + 1)
        elif action == self.ACTION_LEFT:
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == self.ACTION_RIGHT:
            new_pos[1] = min(self.size - 1, new_pos[1] + 1)
        
        # 检查是否撞墙
        hit_wall = False
        if self.maze[new_pos[0], new_pos[1]] == 1:
            hit_wall = True
            new_pos = self.agent_pos.copy()  # 撞墙则停留在原位置
        
        # 更新位置
        self.agent_pos = new_pos
        
        # 计算奖励
        reward = 0.0
        
        # 到达目标获得大奖励
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward = 10.0
            done = True
            truncated = False
        # 撞墙小惩罚
        elif hit_wall:
            reward = -0.5
            done = False
            truncated = self.steps >= self.max_steps
        # 每步小惩罚，鼓励快速到达目标
        else:
            reward = -0.01
            done = False
            truncated = self.steps >= self.max_steps
        
        # 额外奖励：距离目标越近奖励越高（可选，帮助训练）
        distance_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)
        reward += 0.1 * (self.size * np.sqrt(2) - distance_to_goal) / (self.size * np.sqrt(2))
        
        info = {
            "distance_to_goal": distance_to_goal,
            "steps": self.steps
        }
        
        return self.agent_pos.astype(np.float32), reward, done, truncated, info
    
    def render(self, mode='human'):
        """渲染环境（可选实现）"""
        if mode == 'human':
            grid = np.zeros((self.size, self.size), dtype=str)
            
            # 绘制墙壁
            for i in range(self.size):
                for j in range(self.size):
                    if self.maze[i, j] == 1:
                        grid[i, j] = '#'
                    else:
                        grid[i, j] = '.'
            
            # 绘制目标
            grid[self.goal_pos[0], self.goal_pos[1]] = 'G'
            
            # 绘制智能体
            if self.agent_pos is not None:
                grid[self.agent_pos[0], self.agent_pos[1]] = 'A'
            
            print("\n" + "-" * (self.size * 2 + 1))
            for row in grid:
                print("|" + " ".join(row) + "|")
            print("-" * (self.size * 2 + 1) + "\n")


def maze_env_creator(env_config):
    """环境创建函数"""
    return SimpleMazeEnv(env_config)


# 注册环境
register_env("SimpleMaze-v0", maze_env_creator)


def plot_training_curve(episode_rewards, save_path="training_curve.png"):
    """
    绘制训练曲线
    """
    plt.figure(figsize=(12, 6))
    
    # 绘制原始reward曲线
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.3, color='blue', label='每轮奖励')
    
    # 计算移动平均
    if len(episode_rewards) > 10:
        window_size = min(50, len(episode_rewards) // 10)
        moving_avg = []
        for i in range(len(episode_rewards)):
            start = max(0, i - window_size + 1)
            moving_avg.append(np.mean(episode_rewards[start:i+1]))
        plt.plot(moving_avg, color='red', linewidth=2, label=f'移动平均 (窗口={window_size})')
    
    plt.xlabel('训练轮次 (Episode)')
    plt.ylabel('奖励 (Reward)')
    plt.title('训练过程奖励曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制累积平均奖励
    plt.subplot(1, 2, 2)
    cumulative_avg = []
    for i in range(len(episode_rewards)):
        cumulative_avg.append(np.mean(episode_rewards[:i+1]))
    plt.plot(cumulative_avg, color='green', linewidth=2, label='累积平均奖励')
    plt.xlabel('训练轮次 (Episode)')
    plt.ylabel('累积平均奖励')
    plt.title('累积平均奖励曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存到: {save_path}")
    plt.close()


def main():
    """主训练函数"""
    # 1. 启动Ray
    ray.init(ignore_reinit_error=True)
    
    print("=" * 60)
    print("开始训练迷宫导航智能体")
    print("=" * 60)
    
    # 2. 配置PPO算法 - 使用RLlib默认模型配置
    config = (
        PPOConfig()
        .environment(env="SimpleMaze-v0", env_config={"size": 5})
        .env_runners(num_env_runners=4)  # 并行环境数量
        .framework("torch")
        .training(
            # 使用RLlib默认模型，通过fcnet_hiddens配置隐藏层
            # 这样可以避免自定义TorchModelV2，使用官方推荐的成熟方案
            model={
                "fcnet_hiddens": [128, 128, 64],  # 三层全连接网络：128 -> 128 -> 64
                "fcnet_activation": "relu",  # 使用ReLU激活函数
                "vf_share_layers": False,  # 价值函数和策略网络不共享底层
            },
            train_batch_size=4000,  # 训练批次大小
            lr=3e-4,  # 学习率
            gamma=0.99,  # 折扣因子
            lambda_=0.95,  # GAE lambda参数
            clip_param=0.2,  # PPO clip参数
            entropy_coeff=0.01,  # 熵系数，鼓励探索
        )
    )
    
    # 设置SGD参数（需要在config对象上直接设置，而不是在training()中）
    config.sgd_minibatch_size = 128
    config.num_sgd_iter = 10
    
    # 禁用新API栈以兼容旧版API
    config.api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False
    )
    
    # 3. 创建trainer
    trainer = config.build_algo()
    
    # 4. 训练并记录奖励
    episode_rewards = []
    episode_lengths = []
    training_iterations = 100  # 训练轮次
    
    print(f"\n开始训练，共 {training_iterations} 轮...")
    print("-" * 60)
    
    for i in range(training_iterations):
        result = trainer.train()
        
        # 提取episode奖励信息
        episode_reward_mean = 0.0
        episode_len_mean = 0.0
        
        # 尝试不同的方式获取episode奖励
        if 'env_runners' in result:
            if 'episode_reward_mean' in result['env_runners']:
                episode_reward_mean = result['env_runners']['episode_reward_mean']
            if 'episode_len_mean' in result['env_runners']:
                episode_len_mean = result['env_runners']['episode_len_mean']
        elif 'episode_reward_mean' in result:
            episode_reward_mean = result['episode_reward_mean']
        elif 'episode_reward_max' in result:
            # 如果没有mean，尝试使用max的估计
            episode_reward_mean = result.get('episode_reward_max', 0) * 0.5
        
        episode_rewards.append(episode_reward_mean)
        episode_lengths.append(episode_len_mean)
        
        # 每10轮打印一次详细信息
        if (i + 1) % 10 == 0 or i == 0:
            print(f"轮次 {i+1}/{training_iterations}")
            print(f"  平均奖励: {episode_reward_mean:.2f}")
            print(f"  平均步数: {episode_len_mean:.2f}")
            print(f"  总时间步数: {result.get('timesteps_total', 0)}")
            print(f"  本轮采样步数: {result.get('num_env_steps_sampled_this_iter', 0)}")
            print("-" * 60)
    
    # 5. 保存模型
    checkpoint_path = trainer.save("trained_maze_model")
    print(f"\n模型已保存到: {checkpoint_path}")
    
    # 6. 绘制并保存训练曲线
    print("\n正在生成训练曲线...")
    os.makedirs("results", exist_ok=True)
    plot_training_curve(episode_rewards, save_path="results/training_curve.png")
    
    # 7. 保存训练数据
    np.save("results/episode_rewards.npy", np.array(episode_rewards))
    np.save("results/episode_lengths.npy", np.array(episode_lengths))
    print("训练数据已保存到 results/ 目录")
    
    # 8. 测试训练好的模型
    print("\n测试训练好的模型...")
    env = SimpleMazeEnv(config={"size": 5})
    obs, info = env.reset()
    
    policy = trainer.get_policy()
    total_reward = 0
    max_test_steps = 100
    
    for step in range(max_test_steps):
        action = policy.compute_single_action(obs, explore=False)[0]
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if (step + 1) % 10 == 0:
            print(f"  测试步数 {step+1}: 位置={obs}, 奖励={reward:.2f}, 距离目标={info['distance_to_goal']:.2f}")
        
        if done or truncated:
            break
    
    print(f"\n测试完成! 总奖励: {total_reward:.2f}, 总步数: {step+1}")
    if np.array_equal(obs.astype(int), env.goal_pos):
        print("✓ 成功到达目标!")
    else:
        print(f"✗ 未到达目标，最终位置: {obs.astype(int)}, 目标位置: {env.goal_pos}")
    
    # 9. 清理资源
    ray.shutdown()
    print("\n训练完成!")


if __name__ == "__main__":
    main()

