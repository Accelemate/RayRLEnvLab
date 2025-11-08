import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override

# 修改导入语句
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

import ray
# 更新为新版导入方式
from ray.rllib.algorithms.ppo import PPOConfig
import os

"""
一个完整的、可运行的强化学习示例，包含了从环境定义到训练完成的全过程，用于演示或测试RLlib的基本功能。

1. 初始化Ray框架：
   - 启动Ray分布式计算框架

2. 创建自定义环境：
   - 注册一个名为"MyEnv-v0"的自定义环境
   - 该环境是一个简单的离散动作空间环境，观测空间是4维向量，动作空间有2个离散动作
    // 观测空间(observation space)：这是环境向智能体(agent)展示的状态信息。
    在这个例子中，观测空间是4维向量，意味着智能体在每个时间步都会接收到一个包含4个数值的向量作为当前环境状态的描述。

    // 动作空间(action space)：这是智能体可以执行的动作集合。"离散动作空间"意味着动作是有限且可数的。
    "2个离散动作"表示智能体在每个时间步只能选择两个动作中的一个，比如动作0和动作1。

    // 离散动作空间环境：指动作空间是离散的强化学习环境，与之相对的是连续动作空间环境（如机器人控制中需要输出连续的力或扭矩）。
    // Gymnasium（环境框架）：
        - 负责定义任务环境
        - 提供环境接口（reset(), step()）
        - 定义观测空间和动作空间

3. 定义自定义神经网络模型：
   - 注册一个名为"my_dnn_model"的自定义深度神经网络模型
   - 该模型包含两个隐藏层，分别用于策略输出和价值函数估计
    // 自定义的深度神经网络模型在这里是策略网络(Policy Network)和价值网络(Value Network)的组合：
      - 策略网络：用于决定在给定状态下应该采取什么动作
      - 价值网络：用于评估当前状态的价值，即从当前状态开始未来能获得多少奖励
    它不是预训练模型，也不是奖励模型。在标准的强化学习中（不包括RLHF），奖励是由环境直接提供的，不需要单独的奖励模型。环境的step方法直接返回奖励值：
    def step(self, action):
        # 简单示例逻辑：action=1时+1奖励，否则0
        reward = 1.0 if action == 1 else 0.0
        # ...
        return self.state, reward, done, info

    这个模型从随机初始化的参数开始，通过与环境交互、收集经验，并使用PPO算法不断更新参数，最终学会一个较好的策略，是一个需要从零开始训练的策略模型。


4. 配置和训练PPO算法：
   - 使用PPO算法进行强化学习训练
   - 设置训练参数：2个工作进程、PyTorch框架、自定义模型等
   - 进行10轮训练迭代

5. 输出训练结果：
   - 每轮训练后打印该轮的平均奖励

6. 清理资源：
   - 关闭Ray框架

整个过程是标准的强化学习训练流程：环境定义→模型构建→算法配置→训练执行→结果输出。这是一个完整的端到端强化学习实验流程。

端到端：指从输入到输出的完整流程，不需要额外的人工干预或外部组件。在这个例子中，端到端意味着：
    - 输入：随机初始化的环境和神经网络
    - 输出：训练完成的模型和训练过程中的奖励统计
整个流程自动完成，从环境初始化、模型构建、算法配置、训练执行到结果输出，形成了一个完整的闭环。
"""


class MyCustomEnv(gym.Env):
    def __init__(self, config=None):
        super(MyCustomEnv, self).__init__()
        self.config = config or {}

        # 假设观测空间是一个长度为4的向量
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        # 假设动作空间离散，有2个动作可选
        self.action_space = spaces.Discrete(2)

        self.state = None

    def reset(self, *, seed=None, options=None):
        # 重置环境状态并返回初始观测
        self.state = np.zeros(4, dtype=np.float32)
        return self.state, {}

    def step(self, action):
        # 简单示例逻辑：action=1时+1奖励，否则0
        reward = 1.0 if action == 1 else 0.0
        # 用随机噪声更新一下 state，让它看起来不像纯0
        self.state = self.state + np.random.randn(4) * 0.01

        # 让episode在小概率下结束
        done = np.random.rand() < 0.05
        truncated = False
        info = {}

        return self.state, reward, done, truncated, info


class MyCustomDNNModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # 假设观测是shape=(4,)的向量
        input_size = obs_space.shape[0]
        hidden_size = 64  # 可以根据需求调整

        # 构建一个简单两层全连接网络
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # 策略输出层和价值函数输出层
        self.policy_layer = nn.Linear(hidden_size, num_outputs)
        self.value_layer = nn.Linear(hidden_size, 1)

        self._value_out = None  # 用于存放价值函数输出

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        # 处理观测
        obs = input_dict["obs"].float()
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))

        # 输出策略 (logits) 与价值 (self._value_out)
        logits = self.policy_layer(x)
        self._value_out = self.value_layer(x).squeeze(1)  # shape: [B]

        return logits, state

    @override(ModelV2)
    def value_function(self):
        # RLlib在计算价值函数损失时会自动调用这个方法
        return self._value_out


def my_env_creator(env_config):
    return MyCustomEnv(env_config)


# 更新注册方式
register_env("MyEnv-v0", my_env_creator)
ModelCatalog.register_custom_model("my_dnn_model", MyCustomDNNModel)


if __name__ == "__main__":
    # 1. 启动Ray分布式框架
    ray.init()

    # 2. 配置训练参数 - 使用新版配置方式
    config = (
        PPOConfig()
        .environment(env="MyEnv-v0")  # 使用刚才注册的自定义环境
        .env_runners(num_env_runners=2)  # 并行worker数量；可根据CPU核心数灵活调整
        .framework("torch")  # 使用PyTorch
        .training(
            model={
                "custom_model": "my_dnn_model",  # 自定义的网络结构
            },
            train_batch_size=4000,
            lr=1e-3,
        )
    )
    
    # 设置SGD参数
    config.sgd_minibatch_size = 128
    
    # 禁用新API栈以兼容自定义ModelV2
    config.api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False
    )

    # 3. 创建PPO的Trainer实例 - 使用新版创建方式
    trainer = config.build_algo()

    # 4. 多轮训练
    for i in range(10):
        result = trainer.train()
        # 显示训练进度和相关信息
        print(f"轮次 {i+1}/10")
        print(f"  训练迭代次数: {result['training_iteration']}")
        print(f"  总时间步数: {result['timesteps_total']}")
        print(f"  本轮采样时间步数: {result['num_env_steps_sampled_this_iter']}")
        print(f"  本轮训练时间步数: {result['num_env_steps_trained_this_iter']}")
        
        # 检查是否有episode相关的奖励信息
        if 'env_runners' in result and 'episode_reward_mean' in result['env_runners']:
            print(f"  平均每轮奖励: {result['env_runners']['episode_reward_mean']}")
        print("-" * 40)

    # 5. 保存训练好的模型
    checkpoint_path = trainer.save("trained_model")
    print(f"模型已保存到: {checkpoint_path}")
    
    # 6. 清理资源
    ray.shutdown()