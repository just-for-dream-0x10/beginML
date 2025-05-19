# PyTorch 强化学习 (Reinforcement Learning) 深度剖析笔记
## 1. 核心概念

- 智能体 (Agent)：学习者或决策者，它与环境进行交互。
- 环境 (Environment)：智能体所处的外部世界，它响应智能体的动作并提供新的状态和奖励。
- 状态 (State, S)：对环境特定时刻的描述。
- 动作 (Action, A)：智能体在特定状态下可以执行的操作。
- 奖励 (Reward, R)：环境在智能体执行一个动作后反馈给智能体的标量信号，用于评价该动作的好坏。
- 策略 (Policy, π)：智能体的行为函数，即在给定状态下选择动作的规则或概率分布。$π(a∣s)=P(A_t=a∣S_t=s)$
- 价值函数 (Value Function)：评估一个状态或状态-动作对的长期价值。
- 状态价值函数 V<sup>π</sup>(s)：从状态 s 开始，遵循策略 π 能获得的期望累积奖励。
- 动作价值函数 Q<sup>π</sup>(s, a)：在状态 s 执行动作 a 后，继续遵循策略 π 能获得的期望累积奖励。
- 模型 (Model)（可选）：对环境行为的模拟，可以预测下一个状态和奖励，即 $P(s',r∣s,a)$

目标：学习一个策略 $π^*$，使得智能体获得的累积折扣奖励（Discounted Cumulative Reward）最大化：
$G_t=R_{t+1}+γR_{t+2}+γ^2R_{t+3}+⋯=∑_{k=0}^∞γ^kR_{t+k+1}$
其中 $γ$ 是折扣因子，表示未来奖励相对于当前奖励的重要性。

## 2.马尔可夫决策模型 (Markov Decision Process, MDP)

RL 问题通常被形式化为马尔可夫决策过程 (MDP)。一个 MDP 由一个五元组定义：
$(S,A,P,R,γ)$
- S：状态集合 (State Space)
- A：动作集合 (Action Space)
- P：状态转移概率 (Transition Probability) $P(s'∣s,a)=P(S_{t+1}'∣S_{t}=s,A_{t}=a)$
- R：奖励函数 (Reward Function) $R(s,a)$  $R(s,a,s')=E[R_{t+1∣St=s,At=a,S_{t+1}=s'}]$
- γ：折扣因子 (Discount Factor) $γ∈[0,1)$

## 3. 关键方程：贝尔曼方程 (Bellman Equations)

贝尔曼方程是 RL 中大多数算法的基础，它们建立了当前状态（或状态-动作对）的价值与其后继状态价值之间的关系。

贝尔曼期望方程 (Bellman Expectation Equation)：描述了在策略 π 下的价值函数。
- 对于 $V^π(s)$，有 $$V^π(s)=E[R_{t+1}+γV^π(S_{t+1})∣St=s] \\ V^π(s)=\sum_{a\in A}\pi(a∣s)\sum_{s'\in S}P(s'∣s,a)[R(s,a,s')+γV^π(s')]$$

- 对于 $Q^π(s,a)$，有 $$Q^π(s,a)=E[R_{t+1}+γQ^π(S_{t+1},A_{t+1})∣St=s,At=a] \\ Q^π(s,a)=\sum_{s'\in S}\sum_{a'\in A}P(s'∣s,a)[R(s,a,s')+γQ^π(s',a')]$$

贝尔曼最优方程 (Bellman Optimality Equation)：描述了最优价值函数。最优策略 $π^*$ 对应的价值函数 $V^*(s)$ 和 $Q^*(s,a)$ 是唯一的且满足以下方程：
- 对于 $V^*(s)$，有 $$V^*(s) = \max_a \mathbb{E} \left[ R_{t+1} + \gamma V^*(S_{t+1}) \mid S_t = s, A_t = a\right] \\ V^*(s) = \max_a \sum_{s' \in S} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^*(s') \right]$$

- 对于 $Q^*(s,a)$，有
$$
Q^*(s, a) = \mathbb{E} \left[ R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1}, a') \mid S_t = s, A_t = a \right]     \\ 
Q^*(s, a) = \sum_{s' \in S} P(s' \mid s, a) \left[ R(s, a, s') + \gamma \max_{a'} Q^*(s', a') \right]
$$

一旦得到最优动作价值函数 $Q^*(s,a)$，最优策略可以通过贪婪地选择使得 $Q^*(s,a)$ 最大的动作来获得：$pi^*(s) = argmax_a Q^*(s,a)$


## 4. PyTorch 在强化学习中的角色
在现代强化学习中，状态空间和动作空间往往非常大甚至是连续的，传统的表格方法不再适用。深度学习，特别是神经网络，被用来近似这些复杂的函数：
- 价值函数近似 (Value Function Approximation, VFA)：使用神经网络近似 $V^π(s)$ 和 $Q^π(s,a)$。(或最优价值函数）

- 策略近似 (Policy Approximation)：使用神经网络来近似策略函数 $\pi_\theta(a∣s)$，即在给定状态下选择动作的概率分布。

PyTorch 的核心作用：
1. 神经网络构建 (torch.nn)：使用 nn.Module 定义策略网络或价值网络。这些网络可以是全连接层 (nn.Linear)、卷积层 (nn.Conv2d，常用于处理图像状态) 等的组合。
2. 参数优化 (torch.optim)：通过定义损失函数（通常源自贝尔曼方程或策略梯度定理），使用 PyTorch 的优化器（如 Adam, SGD）来更新网络参数$θ$。
3. 自动微分 (torch.autograd)：这是 PyTorch 的核心。在计算损失函数后，调用 .backward() 方法会自动计算损失相对于所有网络参数的梯度，然后优化器使用这些梯度进行参数更新。
4. 张量运算 (torch.Tensor)：RL 算法涉及大量的数值计算，PyTorch 提供了高效的张量运算库，并支持 GPU 加速。

## 5. 主流强化学习算法与 PyTorch 实现思路

### 1.  基于价值的算法 (Value-Based Methods)
核心是学习一个价值函数，然后根据价值函数隐式地推导出策略。

#### a. 深度Q网络 (Deep Q-Network, DQN)
- 理念：使用神经网络 $Q_θ(s,a)$ 来近似最优动作价值函数 $Q^*(s,a)$。

- 算法流程：
    1. 初始化神经网络 $Q_θ(s,a)$ 和目标网络 $Q_θ^*(s,a)$
    2. 从环境中采样经验 $(s,a,r,s')$
    3. 计算目标值：$y = r + γmax_a Q_θ^*(s',a)$
    4. 更新网络参数：$θ = argmin_θ E[(y - Q_θ(s,a))^2]$
    5. 重复步骤 2-4，直到收敛

- 数学核心- 损失函数： DQN 的目标是最小化贝尔曼误差（Bellman error）。对于一个经验样本 $(s,a,r,s')$，损失函数通常是均方误差 (MSE)：
$$
L(\theta) = \mathbb{E}_{(s, a, r, s') \sim D} \left[ \left( r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a) \right)^2 \right]
$$
- D 是经验回放池 (Experience Replay Buffer)。 
- $Q_{θ-}$ 是目标网络 (Target Network)，其参数 ${θ-}$ 定期从主网络 $Q_θ$复制而来，以增加训练稳定性。
- $y=r+γmax_a' Q^{θ-}(s',a')$

PyTorch 底层视角与实现：
- Q-Network (nn.Module): 定义一个神经网络。输入是状态 s，输出是每个可能动作a的 Q 值。
    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    
    class QNetwork(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(QNetwork, self).__init__()
            self.fc1 = nn.Linear(state_dim, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, action_dim)
    
        def forward(self, state):
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            return self.fc3(x) # Output Q-values for each action
    ```

- 经验回放 (collections.deque): 存储 $(s,a,r,s',done)$ 元组。
- 目标网络: 通过 target_net.load_state_dict(policy_net.state_dict()) 来同步参数。
- 损失计算:
    ```python
    # state_batch, action_batch, reward_batch, next_state_batch, done_batch
    # action_batch needs to be shaped correctly to gather Q-values
    # e.g., action_batch = action_batch.unsqueeze(1) for discrete actions
    
    q_values = policy_net(state_batch).gather(1, action_batch) # Q(s,a)
    
    with torch.no_grad(): # Target network calculations don't need gradients
        next_q_values_target = target_net(next_state_batch).max(1)[0].detach()
        # max(1)[0] gives max values, .detach() to prevent gradient flow
        td_target = reward_batch + gamma * next_q_values_target * (1 - done_batch)
        # (1 - done_batch) handles terminal states where future reward is 0
    
    loss = F.mse_loss(q_values, td_target.unsqueeze(1))
    # or loss = F.smooth_l1_loss(q_values, td_target.unsqueeze(1)) (Huber loss)
    ```

- 优化：
    ```python
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    loss.backward() # Autograd computes gradients
    # Optional: torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm)
    optimizer.step() # Updates network weights
    ```

### 2. 基于策略的算法 (Policy-Based Methods)
这类算法直接学习策略 $π_θ(a∣s)$

#### a. REINFORCE (Monte Carlo Policy Gradient)
- 理念：通过调整策略参数 θ 来增加能够获得高回报的动作的概率。
- 数学核心 - 策略梯度定理 (Policy Gradient Theorem)：目标函数是期望累积回报 $ J(θ)=E_{τ∼πθ}[G_t]$ 其梯度为：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(A_t \mid S_t) G_t \right]
$$
其中 $G_t = \sum_{k=t}^{T-1} \gamma^{k-t} R_{k+1}$  是从时间步t开始的累积折扣回报。


- PyTorch 底层视角与实现：
- Policy Network (nn.Module): 定义一个神经网络。输入是状态 s，输出是动作的概率分布（例如，对离散动作使用 softmax，对连续动作输出高斯分布的均值和标准差）。
    ```python
    class PolicyNetwork(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(PolicyNetwork, self).__init__()
            self.fc1 = nn.Linear(state_dim, 128)
            self.fc2 = nn.Linear(128, action_dim) # Outputs logits for actions
    
        def forward(self, state):
            x = F.relu(self.fc1(state))
            # For discrete actions, output logits, then apply softmax externally
            # or use torch.distributions.Categorical
            return self.fc2(x)
    ```
- 动作选择与对数概率:
    ```python
    from torch.distributions import Categorical
    
    # Inside training loop, for each step in an episode:
    # state is the current state tensor
    action_logits = policy_net(state)
    action_dist = Categorical(logits=action_logits) # Creates a distribution
    action = action_dist.sample() # Sample an action
    log_prob = action_dist.log_prob(action) # Calculate log P(a|s)
    # Store log_prob and reward for the episode
    ```

- 损失计算 (通常是梯度的负数，因为优化器是最小化损失)：
    ```python
    # After an episode finishes, calculate returns G_t for each step
    # returns is a list/tensor of G_t values for the episode
    # saved_log_probs is a list/tensor of log_prob(a_t|s_t) for the episode
    
    policy_loss = []
    for log_prob, G_t in zip(saved_log_probs, returns):
        policy_loss.append(-log_prob * G_t) # Negative for minimization
    
    policy_loss = torch.stack(policy_loss).sum() # Sum over the episode
    # Or if batching episodes, sum over batch dimension as well after stacking
    ```

- 优化：
    ```python
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    policy_loss.backward() # Autograd computes gradients
    # Optional: torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm)
    optimizer.step() # Updates network weights
    ```
    
- 改进 - 基线 (Baseline)：为了减小梯度的方差，通常会从回报中减去一个基线值(例如状态价值函数)。这不改变梯度的期望，但能显著降低方差。

### 3. Actor-Critic 方法

结合了基于价值和基于策略的方法：
- **Actor** (演员)：负责选择动作，即策略 
  $$
   πθ(a∣s) 
  $$
  。

- **Critic (评论家)**：负责评估 Actor 选择的动作的好坏，即价值函数 $V_ϕ(s)$ 或 $Q_ϕ(s,a)$

#### a. Advantage Actor-Critic (A2C / A3C)

- 理念：Actor 使用 Critic 提供的优势函数 (Advantage Function) $A(s,a)=Q(s,a)−V(s)$ 来指导策略更新。通常 Critic 使用贝尔曼方程来近似 $V(s)$，Actor 使用策略梯度来更新 $θ$。

- 数学核心 
    - Critic Loss (Value Loss)：基于 TD 误差，通常是均方误差：$$ L(ϕ)=E_πθ[(r+γV_ϕ (S_{t+1})−V_ϕ(S_t))^2]$$ 
    - Actor Loss (Policy Loss)：基于策略梯度：$$ L(θ)=E_πθ[logπθ(A_t∣S_t)(r+γV_ϕ (S_{t+1})−V_ϕ(S_t))]$$ 

- PyTorch 底层视角与实现：
    - 网络结构: 通常有两个独立的网络 (Actor 和 Critic)，或者一个共享底层参数、上层分为两个头的网络。
    ```python
        class ActorCritic(nn.Module):
        def __init__(self, state_dim, action_dim, hidden_dim):
            super(ActorCritic, self).__init__()
            self.shared_layers = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU()
            )
            self.actor_head = nn.Linear(hidden_dim, action_dim) # Logits for policy
            self.critic_head = nn.Linear(hidden_dim, 1)      # State value

        def forward(self, state):
            shared_features = self.shared_layers(state)
            action_logits = self.actor_head(shared_features)
            state_value = self.critic_head(shared_features)
            return action_logits, state_value
    ```
    - 损失与优化
    ```python
    # For each step (s, a, r, s_next, done)
    action_logits, state_value = model(s) # model is an ActorCritic instance
    _, next_state_value = model(s_next) # Detach if not end-to-end training on value path

    # Critic update
    td_target = r + gamma * next_state_value * (1 - done)
    advantage = td_target - state_value # A_t
    value_loss = F.mse_loss(state_value, td_target.detach()) # Critic learns V(s)
    # .detach() on td_target is crucial to stop gradients from flowing from critic loss to actor params through state_value

    # Actor update
    action_dist = Categorical(logits=action_logits)
    log_prob = action_dist.log_prob(a)
    # advantage.detach() ensures advantage is treated as a constant for policy gradient
    policy_loss = -(log_prob * advantage.detach()).mean() # or .sum()

    # Entropy bonus (optional, encourages exploration)
    entropy_loss = -action_dist.entropy().mean()

    # Total loss
    total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss # Example weights

    optimizer.zero_grad()
    total_loss.backward() # Autograd computes gradients for both actor and critic parts
    optimizer.step()
    ```
## 6. PyTorch RL 底层注意事项与进阶
- 数据流与梯度：
    -理解 detach() 的使用：在计算 TD 目标或优势时，目标网络的值或价值函数的输出通常需要 .detach()，以防止梯度流向它们，因为它们被视为固定的目标或基线。
    -共享参数：在 Actor-Critic 中，如果 Actor 和 Critic 共享参数，backward() 会自动处理梯度的正确分配。
- GPU 加速：
    - 通过 .to(device) 将网络和张量移动到 GPU。
    - device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
- Batching:
    - DQN 从经验回放池中采样 mini-batch。
    - A2C/A3C (A3C 是异步版本) 通常会收集一定数量的并行经验（例如，多个 actor 并行收集数据）或一个 batch 的轨迹。
    PyTorch 的张量操作天然支持 batch 处理，确保输入网络的张量具有正确的 (batch_size, ...) 维度。
- Tensor Shapes: RL 中张量形状的转换和对齐非常关键，unsqueeze(), squeeze(), gather(), view(), permute() 等函数会经常用到。
- torch.distributions: 对于基于策略的方法，尤其是处理连续动作空间或需要计算熵时，torch.distributions 模块（如 Categorical, Normal）非常有用。
- 高效实现:
    - Vectorized Environments: 使用 gymnasium.vector.AsyncVectorEnv 或 SyncVectorEnv 可以并行运行多个环境实例，大幅提高数据采样效率。
    - Gradient Clipping: torch.nn.utils.clip_grad_norm_ 或 torch.nn.utils.clip_grad_value_ 可以防止梯度爆炸，稳定训练。
- 进阶算法:
    - Proximal Policy Optimization (PPO): 目前非常流行且表现稳健的 Actor-Critic 算法。
    - Soft Actor-Critic (SAC): 基于最大熵 RL 框架的 Actor-Critic 算法，在连续控制任务上表现优异。
    - Distributional RL (C51, Rainbow DQN): 学习回报的完整分布而不是期望值。
    - Model-Based RL: 学习环境模型，并利用模型进行规划或生成模拟经验。

## 7. 总结
PyTorch 为强化学习研究和应用提供了一个非常灵活和强大的平台。通过 nn.Module 构建复杂的策略和价值网络，利用 autograd 进行自动微分，并通过 optim 更新参数，研究者可以相对容易地实现从基础到前沿的各种 RL 算法。理解 RL 的核心数学原理（尤其是贝尔曼方程和策略梯度定理）并将其与 PyTorch 的组件相结合，是成功应用深度强化学习的关键。
