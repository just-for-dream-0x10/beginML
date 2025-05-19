# PyTorch 中的概率编程 (以 Pyro 和 NumPyro 为例)

与 TensorFlow Probability (TFP) 类似，PyTorch 生态中也有强大的库用于概率编程、贝叶斯推断和统计分析。其中最主要的库是 **Pyro** 和 **NumPyro**。Pyro 由 Uber AI Labs 开发，直接构建在 PyTorch 之上。NumPyro 则是 Pyro 的一个姊妹项目，它使用 NumPy 作为其主要后端，并通过 JAX 进行即时编译和硬件加速，通常能提供更高的性能。此外，**GPyTorch** 是一个专门用于高斯过程建模的库，而 **BoTorch** 则基于 GPyTorch 和 PyTorch，专注于贝叶斯优化。

本文档将参照 TFP 相关内容的结构，对 PyTorch 生态中的这些技术进行梳理和总结。

## 1. PyTorch 概率编程的基本原理

Pyro/NumPyro 的核心思想与 TFP 一致，即通过概率模型显式地表示和处理不确定性。概率编程使得用户能够定义数据的生成过程（包含随机性），然后根据观测到的数据，反向推断模型中的未知参数或潜变量。

### 1.1 概率编程的基本概念

这些基础概念在 PyTorch 的概率编程框架中是通用的：

*   **概率分布 (Probability distribution)**: 描述一个随机变量所有可能取值及其对应概率的数学函数。在 Pyro 中，这些分布位于 `pyro.distributions` 模块；在 NumPyro 中，则位于 `numpyro.distributions` 模块。
*   **随机变量 (Random variable)**: 一个其数值由随机现象决定的变量。在 Pyro/NumPyro 中，随机变量通常通过调用 `pyro.sample("name", distribution)` 或 `numpyro.sample("name", distribution)` 语句来声明，其中 "name" 是变量的名称，distribution 是其遵循的概率分布。
*   **联合分布 (Joint distribution)**: 描述多个随机变量同时取特定值的概率。在 Pyro/NumPyro 中，一个概率模型（通常表示为一个 Python 函数，其中包含多个 `sample` 语句）隐式地定义了模型中所有随机变量的联合分布。
*   **条件分布 (Conditional distribution)**: 在给定一个或多个其他随机变量的取值后，某个随机变量的概率分布。模型中变量间的依赖关系自然地形成了条件分布。
*   **先验分布 (Prior distribution)**: 在观测到任何数据之前，我们对模型参数的初始信念或假设，用概率分布表示。
*   **后验分布 (Posterior distribution)**: 在观测到数据之后，根据贝叶斯定理更新得到的参数分布。它结合了先验信念和数据提供的信息。
*   **似然函数 (Likelihood function)**: 给定模型参数的特定值时，观测到当前数据集的概率。它通常表示为 $P(D|\theta)$，其中 $D$ 是数据，$\theta$ 是参数。

### 1.2 贝叶斯推断

贝叶斯推断是概率编程的核心，其数学基础是贝叶斯定理：
$$
P(\theta|D) = \frac{P(D|\theta) \times P(\theta)}{P(D)}
$$
其中：
*   $P(\theta|D)$ 是**后验概率**：在观测到数据 $D$ 后，参数 $\theta$ 的概率。
*   $P(D|\theta)$ 是**似然函数**：给定参数 $\theta$ 时，观测到数据 $D$ 的概率。
*   $P(\theta)$ 是**先验概率**：我们对参数 $\theta$ 的初始信念。
*   $P(D)$ 是**边际似然** (或证据)：观测到数据 $D$ 的总概率，通常作为归一化常数。

Pyro 和 NumPyro 的主要功能就是帮助用户方便地定义模型（即先验 $P(\theta)$ 和似然 $P(D|\theta)$），然后有效地计算或近似后验分布 $P(\theta|D)$。

## 2. Pyro/NumPyro 的主要组件

### 2.1 概率分布 (`pyro.distributions` / `numpyro.distributions`)

Pyro 和 NumPyro 提供了丰富的概率分布库，其功能和种类与 `tfp.distributions` 类似。这些分布构建在 PyTorch (Pyro) 或 JAX (NumPyro) 的张量运算之上，因此可以无缝集成到各自的计算图中，并支持自动微分。

```python
# Pyro 示例
import torch
import pyro
import pyro.distributions as dist

# 创建一个正态分布 (均值为0, 标准差为1)
normal_dist_pyro = dist.Normal(loc=0., scale=1.)

# 从分布中采样
samples_pyro = normal_dist_pyro.sample(sample_shape=torch.Size()) # 生成1000个样本

# 计算某个值的概率密度 (对于连续分布)
# P(X = 0.5) - 注意：对于连续分布，单个点的概率为0，这里是概率密度函数在该点的值
prob_density_pyro = normal_dist_pyro.log_prob(torch.tensor(0.5)).exp()

# 计算某个值的对数概率密度
log_prob_density_pyro = normal_dist_pyro.log_prob(torch.tensor(0.5)) # log(P(X = 0.5))

# 计算累积分布函数 (CDF)
# P(X <= 0.5)
cdf_pyro = normal_dist_pyro.cdf(torch.tensor(0.5))

# 计算分布的熵
entropy_pyro = normal_dist_pyro.entropy()

print(f"Pyro Normal(0,1) - Sample mean: {samples_pyro.mean():.4f}")
print(f"Pyro Normal(0,1) - log_prob(0.5): {log_prob_density_pyro:.4f}")
print(f"Pyro Normal(0,1) - CDF(0.5): {cdf_pyro:.4f}")
print(f"Pyro Normal(0,1) - Entropy: {entropy_pyro:.4f}")

# NumPyro 示例 (语法非常相似)
import numpyro
import numpyro.distributions as dist_np
import jax.numpy as jnp
from jax import random

key = random.PRNGKey(0) # JAX 需要显式的伪随机数生成器密钥
normal_dist_numpyro = dist_np.Normal(loc=0., scale=1.)
samples_numpyro = normal_dist_numpyro.sample(key, (1000,))
log_prob_density_numpyro = normal_dist_numpyro.log_prob(0.5)

print(f"\nNumPyro Normal(0,1) - Sample mean: {samples_numpyro.mean():.4f}")
print(f"NumPyro Normal(0,1) - log_prob(0.5): {log_prob_density_numpyro:.4f}")
```

常见的分布类型包括：
- 连续分布 (Continuous distributions)：Normal, Beta, Gamma, StudentT, Uniform, Exponential, Laplace, LogNormal, Chi2 等。
- 离散分布 (Discrete distributions)：Bernoulli, Binomial, Categorical, Poisson, Geometric, NegativeBinomial 等。
- 多元分布 (Multivariate distributions)：MultivariateNormal, Dirichlet, Multinomial, Wishart, LKJCholesky (用于协方差矩阵的先验) 等。
- 转换分布 (Transformed distributions)：通过对基础分布应用可逆变换得到新的分布，如 TransformedDistribution。
- 混合分布 (Mixture distributions)：如 MixtureSameFamily (混合相同类型的分布) 或 Mixture (更通用的混合)。

### 2.2 概率模型与联合分布
在 Pyro 和 NumPyro 中，概率模型通常被定义为一个 Python 函数。这个函数内部使用 pyro.sample (或 numpyro.sample) 语句来声明模型中的随机变量（包括潜变量和参数）及其遵循的概率分布。这些 sample 语句的序列以及它们之间的依赖关系（例如，一个 sample 语句的分布参数可能依赖于先前 sample 语句的结果）共同隐式地定义了模型中所有随机变量的联合概率分布。
```python
# Pyro 模型示例: 简单的线性回归
# x_data 和 y_data 是观测到的数据
# def linear_regression_model_pyro(x_data, y_data):
#     # 先验分布
#     # weight ~ Normal(0, 1)
#     weight = pyro.sample("weight", dist.Normal(0., 1.))
#     # bias ~ Normal(0, 5)
#     bias = pyro.sample("bias", dist.Normal(0., 5.))
#     # sigma (观测噪声的标准差) ~ Uniform(0, 10)
#     sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
#
#     # 模型的确定性部分 (均值函数)
#     mean_prediction = weight * x_data + bias
#
#     # 似然函数 (观测模型)
#     # 使用 pyro.plate 来表示数据点的条件独立性
#     # len(y_data) 是数据点的数量
#     with pyro.plate("data_plate", size=len(y_data)):
#         # y_observed ~ Normal(mean_prediction, sigma)
#         pyro.sample("obs", dist.Normal(mean_prediction, sigma), obs=y_data)

# NumPyro 模型示例 (非常相似的结构)
# def linear_regression_model_numpyro(x_data, y_data=None): # y_data=None 允许在采样时仅运行先验
#     weight = numpyro.sample("weight", dist_np.Normal(0., 1.))
#     bias = numpyro.sample("bias", dist_np.Normal(0., 5.))
#     sigma = numpyro.sample("sigma", dist_np.Uniform(0., 10.))
#
#     mean_prediction = weight * x_data + bias
#
#     # 在 NumPyro 中，plate 也是通过 numpyro.plate 实现的
#     with numpyro.plate("data_plate", size=x_data.shape): # 假设x_data和y_data的第一个维度是数据点
#         numpyro.sample("obs", dist_np.Normal(mean_prediction, sigma), obs=y_data)
```

与 TFP 中显式的 `JointDistributionCoroutine` 或 `JointDistributionSequential` 不同，Pyro/NumPyro 中的模型函数通过其执行流程和 sample 语句隐式定义联合分布。pyro.plate (或 numpyro.plate) 是一个重要的上下文管理器，用于声明数据子集之间的条件独立性，这对于高效处理批量数据至关重要。

### 2.3 变分推断 (Pyro: pyro.infer.SVI, NumPyro: numpyro.infer.SVI)
变分推断 (Variational Inference, VI) 是 Pyro 和 NumPyro 中的核心推断算法之一。它将后验推断问题转化为一个优化问题：寻找一个简单参数化分布族（称为变分族或指导函数 guide）中的某个分布 
$$
q(\theta;\phi)
$$
，使其尽可能接近真实的后验分布 
$$
P(\theta|D)
$$
。这种“接近”通常通过最小化它们之间的 KL 散度 (Kullback-Leibler divergence) 来衡量，这等价于最大化证据下界 (Evidence Lower Bound, ELBO)。

```python
# Pyro 变分推断示例 (续上文线性回归模型)
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam # Pyro 提供的优化器，也可以用 torch.optim

# 1. 定义模型 (如上例: linear_regression_model_pyro)

# 2. 定义指导函数 (guide)，即变分后验 q(θ; φ)
# guide 函数应该与 model 函数具有相同的参数签名 (除了观测值)
# 并且包含与 model 中每个非观测 pyro.sample 语句对应的 pyro.sample 语句
# 但这些 sample 语句的分布是由 pyro.param 定义的变分参数参数化的。
# def linear_regression_guide_pyro(x_data, y_data): # y_data 在 guide 中通常不用，但为了签名一致
#     # 变分参数 (φ): 每个参数都有其均值 (loc) 和标准差的对数 (scale_log)
#     # 使用 pyro.param 来声明这些可学习的参数
#     w_loc = pyro.param("w_loc", torch.tensor(0.))
#     w_scale_log = pyro.param("w_scale_log", torch.tensor(0.))
#     w_scale = torch.exp(w_scale_log) # 确保 scale 为正
#
#     b_loc = pyro.param("b_loc", torch.tensor(0.))
#     b_scale_log = pyro.param("b_scale_log", torch.tensor(0.))
#     b_scale = torch.exp(b_scale_log)
#
#     sigma_loc_log = pyro.param("sigma_loc_log", torch.tensor(0.)) # sigma 通常是正的，用 LogNormal 或约束
#     sigma_scale = pyro.param("sigma_scale", torch.tensor(1.0), constraint=dist.constraints.positive) # 另一种方法
#
#     # 变分分布
#     pyro.sample("weight", dist.Normal(w_loc, w_scale))
#     pyro.sample("bias", dist.Normal(b_loc, b_scale))
#     # 对于受约束的参数如 sigma (必须为正)，可以使用 TransformedDistribution
#     # 或者选择一个本身就是正值的分布，如 LogNormal
#     # 这里简化，假设用 Normal，然后依赖优化和约束
#     # 一个更好的做法是使用 dist.LogNormal 或通过 softplus 变换来确保正性
#     sigma_param_unconstrained = pyro.param("sigma_param_unconstrained", torch.tensor(1.))
#     sigma = pyro.sample("sigma", dist.TransformedDistribution(
#                                     dist.Normal(sigma_param_unconstrained, torch.tensor(0.1)),
#                                     dist.transforms.ExpTransform() # 保证 sigma 为正
#                                 ))

# 假设 x_data_tensor, y_data_tensor 已定义
# pyro.clear_param_store() # 清除全局参数存储中的任何先前参数
#
# # 设置优化器
# adam_params = {"lr": 0.005} # 学习率
# optimizer = Adam(adam_params)
#
# # 创建 SVI 对象
# svi = SVI(model=linear_regression_model_pyro,
#           guide=linear_regression_guide_pyro,
#           optim=optimizer,
#           loss=Trace_ELBO()) # Trace_ELBO 是一类通用的ELBO计算方法
#
# # 训练循环
# num_epochs = 2000
# for epoch in range(num_epochs):
#     # svi.step() 会计算损失的梯度并更新参数
#     loss = svi.step(x_data_tensor, y_data_tensor)
#     if epoch % 100 == 0:
#         print(f"Epoch {epoch} : ELBO = {-loss:.4f}") # SVI 返回的是 -ELBO

# NumPyro 变分推断示例 (结构类似)
# from numpyro.infer import SVI, Trace_ELBO
# from numpyro.optim import Adam as AdamNumPyro # NumPyro 的优化器
#
# # 定义 NumPyro 的 model 和 guide (如 linear_regression_model_numpyro 和相应的 guide_numpyro)
# optimizer_numpyro = AdamNumPyro(step_size=0.005)
# svi_numpyro = SVI(model=linear_regression_model_numpyro,
#                   guide=linear_regression_guide_numpyro,
#                   optim=optimizer_numpyro,
#                   loss=Trace_ELBO())
#
# # svi_result = svi_numpyro.run(random.PRNGKey(1), num_steps=2000, x_data=x_data_np, y_data=y_data_np)
# # params_numpyro = svi_result.params # 获取学到的变分参数
# # w_loc_learned = params_numpyro["w_loc"]
```

### 2.4 马尔可夫链蒙特卡洛 (Pyro: pyro.infer.mcmc, NumPyro: numpyro.infer.mcmc)
马尔可夫链蒙特卡洛 (Markov Chain Monte Carlo, MCMC) 方法是另一类主要的后验推断技术。它们通过构建一个马尔可夫链，使其平稳分布恰好是目标后验分布 
$$
P(θ∣D)
$$
，然后从这条链中抽取样本来近似后验。NumPyro 的 MCMC 实现因其基于 JAX 的后端而特别高效，尤其是其 NUTS (No-U-Turn Sampler) 算法。

```python

# NumPyro MCMC 示例 (NumPyro 在 MCMC 方面通常性能更优且更受推荐)
# import numpyro
# import numpyro.distributions as dist_np
# from numpyro.infer import MCMC, NUTS
# from jax import random

# 假设 linear_regression_model_numpyro(x_data, y_data) 已定义
# key_mcmc = random.PRNGKey(0)
#
# # 初始化 NUTS 内核
# nuts_kernel = NUTS(model=linear_regression_model_numpyro)
#
# # 创建 MCMC 对象
# mcmc = MCMC(kernel=nuts_kernel,
#             num_warmup=500,       # "燃烧期" 或 "预热期" 步数
#             num_samples=1000,     # 实际采样步数
#             num_chains=1,         # 运行的马尔可夫链数量 (并行运行多条链是好做法)
#             progress_bar=True)    # 显示进度条
#
# # 运行 MCMC
# # 假设 x_data_np 和 y_data_np 是 JAX/NumPy 数组
# mcmc.run(key_mcmc, x_data=x_data_np, y_data=y_data_np)
#
# # 打印采样结果摘要 (均值, 标准差, 分位数等)
# mcmc.print_summary()
#
# # 获取后验样本
# posterior_samples_numpyro = mcmc.get_samples()
# # posterior_samples_numpyro 是一个字典，键是模型中的参数名 (如 "weight", "bias", "sigma")

# Pyro MCMC 示例 (语法相似，但性能和底层实现不同)
# from pyro.infer import MCMC, NUTS
#
# # 假设 linear_regression_model_pyro(x_data_tensor, y_data_tensor) 已定义
# nuts_kernel_pyro = NUTS(model=linear_regression_model_pyro)
# mcmc_pyro = MCMC(kernel=nuts_kernel_pyro,
#                  warmup_steps=500,
#                  num_samples=1000,
#                  num_chains=1)
#
# mcmc_pyro.run(x_data_tensor, y_data_tensor) # 传入模型参数
#
# posterior_samples_pyro = mcmc_pyro.get_samples()
# # Pyro 的 MCMC 结果也是一个字典
```

Hamiltonian Monte Carlo (HMC) 及其自适应变体 NUTS 是目前最先进的 MCMC 算法之一，尤其适用于高维连续参数空间。它们利用目标分布的梯度信息来指导采样过程，从而实现更有效的探索。

## 3. 实际应用案例
下面将简要介绍如何在 PyTorch 生态中实现与 TFP 文档中类似的概率模型。

### 3.1 贝叶斯线性回归
使用 Pyro/NumPyro 实现贝叶斯线性回归的步骤如下：

1. 定义模型函数：如 linear_regression_model_pyro 或 linear_regression_model_numpyro 所示，为权重 (weight) 和偏置 (bias) 设置先验分布（例如正态分布），为观测噪声的标准差 (sigma) 设置先验（例如均匀分布或半正态分布），并定义似然函数（通常是正态分布，其均值为 
$$
weight\times x+bias
$$
，标准差为 sigma）。

2. 选择推断算法：

- 变分推断 (SVI)：定义一个 guide 函数，为模型中的每个潜变量（weight, bias, sigma）指定参数化的变分分布（例如，都使用正态分布，其均值和标准差是可学习的 pyro.param 或 numpyro.param）。然后使用 SVI 对象通过优化 ELBO 来学习这些变分参数。

- MCMC: 使用 MCMC 对象（通常配合 NUTS 核）直接从后验分布中采样。

3. 结果分析：
- 对于 VI，学到的变分参数（如 w_loc, w_scale）直接描述了近似后验分布。
- 对于 MCMC，得到的后验样本集合可以用来计算参数的均值、中位数、置信区间（例如，95% 可信区间），以及绘制后验分布的直方图或密度图。
- 可以从后验（或近似后验）中采样参数，然后对新的 x 值进行预测，从而得到预测分布，并量化预测的不确定性。

可视化部分可以使用 matplotlib 或 seaborn，与 TFP 示例中的代码类似，只需将 TensorFlow/TFP 的张量和分布操作替换为 PyTorch/Pyro 或 JAX/NumPy/NumPyro 的操作。

### 3.2 贝叶斯神经网络 (BNN)
在 Pyro/NumPyro 中，构建贝叶斯神经网络 (BNN) 的核心思想是为网络的权重和偏置参数赋予先验分布，而不是将它们视为固定的点估计。

```python
# Pyro BNN 模型示例 (概念性)
import torch.nn as nn
import pyro.nn as pynn # Pyro 提供的用于构建贝叶斯神经网络的模块

# class BayesianNN(pynn.PyroModule): # 继承自 PyroModule
#     def __init__(self, in_features, hidden_features, out_features):
#         super().__init__()
#         self.fc1 = pynn.PyroLinear(in_features, hidden_features) # 贝叶斯线性层
#         self.fc1.weight = pynn.PyroSample(dist.Normal(0., 1.).expand([hidden_features, in_features]).to_event(2))
#         self.fc1.bias = pynn.PyroSample(dist.Normal(0., 1.).expand([hidden_features]).to_event(1))
#
#         self.fc2 = pynn.PyroLinear(hidden_features, out_features)
#         self.fc2.weight = pynn.PyroSample(dist.Normal(0., 1.).expand([out_features, hidden_features]).to_event(2))
#         self.fc2.bias = pynn.PyroSample(dist.Normal(0., 1.).expand([out_features]).to_event(1))
#
#         self.relu = nn.ReLU()
#
#     def forward(self, x, y=None): # y 是观测值
#         x = x.view(-1, self.fc1.in_features)
#         x = self.relu(self.fc1(x))
#         mu = self.fc2(x).squeeze()
#         sigma = pyro.sample("sigma_obs", dist.Uniform(0., 1.)) # 观测噪声
#
#         with pyro.plate("data", x.shape):
#             obs = pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
#         return mu

# 对应的 Guide 函数 (例如，均值场变分族):
# class BayesianNNGuide(pynn.PyroModule):
#     def __init__(self, in_features, hidden_features, out_features):
#         super().__init__()
#         # 为每个参数定义变分参数和分布 (通常是正态分布)
#         # 例如，对于 fc1.weight:
#         self.fc1_weight_loc = nn.Parameter(torch.randn(hidden_features, in_features))
#         self.fc1_weight_scale = nn.Parameter(torch.randn(hidden_features, in_features))
#         # ... 其他参数类似 ...
#         self.sigma_obs_loc = nn.Parameter(torch.tensor(0.5)) # 假设
#
#     def forward(self, x, y=None):
#         # 从变分分布中采样，对应模型中的 PyroSample 语句
#         # fc1_weight = pyro.sample("fc1.weight", dist.Normal(self.fc1_weight_loc, torch.exp(self.fc1_weight_scale)).to_event(2))
#         # ...
#         # sigma_obs = pyro.sample("sigma_obs", dist.LogNormal(self.sigma_obs_loc, torch.tensor(0.1))) # 确保正性
#         # 通常在 guide 中，我们不需要实际执行神经网络的前向传播，只需要确保所有模型中的参数都有对应的变分采样。
#         # 一个更简单的方法是使用 pyro.infer.autoguide 模块，它可以自动生成常见的 guide 函数，如 AutoDiagonalNormal。

# model_bnn = BayesianNN(input_size, hidden_size, output_size)
# guide_bnn = pyro.infer.autoguide.AutoDiagonalNormal(model_bnn) # 自动生成均值场正态guide
# svi_bnn = SVI(model_bnn, guide_bnn, optimizer, loss=Trace_ELBO())
# # 训练循环...
```

`pynn.PyroModule, pynn.PyroLinear`, 和` pynn.PyroSample` 使得将标准 `PyTorch nn.Module` 转换为贝叶斯版本更加容易。推断同样可以使用 SVI (配合 AutoGuide 或手动定义的 guide) 或 MCMC。

### 3.3 高斯过程回归 (GPyTorch)
对于高斯过程 (Gaussian Processes, GP)，GPyTorch 是 PyTorch 生态系统中的首选库。它被设计为模块化、可扩展，并能充分利用 PyTorch 的 GPU 加速和自动微分能力。

```python
# GPyTorch 示例 (概念性)
import gpytorch

# 假设 train_x_tensor 和 train_y_tensor 是 PyTorch 张量
# class ExactGPModel(gpytorch.models.ExactGP): # 精确 GP 模型
#     def __init__(self, train_x, train_y, likelihood):
#         super().__init__(train_x, train_y, likelihood)
#         # 定义均值函数 (例如，常数均值)
#         self.mean_module = gpytorch.means.ConstantMean()
#         # 定义核函数 (例如，RBF 核，也叫平方指数核)
#         self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
#
#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         # 返回多元正态分布作为 GP 的预测
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# # 初始化似然函数 (例如，高斯似然，表示观测噪声)
# likelihood = gpytorch.likelihoods.GaussianLikelihood()
# model_gp = ExactGPModel(train_x_tensor, train_y_tensor, likelihood)

# # 训练 GP 模型 (通过优化边际对数似然来学习超参数)
# model_gp.train()
# likelihood.train()
#
# # 使用 PyTorch 的优化器
# optimizer_gp = torch.optim.Adam(model_gp.parameters(), lr=0.1) # model.parameters() 包括核函数和均值函数的参数
#
# # "损失" 函数是负的边际对数似然 (Negative Marginal Log Likelihood)
# mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model_gp)
#
# training_iter = 100
# for i in range(training_iter):
#     optimizer_gp.zero_grad()
#     output = model_gp(train_x_tensor) # 模型前向传播，得到预测分布
#     loss = -mll(output, train_y_tensor) # 计算负边际对数似然
#     loss.backward() # 反向传播计算梯度
#     if (i + 1) % 10 == 0:
#         print(f"Iter {i+1}/{training_iter} - Loss: {loss.item():.3f} "
#               f"lengthscale: {model_gp.covar_module.base_kernel.lengthscale.item():.3f} "
#               f"noise: {likelihood.noise.item():.3f}")
#     optimizer_gp.step() # 更新参数

# # 进行预测
# model_gp.eval()
# likelihood.eval()
#
# # test_x_tensor 是新的输入点
# with torch.no_grad(), gpytorch.settings.fast_pred_var(): # 上下文管理器用于快速预测方差
#     observed_pred = likelihood(model_gp(test_x_tensor)) # likelihood() 将潜在函数的预测转换为观测的预测
#     predictive_mean = observed_pred.mean
#     lower_ci, upper_ci = observed_pred.confidence_region() # 获取置信区间 (通常是95%)

```
GPyTorch 提供了多种均值函数、核函数（包括 RBF, Matern, Periodic 等，以及它们的组合）、似然函数和推断策略（包括精确 GP 和各种稀疏 GP 近似，用于处理大规模数据集）。

### 高斯过程回归模型的评估
评估 GP 模型的指标与 TFP 文档中描述的类似：

- 预测准确性指标：
  - 均方误差 (MSE)
  - 均方根误差 (RMSE)
  - 平均绝对误差 (MAE)
  - 决定系数 ($R^2$)
- 不确定性量化指标：
  - 预测区间覆盖率 (Prediction Interval Coverage Probability, PICP)：真实的观测值落在预测区间（例如 95% 置信区间）内的比例。理想情况下应接近区间的名义覆盖率（例如 95%）。
  - 平均预测区间宽度 (Mean Prediction Interval Width, MPIW)：预测区间的平均宽度，越窄越好（在保持良好覆盖率的前提下）。
  - 负对数预测密度 (Negative Log Predictive Density, NLPD)：评估整个预测分布的质量。
- 核参数评估：学习到的核参数（如 RBF 核的 lengthscale 和 outputscale，以及似然的 noise）可以提供关于数据平滑度、变异性和噪声水平的洞见。

这些指标可以使用 GPyTorch 的预测结果和真实的测试数据来计算。

## 4. 高级主题
### 4.1 变分自编码器 (VAE)
VAE 是一种结合了神经网络和变分推断的生成模型。在 Pyro/NumPyro 中，VAE 的实现涉及：

- 编码器 (Encoder) 网络：一个神经网络（例如用 torch.nn.Module 实现），它将输入数据 $x$ 映射到潜变量 $z$ 的近似后验分布 $q_ϕ (z∣x)$ 的参数（通常是正态分布的均值和对数方差）。
- 解码器 (Decoder) 网络：另一个神经网络，它将从潜空间采样的 $z$ 映射回数据空间的分布 $p_θ (x∣z)$ 的参数（例如，对于二值图像是伯努利分布的 logits，对于连续数据是正态分布的均值）。
- 概率模型 (Model)：定义数据的生成过程 $p(x,z)=p_θ (x∣z)p(z)$。其中 $p(z)$ 是潜变量的先验（通常是标准正态分布）。在 Pyro/NumPyro 中，解码器是模型的一部分，潜变量 $z$ 通过 pyro.sample 从先验中采样，然后通过解码器生成 $x$ 的分布参数，最后 pyro.sample 观测值 $x$。
- 指导函数 (Guide)：定义潜变量的近似后验 $q_ϕ (z∣x)$。编码器是指导函数的一部分，它接收输入 $x$，输出 $q_ϕ (z∣x)$ 的参数，然后 pyro.sample 潜变量 $z$ 从这个近似后验中采样。

- 损失函数：使用 SVI 和 ELBO 进行训练。ELBO 可以分解为重建项（期望的 $log p_θ (x∣z)$）和正则化项（$KL(q_ϕ (z∣x)∣∣p(z))$）。

```python
# Pyro VAE 示例 (概念性结构)
# class Encoder(nn.Module):
#     # ... (例如, 卷积层, 全连接层, 输出潜变量分布的参数)
#     def forward(self, x):
#         # ...
#         return z_loc, z_scale
#
# class Decoder(nn.Module):
#     # ... (例如, 全连接层, 转置卷积层, 输出数据分布的参数)
#     def forward(self, z):
#         # ...
#         return reconstruction_params
#
# class VAE(pynn.PyroModule): # 或普通的 nn.Module 配合外部 model/guide
#     def __init__(self, latent_dim, use_cuda=False):
#         super().__init__()
#         self.encoder = Encoder(...)
#         self.decoder = Decoder(...)
#         self.latent_dim = latent_dim
#         # ...
#
#     def model(self, x):
#         # 将解码器注册到 Pyro (如果 VAE 不是 PyroModule，则在模型函数内部 pyro.module("decoder", self.decoder))
#         self.decoder.train() # 确保在训练模式
#         pyro.module("decoder", self.decoder) # 如果VAE继承自nn.Module
#
#         with pyro.plate("data", x.size(0)):
#             # 先验 P(z)
#             z_loc = torch.zeros(x.size(0), self.latent_dim, dtype=x.dtype, device=x.device)
#             z_scale = torch.ones(x.size(0), self.latent_dim, dtype=x.dtype, device=x.device)
#             z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
#
#             # 解码器 P(x|z)
#             loc_img = self.decoder(z) # 解码器输出重建图像的参数
#             # pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.view(-1, 784)) # 例如 MNIST
#             # 或其他合适的观测分布，如 Normal
#
#     def guide(self, x):
#         # 将编码器注册到 Pyro
#         self.encoder.train()
#         pyro.module("encoder", self.encoder) # 如果VAE继承自nn.Module
#
#         with pyro.plate("data", x.size(0)):
#             # 编码器 q(z|x)
#             z_loc, z_scale_raw = self.encoder(x)
#             z_scale = torch.nn.functional.softplus(z_scale_raw) # 确保 scale 为正
#             pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
#
# # 损失函数 (ELBO) 的计算和优化步骤与标准 SVI 流程类似。
# # optimizer_vae = Adam({"lr": 1.0e-3})
# # svi_vae = SVI(model_vae.model, model_vae.guide, optimizer_vae, loss=Trace_ELBO())
# # 训练循环...
```

### 4.2 隐马尔可夫模型 (HMM)
Pyro 和 NumPyro 都支持隐马尔可夫模型 (HMM)。可以直接使用它们提供的分布和控制流原语（如 Python 循环或 Pyro/NumPyro 的 scan 操作）来构建 HMM。NumPyro 还提供了一个高级的 numpyro.distributions.HiddenMarkovModel 分布，类似于 TFP 中的 HMM 分布，它可以简化 HMM 的定义和推断。

```python
# NumPyro HMM 使用高级分布的示例 (概念性)
# key_hmm = random.PRNGKey(2)
# num_timesteps = 100
# num_hidden_states = 2
# num_obs_dim = 1 # 假设一维观测
#
# # 假设我们已经估计或定义了HMM的参数
# initial_probs_np = jnp.array([0.8, 0.2])
# transition_matrix_np = jnp.array([[0.7, 0.3], [0.2, 0.8]])
# # 观测分布参数 (例如，每个隐状态对应一个正态分布的均值和标准差)
# obs_locs_np = jnp.array([-2.0, 2.0])
# obs_scales_np = jnp.array([1.0, 1.5])
#
# # 创建 HMM 分布对象
# hmm_dist = dist_np.HiddenMarkovModel(
#     initial_distribution=dist_np.Categorical(probs=initial_probs_np),
#     transition_distribution=dist_np.Categorical(probs=transition_matrix_np),
#     observation_distribution=dist_np.Normal(loc=obs_locs_np, scale=obs_scales_np), # 观测分布的参数维度应与隐状态数匹配
#     num_steps=num_timesteps
# )
#
# # 从 HMM 中采样观测序列
# # observations_np = hmm_dist.sample(key_hmm) # observations_np.shape 会是 (num_timesteps,)
#
# # 如果要对参数进行推断 (例如，给定观测序列 observations_np)
# # def hmm_parameter_inference_model(observations, num_hidden_states, num_timesteps):
# #     # 为 initial_probs, transition_matrix, obs_locs, obs_scales 设置先验
# #     # init_probs_prior = numpyro.sample("init_probs", dist_np.Dirichlet(jnp.ones(num_hidden_states)))
# #     # ... 其他参数的先验 ...
# #
# #     # 构造 HMM 分布 (参数是上面采样的先验)
# #     hmm_for_inference = dist_np.HiddenMarkovModel(...)
# #     numpyro.sample("obs_seq", hmm_for_inference, obs=observations)
#
# # 然后可以使用 MCMC (如NUTS) 或 SVI 对 HMM 参数进行后验推断。
# # 经典算法如前向算法 (计算似然)、维特比算法 (解码最可能的隐状态序列)、
# # 以及 Baum-Welch 算法 (EM算法用于参数估计) 也可以在 Pyro/NumPyro 框架内实现，
# # 或者其功能可以通过这些库的推断引擎间接获得。
# # 例如，hmm_dist.log_prob(observed_sequence) 会使用前向算法。
# # hmm_dist.posterior_mode(observed_sequence) (如果可用) 或通过 MCMC/VI 推断隐状态。
```

### 4.3 狄利克雷过程混合模型 (DPMM)
狄利克雷过程混合模型 (Dirichlet Process Mixture Model, DPMM) 是一种非参数贝叶斯方法，常用于聚类分析，其优点是能够自动从数据中推断合适的簇的数量。在 Pyro/NumPyro 中，DPMM 通常通过其 Stick-Breaking 构造来实现。

```python
# Pyro DPMM 示例 (概念性结构 - Stick-Breaking)
# def dpmm_model_pyro(data, max_num_components, alpha_dp): # alpha_dp 是狄利克雷过程的浓度参数
#     # Stick-breaking 过程生成混合权重 (pi_k)
#     # beta_k ~ Beta(1, alpha_dp) for k = 1 to max_num_components-1
#     with pyro.plate("beta_plate", max_num_components - 1):
#         beta_k = pyro.sample("beta_k", dist.Beta(1., alpha_dp))
#
#     # 通过 stick-breaking 计算实际的混合权重 pi
#     # pi_k = beta_k * product_{j<k} (1 - beta_j)
#     # 需要 careful indexing 和 cumprod
#     # Pyro 的 dist.DirichletProcess 也可以用来更简洁地定义
#
#     # 每个簇的参数的先验 (例如，高斯混合模型中的均值和方差)
#     with pyro.plate("components_plate", max_num_components):
#         # mu_k ~ Normal(loc_prior, scale_prior)
#         mu_k = pyro.sample("mu_k", dist.Normal(0., 5.)) # 假设的先验
#         # sigma_k ~ LogNormal(loc_prior, scale_prior) 或 InverseGamma
#         sigma_k = pyro.sample("sigma_k", dist.LogNormal(0., 1.))
#
#     # 为每个数据点分配簇并从该簇生成数据
#     with pyro.plate("data_plate", data.shape):
#         # z_n ~ Categorical(pi) (将数据点 n 分配给某个簇)
#         assignment = pyro.sample("assignment", dist.Categorical(probs=calculated_pi_from_beta))
#         # obs_n ~ Normal(mu_k[z_n], sigma_k[z_n])
#         pyro.sample("obs", dist.Normal(mu_k[assignment], sigma_k[assignment]), obs=data)

# 推断 DPMM 通常比较复杂，可能涉及：
# 1. 变分推断：需要仔细设计 guide，可能涉及结构化变分推断。
# 2. MCMC：如Collapsed Gibbs Sampling，或者使用 NUTS 等通用采样器（但可能效率不高）。
# Pyro 和 NumPyro 提供了一些工具来帮助构建这类模型，但具体的实现细节会比较深入。
```

## 5. 变分推断的原理
变分推断 (VI) 的核心原理在所有概率编程框架中都是一致的：

- 目标：用一个简单的、参数化的概率分布 $q(θ;ϕ)$ (称为变分分布或指导函数) 来近似复杂或难以计算的真实后验分布 $p(θ∣D)$。
- 方法：通过优化变分参数 $ϕ$ 来最小化 $q(θ;ϕ)$ 和 $p(θ∣D)$ 之间的 KL 散度:
$$
KL(q(θ;ϕ)∣∣p(θ∣D))=∫q(θ;ϕ)log 
p(θ∣D)
q(θ;ϕ)
dθ
$$

- 证据下界 (ELBO)：直接最小化 KL 散度通常不可行，因为 $p(θ∣D)$ 未知。取而代之的是最大化 ELBO：
$$ 
ELBO(q)=E 
q(θ;ϕ)
​
 [logp(D,θ)]−E 
q(θ;ϕ)
​
 [logq(θ;ϕ)] \\ 

ELBO(q)=E 
q(θ;ϕ)
​
 [logp(D∣θ)]−KL(q(θ;ϕ)∣∣p(θ))
$$

最大化 ELBO 等价于最小化 KL 散度。第一项 
$
E_{q(θ;ϕ)} [logp(D∣θ)]
$
鼓励变分分布解释数据（似然期望），第二项 
$
KL(q(θ;ϕ)∣∣p(θ))
$
鼓励变分分布接近先验分布。

- 均值场假设 (Mean-Field Assumption)：一种常见的简化，假设变分后验可以分解为各参数（或参数组）独立分布的乘积： 
$
q(θ)=∏_i q_i(θ_i)
$ 
。这使得优化更容易，但可能无法捕捉参数间的后验依赖关系。`AutoDiagonalNormal guide` 在 Pyro/NumPyro 中就是一种均值场变分族，其中每个潜变量都由一个独立的正态分布近似。

- 随机变分推断 (Stochastic Variational Inference, SVI)：当数据集较大时，计算 ELBO 中的期望（尤其是对数似然项）可能成本很高。SVI 使用随机梯度上升法，通过对数据的小批量 (mini-batch) 和从 
$
q(θ;ϕ)
$
 中采样的潜变量来估计 ELBO 的梯度。Pyro/NumPyro 中的 SVI 类就是为此设计的。

## 6.MCMC 方法的原理
马尔可夫链蒙特卡洛 (MCMC) 方法通过构建一个特殊的马尔可夫链来从目标后验分布 
$p(θ∣D)$ 中生成样本。

- 核心思想：设计一个马尔可夫链，使其唯一的平稳分布就是我们想要采样的后验分布 
$p(θ∣D)$
。如果让这个链运行足够长的时间，它生成的样本就会近似于从 
$p(θ∣D)$
中抽取的独立同分布样本。

- Metropolis-Hastings (MH) 算法：一个基础的 MCMC 算法。
    1. 从当前状态 $θ_t$ 出发，根据一个提议分布 (proposal distribution) $q(θ^*∣θ_t)$ 生成一个候选状态 $θ^*$，并根据接受概率 $α$ 决定是否接受这个候选状态。
    2. 接受概率 $α$ 通常定义为：
    $α=min(1,[p(θ^*∣D)q(θ_t∣θ^*)]/[p(θ_t∣D)q(θ^*∣θ_t)])$
    3. 以概率 $α$ 接受候选状态，即令 θ_{t+1}=θ^*；否则，拒绝候选状态，令 θ_{t+1}=θ_t。
- Hamiltonian Monte Carlo (HMC)：一种更高级的 MCMC 方法，尤其适用于高维连续参数空间。它引入辅助的动量变量 $p$，并利用哈密顿动力学来生成距离当前状态较远且接受概率较高的候选样本。
    1. 将参数 $θ$视为位置，引入动量 $p$。构造哈密顿函数 
    $$ 
    H(θ,p)=−logp(θ∣D)+ \frac{1}{2} p^T M^{-1} p
    $$
    其中 $M$ 是质量矩阵，常取单位矩阵。
    $logp(θ∣D)$ 是似然项，$ \frac{1}{2} p^T M^{-1} p$ 是动能项。
    2. 从当前状态 $(θ_t,p_t)$ 开始 (通常 $p_t$ 从 $N(0,M)$ 中随机抽取)，使用数值积分方法（如 Leapfrog 积分器）模拟哈密顿方程一段时间，得到候选状态 $(θ^*,p^*)$。
    3. 根据能量变化计算接受概率 
    $$
    α=min(1,exp(−H(θ^*,p^*)+H(θ_t,p_t)))
    $$
    (如果使用精确积分器，接受率总是1；数值误差会导致接受率略小于1) HMC 利用目标分布的梯度信息 (通过 $−logp(θ∣D)$ 的梯度) 来指导采样，从而避免了 MH 算法中的随机游走行为，提高了采样效率。
- No-U-Turn Sampler (NUTS)：HMC 的一个自适应扩展，它自动调整 Leapfrog 积分的步数，避免了手动调整 HMC 调优参数的困难，并且能有效地探索后验分布。NUTS 是 NumPyro MCMC 的默认和推荐内核。

## 7. 实际应用中的考虑
与 TFP 文档中提到的类似，在 PyTorch 生态中使用概率编程时也需要注意：
- 模型选择与评估:
    - ELBO (用于 VI)：虽然 ELBO 是变分推断的优化目标，但其绝对值本身不直接等同于模型证据 $P(D)$
。不过，对于不同的模型，在相同数据上比较其最优 ELBO 值可以在一定程度上帮助模型选择。
    - 后验预测检查 (Posterior Predictive Checks, PPC)：从后验分布（或近似后验）中采样参数，然后用这些参数生成模拟数据集。将模拟数据集的统计特性（如均值、方差、分位数、特定事件的频率等）与真实观测数据集的相应统计特性进行比较。Pyro/NumPyro 的 Predictive 类 (pyro.infer.Predictive / numpyro.infer.Predictive) 可以方便地进行后验预测。
    - 信息准则: WAIC (Widely Applicable Information Criterion) 和 LOO-CV (Leave-One-Out Cross-Validation) 是基于后验预测的更复杂的模型比较方法。可以使用像 ArviZ (arviz 库) 这样的外部库来计算这些指标，它可以很好地与 Pyro/NumPyro 的输出集成。
- 计算效率:
    - VI vs MCMC: VI 通常比 MCMC 快得多，尤其适用于大规模数据集和非常复杂的模型，但其准确性受限于所选变分族的表达能力。MCMC 理论上可以收敛到真实后验，但计算成本高，可能需要大量样本和长时间运行。
    - 模型参数化: 模型的参数化方式会显著影响推断的效率和收敛性。例如，对受约束的参数（如标准差必须为正）进行变换（如使用对数变换或 softplus 变换）可以将其映射到无约束空间，从而简化优化或采样。
    - 梯度计算: Pyro 依赖于 PyTorch 的 Autograd 系统，而 NumPyro 依赖于 JAX 的自动微分。两者都非常高效。
    - 硬件加速: PyTorch 和 JAX 都支持 GPU 加速。NumPyro 通常能更好地利用 JAX 的 JIT (Just-In-Time) 编译特性在 GPU/TPU 上获得高性能。
    - 数据子采样 (Subsampling)：对于大规模数据集，可以在 SVI 中使用数据的小批量。对于 MCMC，一些高级算法也支持数据子采样，但实现起来更复杂。
- 诊断与收敛检查:
    - MCMC 诊断:
        - 轨迹图 (Trace plots)：绘制每个参数的后验样本随迭代次数变化的图形，检查链是否平稳混合且没有发散趋势。
        - $ R^{\hat{}}  (R-hat, Gelman-Rubin statistic)$：如果运行了多条并行的马尔可夫链，$R$ 比 较链间方差和链内方差。接近 1 的值表明链已收敛到相似的分布。
        - 有效样本大小 (Effective Sample Size, ESS)：由于 MCMC 样本是自相关的，ESS 衡量了与同样数量的独立样本相比，当前 MCMC 样本所包含的关于后验的信息量。ESS 太低表明需要更多样本或采样效率不高。
        - 自相关图 (Autocorrelation plots)：显示样本与其滞后版本之间的相关性，帮助判断链的混合速度。
NumPyro 的 mcmc.print_summary() 会报告 
$R^{
\hat{}
}$
 和 ESS。ArviZ 库提供了更全面的诊断工具。
- VI 诊断:
    - ELBO 收敛: 监控 ELBO 值是否在训练过程中收敛到一个稳定值。
    - 检查变分参数: 检查学习到的变分参数（例如，近似后验分布的均值和标准差）是否合理。
    - 后验预测检查: 与 MCMC 类似，使用从近似后验生成的样本进行 PPC。
    - 敏感性分析: 检查推断结果（尤其是后验分布）对先验分布选择的敏感性。如果结果对先验的微小改变非常敏感，可能表明数据提供的信息不足，或者模型本身存在问题。

## 8. 概率模型对比与总结 (基于 PyTorch 生态)
下表总结了在 PyTorch 生态中（主要使用 Pyro, NumPyro, GPyTorch）实现的一些主要概率模型的特点：

|模型类型|PyTorch 生态核心库/方法|优点|缺点|适用场景|不确定性表示|
|---|---|---|---|---|---|
|贝叶斯线性回归|Pyro/NumPyro (pyro.sample, dist, SVI/MCMC)|1. 提供参数不确定性<br>2. 防止过拟合<br>3. 适合小数据集|1. 表达能力有限<br>2. 难以捕捉非线性关系|1. 数据量小<br>2. 需要可解释性<br>3. 需量化参数不确定性|参数后验分布|
|高斯过程回归|GPyTorch (gpytorch.models.ExactGP, kernels, likelihoods)|1. 非参数，灵活性高<br>2. 提供完整预测分布<br>3. 小数据集上表现好，可利用GPU加速<br>4. 核工程提供强大建模能力|1. 精确GP计算复杂度高O(N³)(GPyTorch有稀疏GP实现以缓解)<br>2. 对核函数选择敏感<br>3. 高维数据表现欠佳 (维度灾难)|1. 中小规模数据集<br>2. 需要平滑函数假设<br>3. 主动学习、贝叶斯优化 (BoTorch 基于 GPyTorch)|函数值的后验分布 (预测均值和方差)|
|隐马尔可夫模型 (HMM)|Pyro/NumPyro (dist.HiddenMarkovModel 或手动构建)|1. 适合序列数据建模<br>2. 隐状态具有可解释性<br>3. 经典推断算法成熟 (前向、维特比)|1. 受马尔可夫假设限制 (当前状态仅依赖前一状态)<br>2. 隐状态数量通常需预先指定(或通过非参方法推断)<br>3. 难以捕捉长期依赖|1. 时间序列分析<br>2. 语音识别<br>3. 自然语言处理 (词性标注)<br>4. 生物序列分析|隐状态序列的后验概率，参数的后验分布|
|变分自编码器 (VAE)|Pyro/NumPyro + torch.nn|1. 强大的非线性生成能力<br>2. 学习有意义的低维潜表示<br>3. 可扩展到大规模高维数据 (如图像)|1. 训练可能不稳定，对超参敏感<br>2. ELBO 优化可能导致“后验坍塌”<br>3. 生成的样本有时较为模糊|1. 图像、文本等复杂数据生成<br>2. 无监督表示学习<br>3. 异常检测、数据压缩|潜变量的近似后验分布|
|狄利克雷过程混合模型 (DPMM)|Pyro/NumPyro (通常使用 Stick-Breaking 构造 dist.DirichletProcess)|1. 自动从数据中推断簇的数量 (非参数特性)<br>2. 提供聚类不确定性<br>3. 避免了K-Means等方法中预设K值的困难|1. 推断计算复杂度高，收敛可能较慢<br>2. 在非常高维的数据上表现可能不佳<br>3. 模型和推断实现相对复杂|1. 数据中簇数量未知时的聚类分析<br>2. 密度估计<br>3. 主题模型 (Latent Dirichlet Allocation 是其特例)|簇分配的后验概率，簇参数的后验分布，簇数量的后验分布|

## 9. 选择合适模型的指导方针 (PyTorch 生态)
在实际应用中选择合适的概率模型时，应综合考虑以下因素：

1. 数据特征:
    - 数据量 (Data size):
        - 小到中等数据集 (<几千个样本)：贝叶斯线性回归、高斯过程 (GPyTorch 的精确 GP)、HMM。
        - 大规模数据集：变分自编码器 (VAE)、使用 SVI 推断的贝叶斯神经网络、稀疏高斯过程 (GPyTorch)。
    - 数据维度 (Data dimensionality):
        - 低到中等维度：GP 表现良好。
        - 高维度：VAE、BNN 更合适。GP 在高维空间中会遇到“维度灾难”，除非使用特殊的降维核或方法。
    - 数据类型与结构:
        - 序列数据：HMM、循环神经网络 (RNN) 的贝叶斯版本。
        - 图像/视频数据：VAE (通常基于卷积网络)、贝叶斯卷积神经网络。
        - 表格数据：贝叶斯线性回归、GP、BNN。
        - 数据点之间有已知依赖关系（如图结构）：图神经网络 (GNN) 的贝叶斯版本 (更前沿)。
2. 任务需求:
    - 预测 vs. 推断: 是只需要准确的点预测，还是需要完整的预测分布和不确定性量化？或者，任务的重点是理解数据生成过程和参数的意义？
    - 不确定性量化: 如果可靠的不确定性估计至关重要（例如，在医疗、金融、安全关键应用中），则贝叶斯方法（如 GP、BNN、MCMC 推断的参数模型）是首选。
    - 可解释性: 简单模型（如贝叶斯线性回归、具有少量状态的 HMM）通常比复杂模型（如深度 BNN 或 VAE）更具可解释性。GP 的核函数有时也能提供一定的可解释性（例如，周期核表示周期性）。
    - 自动决策: 如果模型输出将用于自动决策（如贝叶斯优化中的采集函数），则预测分布的质量非常重要。BoTorch 是这方面的专用库。
3   . 计算资源与限制:
    - 训练时间: MCMC 通常比 VI 慢得多。VI 的训练时间取决于模型复杂度和数据量。精确 GP 的训练时间随数据量三次方的速度增长。
    - 内存需求: 精确 GP 对内存要求较高。大规模神经网络模型也需要大量内存。
    - 推断速度 (预测时)：训练好的 VI 模型或点估计模型通常预测速度快。从 MCMC 后验样本进行预测可能需要对多个样本进行平均，也相对较快。GP 的预测速度取决于测试点数量和训练点数量。
4. 先验知识的可用性与整合:
    - 贝叶斯方法允许通过先验分布自然地融入领域知识。如果存在强烈的、可靠的先验信息，它能显著改善小数据集上的推断结果。
需要仔细考虑先验的选择，因为不合适的先验可能导致有偏的或不稳定的结果（尤其是在数据量较少时）。可以使用先验预测检查 (prior predictive checks) 来评估先验的合理性。


通过综合权衡这些因素，可以选择最适合特定应用场景的概率模型，并充分利用 PyTorch 生态中概率编程工具的强大功能来解决复杂的不确定性建模问题。