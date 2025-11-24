import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. 构造玩具数据：两个高斯分布混合 (双峰)
def get_data(n=1000):
    x1 = torch.randn(n // 2, 2) + torch.tensor([2.0, 2.0]) # 簇1
    x2 = torch.randn(n // 2, 2) + torch.tensor([-2.0, -2.0]) # 簇2
    return torch.cat([x1, x2], dim=0)

# 2. 定义一个极简的 Score Network (MLP)
# 输入: [x, y, t], 输出: [grad_x, grad_y]
class ScoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2)
        )
    
    def forward(self, x, t):
        # t 需要扩展到和 x 一样的 batch size
        t_embed = t.view(-1, 1).expand(x.shape[0], 1)
        inp = torch.cat([x, t_embed], dim=1)
        return self.net(inp)

# 3. 训练 Score Matching
model = ScoreNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
dataset = get_data(2000)

for epoch in range(1000):
    # 随机采样 t (0 到 1)
    t = torch.rand(dataset.shape[0]) 
    # 前向扩散：加噪声 x_t = x_0 + t * z (简化版 SDE: dx = dw)
    noise = torch.randn_like(dataset)
    x_t = dataset + t.view(-1, 1) * noise
    
    # 预测 Score
    # 理论 Score 目标 ≈ -noise / t (这里简化处理直接预测 noise)
    # 实际 Diffusion 通常预测 noise，Score = -pred_noise / sigma
    pred_noise = model(x_t, t)
    
    loss = torch.mean((pred_noise - noise)**2) # 简单的 Denoising Score Matching
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 4. 可视化向量场 (Vector Field)
# 我们观察 t=0.5 时，模型学到的"力场"
grid_x, grid_y = np.meshgrid(np.linspace(-4, 4, 20), np.linspace(-4, 4, 20))
grid_pts = torch.tensor(np.stack([grid_x, grid_y], axis=-1), dtype=torch.float32).reshape(-1, 2)
t_val = torch.ones(grid_pts.shape[0]) * 0.5

with torch.no_grad():
    # 预测噪声
    pred_noise = model(grid_pts, t_val)
    # Score 方向 = -预测噪声方向 (去噪方向)
    score_vectors = -pred_noise.numpy()

plt.figure(figsize=(8, 8), dpi=120)
plt.scatter(dataset[:, 0], dataset[:, 1], alpha=0.1, color='blue', label='Real Data')
plt.quiver(grid_x, grid_y, score_vectors[:, 0], score_vectors[:, 1], color='red', alpha=0.8)
plt.title(f"Learned Score Field (Gradients pointing to Data)", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()