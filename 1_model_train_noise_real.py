import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
import random

from helper_functions.models import Model_Cond_Diffusion, Model_mlp_diff_embed
from helper_functions.data_split import RobotCustomDataset
from chazhi_pinghua.mohusuanzi import get_scale, gaussian_1d_smoothing, downsample_1d
curriculum = "blur"   # "blur" or "downsample"

# Set paths and hyperparameters
DATASET_PATH = "my_real_dataset"
SAVE_DATA_DIR = "output/real_model_bread" 
os.makedirs(SAVE_DATA_DIR, exist_ok=True)

#LOG_DIR = "logs/fit/mohusuanzi" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S_noise_mohusuanzi")
LOG_DIR = "logs/fit/real_model_bread" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S_1")
os.makedirs(LOG_DIR, exist_ok=True)

n_epoch = 2500   ###1500
lrate = 1e-3  
device = "cuda" if torch.cuda.is_available() else "cpu" 
n_hidden = 512 
batch_size = 4096 #4096 
n_T = 50
net_type = "fc"
drop_prob = 0.0
train_prop = 0.80
use_prev = True
noise_prob = 0.3  # 30%的概率添加噪声
noise_magnitude = 0.1  # 10%的噪声幅度

Model_save_name = "real_model_bread.pth"
state_dataset = 'robot_state_train.pkl'
action_dataset = 'robot_action_train.pkl'

# Load training and validation data
tf = transforms.Compose([])

torch_data_train = RobotCustomDataset(
    DATASET_PATH, transform=tf, data_usage="train", train_prop=train_prop,
    state_dataset=state_dataset, action_dataset=action_dataset
)
dataload_train = DataLoader(
    torch_data_train, batch_size=batch_size, shuffle=True, num_workers=0
)

torch_data_val = RobotCustomDataset(
    DATASET_PATH, transform=tf, data_usage="valid", train_prop=train_prop,
    state_dataset=state_dataset, action_dataset=action_dataset
)
dataload_val = DataLoader(
    torch_data_val, batch_size=batch_size, shuffle=False, num_workers=0
)

x_shape = torch_data_train.state_all.shape[1]
y_dim = torch_data_train.action_all.shape[1]

# Initialize the model and optimizer
nn_model = Model_mlp_diff_embed(
    x_shape, n_hidden, y_dim, embed_dim=128, net_type=net_type, use_prev=use_prev
).to(device)

model = Model_Cond_Diffusion(
    nn_model,
    betas=(1e-4, 0.02),
    n_T=n_T,
    device=device,
    x_dim=x_shape,
    y_dim=y_dim,
    drop_prob=drop_prob,
    guide_w=0.0,
)
model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=lrate)

# Set up TensorBoard logging
writer = SummaryWriter(log_dir=LOG_DIR)

def apply_noise_augmentation(x_batch, noise_prob=0.3, noise_magnitude=0.1):
    """
    仅对 x_batch 中的第1-12维 和 19-30维（下标从0开始）添加高斯噪声，其他维度保持不变。

    参数:
    x_batch: 输入数据张量 [batch_size, feature_dim]
    noise_prob: 添加噪声的概率 (0~1)
    noise_magnitude: 噪声幅度系数
    """
    batch_size, feature_dim = x_batch.shape
    noise_mask = torch.rand(batch_size) < noise_prob
    noise_mask = noise_mask.to(device)

    # 指定需要加噪的维度（注意 Python 中是左闭右开区间）
    noise_dims = list(range(7, 13)) + list(range(27, 33))  # 1~12 和 19~30（含）

    # 计算特定维度的标准差
    feature_stds = torch.std(x_batch[:, noise_dims], dim=0)
    noise_scale = noise_magnitude * feature_stds

    # 初始化整个噪声张量为0
    noise = torch.zeros_like(x_batch).to(device)
    
    # 只对需要加噪的维度生成高斯噪声
    noise_subset = torch.randn((batch_size, len(noise_dims)), device=device) * noise_scale.unsqueeze(0)
    noise[:, noise_dims] = noise_subset

    # 应用噪声到选中的样本
    x_noisy = x_batch.clone()
    x_noisy[noise_mask] += noise[noise_mask]

    # 拼接原始数据和加噪数据
    mixed_x = torch.cat([x_batch, x_noisy], dim=0)
    
    return mixed_x, noise_mask
# 数据加噪函数
def apply_noise_augmentation1(x_batch, noise_prob=0.3, noise_magnitude=0.1):
    """
    以指定概率对输入数据添加噪声，同时保留原始数据
    
    参数:
    x_batch: 输入数据张量
    noise_prob: 添加噪声的概率 (0-1)
    noise_magnitude: 噪声幅度 (相对于数据范围)
    
    返回:
    包含原始数据和加噪数据的混合批次
    """
    # 创建噪声掩码 - 确定哪些样本需要加噪
    batch_size = x_batch.shape[0]
    noise_mask = torch.rand(batch_size) < noise_prob
    noise_mask = noise_mask.to(device)
    
    # 创建加噪版本
    # 计算每个特征的噪声范围 (基于特征的标准差)
    feature_stds = torch.std(x_batch, dim=0)
    noise_scale = noise_magnitude * feature_stds
    
    # 生成高斯噪声
    noise = torch.randn_like(x_batch) * noise_scale.unsqueeze(0)
    
    # 应用噪声 (仅对选中的样本)
    x_noisy = x_batch.clone()
    x_noisy[noise_mask] += noise[noise_mask]
    
    # 组合原始数据和加噪数据
    mixed_x = torch.cat([x_batch, x_noisy], dim=0)
    
    return mixed_x, noise_mask

# Main training loop
global_step = 1  
for ep in tqdm(range(n_epoch), desc="Epoch"):

    model.train()

    # Learning rate decay
    optim.param_groups[0]["lr"] = lrate * ((np.cos((ep / n_epoch) * np.pi) + 1) / 2)

    # Training loop
    pbar = tqdm(dataload_train)
    for x_batch, y_batch in pbar:
        x_batch = x_batch.type(torch.FloatTensor).to(device)
        y_batch = y_batch.type(torch.FloatTensor).to(device)
        
        ##### 引入数组增强

        # 应用噪声增强 - 创建混合批次
        mixed_x, noise_mask = apply_noise_augmentation(
            x_batch, 
            noise_prob=noise_prob, 
            noise_magnitude=noise_magnitude
        )
        
        # 复制目标值以匹配混合输入
        mixed_y = torch.cat([y_batch, y_batch], dim=0)
        
        # mixed_x = x_batch
        # mixed_y = y_batch
        # 计算模糊算子
        scale = get_scale(cur_step=global_step)
        
        if curriculum in ["blur", "downsample"]:
            # 分割需要处理的维度和保持不变的维度
            part1 = mixed_x[:, 0:7]    
            part2 = mixed_x[:, 7:13]   
            part3 = mixed_x[:, 13:27]   
            part4 = mixed_x[:, 27:33]   
            part5 = mixed_x[:, 33:40]   
            # 对需要处理的部分应用模糊操作 
            if curriculum == "blur":
                part2 = gaussian_1d_smoothing(part2, scale)
                part4 = gaussian_1d_smoothing(part4, scale)
            elif curriculum == "downsample":
                part2 = downsample_1d(part2, scale)
                part4 = downsample_1d(part4, scale)
            # 重新组合所有部分
            x_batch = torch.cat([part1, part2, part3, part4, part5], dim=1)

        # 计算损失
        #loss = model.loss_on_batch(x_batch, y_batch)
        loss = model.loss_on_batch(mixed_x, mixed_y)
        
        optim.zero_grad()
        loss.backward()
        pbar.set_description(f"train loss: {loss.detach().item():.4f}")
        
        # Log training loss to TensorBoard
        writer.add_scalar('training_loss', loss.detach().item(), global_step)
        #writer.add_scalar('noise_aug_ratio', noise_mask.float().mean().item(), global_step)
        global_step += 1
        #print(global_step)

        optim.step()

    if (ep + 1) % 200 == 0:  # Validate every 5 epochs
        model.eval()
        loss_val, n_batch_val = 0, 0
        with torch.no_grad():
            for x_batch_val, y_batch_val in tqdm(dataload_val, desc="Validation_Loss_MSE"):
                x_batch_val = x_batch_val.type(torch.FloatTensor).to(device)
                y_batch_val = y_batch_val.type(torch.FloatTensor).to(device)

                loss_val_inner = model.loss_on_batch(x_batch_val, y_batch_val)
                loss_val += loss_val_inner.detach().item()
                n_batch_val += 1

            avg_loss_val = loss_val / n_batch_val
            writer.add_scalar('validation_loss_mse', avg_loss_val, global_step)
            tqdm.write(f"Epoch {ep+1}, validation loss mse: {avg_loss_val:.4f}")
            
        # 保存模型检查点
        checkpoint_path = os.path.join(SAVE_DATA_DIR, f"checkpoint_epoch_{ep+1}.pth")
        torch.save({
            'epoch': ep + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': loss_val_inner.detach().item(),
        }, checkpoint_path)
        tqdm.write(f"Saved checkpoint at epoch {ep+1} to {checkpoint_path}")

# Close TensorBoard writer
writer.close()

# Save the final trained model
torch.save(model.state_dict(), os.path.join(SAVE_DATA_DIR, Model_save_name))
print(f"Training completed. Model saved to {os.path.join(SAVE_DATA_DIR, Model_save_name)}")