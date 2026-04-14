import os
import time
import xlsxwriter
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import mean_absolute_error, mean_squared_error
from helper_functions.data_split import RobotCustomDataset
from helper_functions.models import Model_Cond_Diffusion, Model_mlp_diff_embed

# 配置参数
DATASET_PATH = "dataset"
SAVE_FIGURE_DIR = "figures/tacdiffusion_yuan_noise0.5"  # 修改目录以区分噪声测试
FIGURE_ACTION_DIR = os.path.join(SAVE_FIGURE_DIR, "action_plots")
FIGURE_STATE_DIR = os.path.join(SAVE_FIGURE_DIR, "state_plots")
FIGURE_ERROR_DIR = os.path.join(SAVE_FIGURE_DIR, "error_plots")
FIGURE_NOISE_DIR = os.path.join(SAVE_FIGURE_DIR, "noise_plots")  # 新增噪声可视化目录
interval_length = 2000  # 每个区间长度
device = "cuda" if torch.cuda.is_available() else "cpu"

# 噪声参数
NOISE_PROB = 0.3  # 添加噪声的概率
magnitude = 0.5  # 10%的噪声幅度
NOISE_AMPLITUDE = 0.0
# 模型参数
MODEL_PATH = "output/TacDiffusion_model_512_mohusuanzi_deepseek_gaussian_3_4.pth"
x_shape = 36  # 输入维度
y_dim = 6     # 输出维度
n_hidden = 512
net_type = "fc"
drop_prob = 0.0

# 初始化模型
def load_diffusion_model():
    nn_model = Model_mlp_diff_embed(
        x_shape, n_hidden, y_dim, 
        embed_dim=128, net_type=net_type, use_prev=True
    ).to(device)
    
    diffusion_model = Model_Cond_Diffusion(
        nn_model,
        betas=(1e-4, 0.02),
        n_T=50,
        device=device,
        x_dim=x_shape,
        y_dim=y_dim,
        drop_prob=drop_prob,
        guide_w=0.0,
    ).to(device)
    
    diffusion_model.load_state_dict(torch.load(MODEL_PATH))   ##如果是最终的pth
    #checkpoint = torch.load(MODEL_PATH, map_location=device)       ##如果是中间过程的pth
    #diffusion_model.load_state_dict(checkpoint["model_state_dict"])    ##如果是中间过程的pth

    diffusion_model.eval()
    return diffusion_model

# 数据加载
def load_test_data():
    tf = transforms.Compose([])
    dataset = RobotCustomDataset(
        DATASET_PATH, 
        transform=tf,
        data_usage="test",
        train_prop=0.01,
        state_dataset='robot_state_test.pkl',
        action_dataset='robot_action_test.pkl'
    )
    return dataset

# 可视化配置
label_pred = ['f_x_pred', 'f_y_pred', 'f_z_pred', 'tau_x_pred', 'tau_y_pred', 'tau_z_pred']
label_true = ['f_x_true', 'f_y_true', 'f_z_true', 'tau_x_true', 'tau_y_true', 'tau_z_true']
# data_labels = [
#     'f_x_ext', 'f_y_ext', 'f_z_ext', 'tau_x_ext', 'tau_y_ext', 'tau_z_ext',
#     'f_x_in', 'f_y_in', 'f_z_in', 'tau_x_in', 'tau_y_in', 'tau_z_in',
#     'v_x', 'v_y', 'v_z', 'w_x', 'w_y', 'w_z'
# ]
# subplot_labels = [
#     'wrench_ext_f', 'wrench_ext_tau',
#     'wrench_inner_f', 'wrench_inner_tau',
#     'linear velocity', 'angular velocity'
# ]

def add_noise(x, noise_prob=NOISE_PROB, noise_amp=NOISE_AMPLITUDE):
    """仅对前12维和第19-30维添加噪声"""
    noisy_x = x.copy()
    noise_mask_total = np.zeros_like(x, dtype=bool)
    noise_total = np.zeros_like(x)

    # 对前12维添加噪声
    noise_mask_1 = np.random.rand(*x[..., :12].shape) < noise_prob
    noise_1 = np.random.uniform(low=-noise_amp, high=noise_amp, size=x[..., :12].shape)
    noisy_x[..., :12][noise_mask_1] += noise_1[noise_mask_1]
    noise_mask_total[..., :12] = noise_mask_1
    noise_total[..., :12] = noise_1

    # 对第19到30维（Python中索引为18到30）添加噪声
    noise_mask_2 = np.random.rand(*x[..., 18:30].shape) < noise_prob
    noise_2 = np.random.uniform(low=-noise_amp, high=noise_amp, size=x[..., 18:30].shape)
    noisy_x[..., 18:30][noise_mask_2] += noise_2[noise_mask_2]
    noise_mask_total[..., 18:30] = noise_mask_2
    noise_total[..., 18:30] = noise_2

    return noisy_x, noise_mask_total, noise_total

# def add_noise(x, noise_prob=NOISE_PROB, noise_amp=NOISE_AMPLITUDE):
#     """
#     仅对指定维度添加噪声
#     - 维度 7~12 和 19~30 （注意：Python 索引从 0 开始）
#     """
#     noisy_x = x.copy()
#     noise_mask_total = np.zeros_like(x, dtype=bool)
#     noise_total = np.zeros_like(x)

#     # 指定需要加噪的维度（注意 Python 中是左闭右开区间）
#     noise_dims = list(range(7, 13)) + list(range(27, 33))

#     for dim in noise_dims:
#         # 生成噪声 mask
#         noise_mask = np.random.rand(*x[..., dim].shape) < noise_prob
#         # 生成噪声值
#         #noise = np.random.uniform(low=-noise_amp, high=noise_amp, size=x[..., dim].shape)
#         noise = np.random.uniform(low=-8, high=2, size=x[..., dim].shape)
#         # 应用噪声
#         noisy_x[..., dim][noise_mask] += noise[noise_mask]
#         noise_mask_total[..., dim] = noise_mask
#         noise_total[..., dim] = noise

#     return noisy_x, noise_mask_total, noise_total
def add_noise_numpy(x, noise_prob=NOISE_PROB, noise_magnitude=magnitude):
    """
    与训练阶段 apply_noise_augmentation 完全等效的 NumPy 噪声增强逻辑。
    支持 1D 和 2D 输入，保证可以平替旧 add_noise。
    """

    x = np.asarray(x)
    x_copy = x.copy()

    # ---------- 处理输入维度 ----------
    single_sample = False
    if x.ndim == 1:            # 输入是 (feature_dim,)
        x = x.reshape(1, -1)   # 变成 (1, feature_dim)
        single_sample = True

    batch_size, feature_dim = x.shape

    # 选哪些样本加噪（按训练逻辑）
    noise_mask = np.random.rand(batch_size) < noise_prob

    # 指定加噪维度
    noise_dims = list(range(7, 13)) + list(range(27, 33))

    # 按维度计算标准差
    feature_stds = np.std(x[:, noise_dims], axis=0)
    feature_stds = np.where(feature_stds == 0, 1e-6, feature_stds)

    # 生成噪声
    noise = np.zeros_like(x)
    noise_subset = (
        np.random.randn(batch_size, len(noise_dims)) 
        * (noise_magnitude * feature_stds)
    )
    noise[:, noise_dims] = noise_subset

    # 应用噪声
    noisy_x = x.copy()
    noisy_x[noise_mask] += noise[noise_mask]

    # ---------- 恢复 1D 输出 ----------
    if single_sample:
        noisy_x = noisy_x[0]
        noise_mask = noise_mask[0]

    return noisy_x, noise_mask
def main():
    # 创建保存目录
    os.makedirs(SAVE_FIGURE_DIR, exist_ok=True)
    os.makedirs(FIGURE_ACTION_DIR, exist_ok=True)
    os.makedirs(FIGURE_STATE_DIR, exist_ok=True)
    os.makedirs(FIGURE_ERROR_DIR, exist_ok=True)
    os.makedirs(FIGURE_NOISE_DIR, exist_ok=True)  # 创建噪声可视化目录

    # 加载模型和数据
    model = load_diffusion_model()
    test_data = load_test_data()
    
    # 初始化数据存储
    error_df = pd.DataFrame(columns=["Interval", "MAE", "RMSE", "Inference Speed", "Noise Fraction"])
    all_y_true = []
    all_y_pred = []
    speeds = []
    noise_fractions = []  # 记录每个区间的噪声比例

    # 分区间处理
    num_intervals = 3
    # for interval in range(2, num_intervals):
    for interval in range(num_intervals):   # 0, 1, 2, 3        
        start_idx = interval * interval_length
        end_idx = start_idx + interval_length
        indices = range(start_idx, end_idx)
        
        # 准备存储
        y_pred = np.zeros((interval_length, y_dim))
        states = test_data.state_all[start_idx:end_idx].copy()
        noisy_states = np.zeros_like(states)  # 存储添加噪声后的状态
        noise_masks = np.zeros(states.shape, dtype=bool)  # 存储噪声掩码
        actions_true = test_data.action_all[start_idx:end_idx]
        
        # 记录噪声统计
        total_elements = 0
        noisy_elements = 0
        
        # 推理循环
        start_time = time.time()
        with torch.no_grad():
            for i, idx in enumerate(indices):
                # 获取原始状态数据
                original_x = test_data.state_all[idx].copy()
                
                # 添加噪声
                noisy_x, noise_mask, _ = add_noise(original_x)
                #noisy_x, noise_mask = add_noise_numpy(original_x)
                noisy_states[i] = noisy_x
                noise_masks[i] = noise_mask
                
                # 更新噪声统计
                total_elements += original_x.size
                noisy_elements += np.sum(noise_mask)
                
                # 转换为Tensor并送入模型
                x_tensor = torch.FloatTensor(noisy_x).to(device)
                x_tensor = x_tensor.unsqueeze(0)  # 添加batch维度
                
                # 模型推理
                pred = model(x_tensor)  # 直接调用forward
                y_pred[i] = pred.cpu().numpy().flatten()
                # ###act加权
                # y_pred[-3]=y_pred[-2] =y_pred[-1] = y_pred[0]
                # y_pred[i] = 0.4 * y_pred[i] + 0.3 * y_pred[i-1] + 0.2 * y_pred[i-2] + 0.1 * y_pred[i-3]
        
        # 计算噪声比例
        noise_fraction = noisy_elements / total_elements
        noise_fractions.append(noise_fraction)
        print(f"Interval {interval}: Added noise to {noisy_elements}/{total_elements} elements ({noise_fraction*100:.2f}%)")
        
        # 计算指标
        inference_time = time.time() - start_time
        speed = interval_length / inference_time
        print(f"Interval {interval}: inference speeds: {speed} sample/second")

        speeds.append(speed)
        
        mae = mean_absolute_error(actions_true, y_pred)
        rmse = np.sqrt(mean_squared_error(actions_true, y_pred))
        
        # 构造新的 DataFrame 行
        new_row = pd.DataFrame([{
            "Interval": f"{start_idx}-{end_idx}",
            "MAE": mae,
            "RMSE": rmse,
            "Inference Speed": speed,
            "Noise Fraction": noise_fraction
        }])

        # 使用 pd.concat 追加
        error_df = pd.concat([error_df, new_row], ignore_index=True)

        all_y_true.append(actions_true)
        all_y_pred.append(y_pred)

        # 绘制噪声可视化图
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        # 随机选择一个维度展示噪声效果
        dim_idx = np.random.randint(0, states.shape[1])
        plt.plot(states[:, dim_idx], 'b-', label='Original State')
        plt.plot(noisy_states[:, dim_idx], 'r--', alpha=0.6, label='Noisy State')
        plt.title(f"State Dimension {dim_idx} with Noise (Interval {interval})")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        noise_impact = np.abs(noisy_states - states)
        avg_noise_impact = np.mean(noise_impact, axis=0)
        plt.bar(range(states.shape[1]), avg_noise_impact)
        plt.title("Average Noise Impact per Dimension")
        plt.xlabel("Dimension Index")
        plt.ylabel("Average Noise Magnitude")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_NOISE_DIR, f"noise_impact_{start_idx}-{end_idx}.png"))
        plt.close()

        # 绘制动作对比图
        fig, axs = plt.subplots(3, 2, figsize=(15, 10))
        for dim in range(6):
            row, col = divmod(dim, 2)
            axs[row, col].plot(y_pred[:, dim], label=label_pred[dim])
            axs[row, col].plot(actions_true[:, dim], '--', label=label_true[dim])
            axs[row, col].set_title(f"{label_pred[dim]} Comparison (Noise Added)")
            axs[row, col].legend()
            axs[row, col].grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_ACTION_DIR, f"actions_{start_idx}-{end_idx}.png"))
        plt.close()

        # # 绘制状态数据图（包含噪声）
        # fig, axs = plt.subplots(3, 2, figsize=(15, 10))
        # for plot_idx in range(6):
        #     row, col = divmod(plot_idx, 2)
        #     start_dim = plot_idx * 3
        #     for offset in range(3):
        #         dim = start_dim + offset
        #         if dim < 18:
        #             axs[row, col].plot(noisy_states[:, dim], label=f"Noisy {data_labels[dim]}")
        #     axs[row, col].set_title(f"{subplot_labels[plot_idx]} (With Noise)")
        #     axs[row, col].legend()
        #     axs[row, col].grid(True)
        # plt.tight_layout()
        # plt.savefig(os.path.join(FIGURE_STATE_DIR, f"states_{start_idx}-{end_idx}.png"))
        # plt.close()

    # 计算总体指标
    y_true_all = np.vstack(all_y_true)
    y_pred_all = np.vstack(all_y_pred)
    overall_mae = mean_absolute_error(y_true_all, y_pred_all)
    overall_rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
    avg_speed = np.mean(speeds)
    avg_noise_fraction = np.mean(noise_fractions)
    
    new_row = pd.DataFrame([{
        "Interval": "Overall",
        "MAE": overall_mae,
        "RMSE": overall_rmse,
        "Inference Speed": avg_speed,
        "Noise Fraction": avg_noise_fraction
    }])
    error_df = pd.concat([error_df, new_row], ignore_index=True)

    # 保存结果
    error_df.to_excel(os.path.join(SAVE_FIGURE_DIR, "metrics_with_noise.xlsx"), index=False)
    
    # 绘制误差直方图
    plt.figure(figsize=(12, 6))
    # intervals = [f"{i*interval_length}-{(i+1)*interval_length}" for i in range(2, num_intervals)]
    intervals = [f"{i*interval_length}-{(i+1)*interval_length}" for i in range(num_intervals)]
    mae_values = [mean_absolute_error(t, p) for t, p in zip(all_y_true, all_y_pred)]
    plt.bar(intervals, mae_values)
    plt.title("MAE per Interval (With Input Noise)")
    plt.xlabel("Interval")
    plt.ylabel("MAE")
    plt.grid(True)
    plt.savefig(os.path.join(FIGURE_ERROR_DIR, "mae_distribution_with_noise.png"))
    plt.close()

    # 绘制噪声比例与误差关系图
    plt.figure(figsize=(10, 6))
    plt.scatter(noise_fractions, mae_values, s=100)
    for i, txt in enumerate(intervals):
        plt.annotate(txt, (noise_fractions[i], mae_values[i]), fontsize=9)
    plt.title("Noise Fraction vs MAE")
    plt.xlabel("Noise Fraction")
    plt.ylabel("MAE")
    plt.grid(True)
    plt.savefig(os.path.join(FIGURE_ERROR_DIR, "noise_vs_mae.png"))
    plt.close()

    print(f"鲁棒性测试完成！结果保存在：{SAVE_FIGURE_DIR}")
    print(f"平均噪声比例：{avg_noise_fraction*100:.2f}%")
    print(f"总体MAE：{overall_mae:.4f}, 总体RMSE：{overall_rmse:.4f}")

if __name__ == "__main__":
    main()