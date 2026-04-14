import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, signal
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# 时间设置
total_time = 0.5
dt = 0.001
time = np.arange(0, total_time, dt)

# 稀疏 diffusion force 输出
update_interval = 0.02
update_steps = int(update_interval / dt)
Fdf_series = np.zeros_like(time)

sparse_times = np.arange(0, total_time, update_interval)

# 更复杂的人类力信号生成函数
def generate_complex_force(t):
    return (
        0.6 * np.sin(2 * np.pi * t) +
        0.3 * np.sin(5 * np.pi * t + 0.5) +
        0.2 * np.sin(9 * np.pi * t + 1.5)
    ) * (1 + 0.2 * np.random.randn(*t.shape))

sparse_forces = generate_complex_force(sparse_times)

for i, t in enumerate(sparse_times):
    idx = int(t / dt)
    Fdf_series[idx:] = sparse_forces[i]

# 方法1：DMP 插值
Fff = np.zeros_like(time)
Fff_dot = np.zeros_like(time)
Fff_ddot = np.zeros_like(time)
alpha = 900.0
beta = 300.0
for i in range(1, len(time)):
    Fff_ddot[i] = alpha * (beta * (Fdf_series[i] - Fff[i-1]) - Fff_dot[i-1])
    Fff_dot[i] = Fff_dot[i-1] + Fff_ddot[i] * dt
    Fff[i] = Fff[i-1] + Fff_dot[i] * dt

# 方法2：样条 + 滤波
def improved_interpolation(sparse_t, sparse_f, dense_t, cutoff=5.0):
    cs = interpolate.CubicSpline(sparse_t, sparse_f)
    interp_f = cs(dense_t)
    nyq = 0.5 * (1/dt)
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(4, normal_cutoff, btype='low')
    return signal.filtfilt(b, a, interp_f)

improved_Fff = improved_interpolation(sparse_times, sparse_forces, time)

# 方法3：指数衰减插值法
def exponential_decay_interpolation(sparse_t, sparse_f, dense_t, alpha=20.0):
    result = np.zeros_like(dense_t)
    idx = 0
    for i, t in enumerate(dense_t):
        if idx + 1 < len(sparse_t) and t >= sparse_t[idx + 1]:
            idx += 1
        last_target = sparse_f[idx]
        if i == 0:
            result[i] = last_target
        else:
            result[i] = result[i-1] + alpha * (last_target - result[i-1]) * dt
    return result

expdecay_Fff = exponential_decay_interpolation(sparse_times, sparse_forces, time)

# 方法4：改进的高斯过程插值
def gp_interpolation(sparse_t, sparse_f, dense_t):
    kernel = 1.0 * RBF(length_scale=0.1) + WhiteKernel(noise_level=0.05)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-4, normalize_y=True)
    gp.fit(sparse_t[:, None], sparse_f)
    pred, _ = gp.predict(dense_t[:, None], return_std=True)
    return pred

gp_Fff = gp_interpolation(sparse_times, sparse_forces, time)

# 画图对比
plt.figure(figsize=(16, 8))
plt.step(sparse_times, sparse_forces, where='post', label="Sparse Force Input", linestyle='--', color='black')
plt.plot(time, Fff, label="DMP Method", color='blue')
# plt.plot(time, improved_Fff, label="Spline + Filter", color='red', linestyle='-.')
# plt.plot(time, expdecay_Fff, label="Exp. Decay (Improved)", color='green', linestyle=':')
# plt.plot(time, gp_Fff, label="GP Interpolation (Improved)", color='purple', linestyle='--')
plt.title("Force Interpolation Methods Comparison")
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# 性能指标函数
def calculate_performance(ref_signal, test_signal):
    corr = signal.correlate(ref_signal, test_signal, mode='full')
    delay = corr.argmax() - (len(ref_signal) - 1)
    rmse = np.sqrt(np.mean((ref_signal - test_signal) ** 2))
    return delay, rmse

# 理想目标信号
ideal_signal = generate_complex_force(time)

# 性能指标输出
methods = {
    "DMP": Fff,
    "Spline+Filter": improved_Fff,
    "ExpDecay": expdecay_Fff,
    "GP": gp_Fff,
}

print("=== 性能对比（相对理想复杂信号）===")
for name, signal_f in methods.items():
    delay, rmse = calculate_performance(ideal_signal, signal_f)
    print(f"{name:15s} → Delay: {delay*dt*1000:6.1f} ms, RMSE: {rmse:.4f}")

plt.show()
