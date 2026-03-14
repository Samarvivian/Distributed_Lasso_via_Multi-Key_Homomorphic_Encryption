import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================================
# 配置与样式
# ============================================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
# 在 plt.rcParams['font.size'] = 12 附近修改或添加：
plt.rcParams['xtick.labelsize'] = 16  # 横坐标刻度数字字体大小
plt.rcParams['ytick.labelsize'] = 16  # 纵坐标刻度数字字体大小

# 定义颜色 (专业配色)
COLORS = {
    'Computation': '#4C72B0',  # 深蓝
    'Communication': '#DD8452',  # 橙色
    'Decryption': '#55A868',  # 绿色
    'CRC (Ours)': '#C44E52'  # 红色 (突出显示)
}


def plot_latency_breakdown(csv_path):
    # 1. 加载数据
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    # 2. 归一化/合并指标 (单位: ms)
    # 根据你的 timing_log.pdf 指标分类：
    # 计算类：t_mask_ms, t_cheby_ms, t_wupdate
    df['Computation'] = df['t_mask_ms'] + df['t_cheby_ms'] + df.get('t_wupdate', 0)
    # 通信类：t_xcollect, t_broadcast
    df['Communication'] = df['t_xcollect_ms'] + df['t_broadcast_ms']
    # 联合解密/刷新类：t_vdecrypt, t_udecrypt
    df['Decryption'] = df['t_vdecrypt_ms'] + df['t_udecrypt_ms']
    # 你的核心贡献
    df['CRC (Ours)'] = df['t_crc_ms']

    # 只取前 50 轮或感兴趣的样本
    plot_df = df.head(50)
    iters = plot_df['iter']

    # 3. 绘图
    fig, ax = plt.subplots(figsize=(12, 6))

    categories = ['Computation', 'Communication', 'Decryption', 'CRC (Ours)']
    bottom = np.zeros(len(plot_df))

    for cat in categories:
        ax.bar(iters, plot_df[cat], bottom=bottom, label=cat, color=COLORS[cat], width=0.8)
        bottom += plot_df[cat]

    # 4. 细节美化
    ax.set_xlabel('Iteration', fontsize=20)
    ax.set_ylabel('Latency (ms)', fontsize=20)

    ax.legend(loc='upper right', frameon=True, fontsize=16)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # 5. 打印关键统计数据（用于写论文）
    total_avg = bottom.mean()
    crc_avg = df['t_crc_ms'].mean()
    print("-" * 40)
    print(f"Latency Statistics (Averaged over {len(df)} iters):")
    print(f"Average Total Latency: {total_avg:.2f} ms")
    print(f"Average CRC Latency:   {crc_avg:.2f} ms")
    print(f"CRC Overhead Ratio:    {(crc_avg / total_avg) * 100:.4f}%")
    print("-" * 40)

    plt.tight_layout()
    plt.savefig('latency_breakdown.pdf', bbox_inches='tight')
    plt.show()


# 执行
# 请确保将 timing_log.pdf 的数据导出为 timing_log.csv
if __name__ == "__main__":
    # 假设你已经把 PDF 数据转成了 CSV
    plot_latency_breakdown('build/timing_log.csv')