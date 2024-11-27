import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from exp.exp_main import Exp_Main
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # 设置与训练时相同的参数
    parser.add_argument('--model', type=str, default='CycleNet')
    parser.add_argument('--root_path', type=str, default='./dataset/')
    parser.add_argument('--data_path', type=str, default='electricity.csv')
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--seq_len', type=int, default=720)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--enc_in', type=int, default=320)
    parser.add_argument('--cycle', type=int, default=168)
    parser.add_argument('--model_type', type=str, default='mlp')
    parser.add_argument('--model_id', type=str, default='Electricity_720_96')
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--use_revin', type=int, default=1)
    parser.add_argument('--random_seed', type=int, default=1024)
    
    args = parser.parse_args()
    return args

def load_model_and_data(args):
    # 构建模型保存路径
    setting = '{}_{}_{}_ft{}_sl{}_pl{}_cycle{}_{}_seed{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.pred_len,
        args.cycle,
        args.model_type,
        args.random_seed
    )
    
    # 初始化实验
    exp = Exp_Main(args)
    
    # 加载训练好的模型
    checkpoint_path = f'./checkpoints/{setting}/checkpoint.pth'
    model = exp.model
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    # 获取数据
    data_loader = exp._get_data(flag='test')
    return model, data_loader

def extract_and_visualize(model, data_loader, sample_idx=0, channel_idx=0):
    # 获取周期模式
    periodic_patterns = model.cycleQueue.data.detach().cpu().numpy()
    
    # 获取一批数据
    batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(data_loader))
    
    # 获取时间索引
    time_index = torch.arange(model.cycle_len)
    sequence_length = batch_x.shape[1]
    
    # 生成周期部分
    periodic_component = model.cycleQueue(time_index[:1], sequence_length)
    periodic_component = periodic_component.detach().cpu().numpy()
    
    # 获取原始数据
    original_data = batch_x[sample_idx, :, channel_idx].numpy()
    
    # 计算残差
    residual = original_data - periodic_component[0, :, channel_idx]
    
    # 可视化
    plt.figure(figsize=(15, 10))
    
    # 原始数据
    plt.subplot(3, 1, 1)
    plt.plot(original_data)
    plt.title('Original Time Series')
    plt.grid(True)
    
    # 周期模式
    plt.subplot(3, 1, 2)
    plt.plot(periodic_component[0, :, channel_idx])
    plt.title('Periodic Pattern')
    plt.grid(True)
    
    # 残差
    plt.subplot(3, 1, 3)
    plt.plot(residual)
    plt.title('Residual')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('time_series_decomposition.png')
    plt.close()
    
    # 保存数据到CSV
    df = pd.DataFrame({
        'original': original_data,
        'periodic': periodic_component[0, :, channel_idx],
        'residual': residual
    })
    df.to_csv('time_series_components.csv', index=False)
    
    return periodic_patterns, original_data, periodic_component, residual

if __name__ == '__main__':
    args = get_args()
    model, data_loader = load_model_and_data(args)
    
    # 提取并可视化周期模式（默认选择第一个样本的第一个通道）
    periodic_patterns, original, periodic, residual = extract_and_visualize(
        model, data_loader, sample_idx=0, channel_idx=0
    )
    
    print("Data saved to 'time_series_components.csv'")
    print("Plot saved as 'time_series_decomposition.png'")
    
    # 打印一些基本统计信息
    print("\nBasic Statistics:")
    print(f"Original data variance: {np.var(original):.4f}")
    print(f"Periodic component variance: {np.var(periodic):.4f}")
    print(f"Residual variance: {np.var(residual):.4f}")