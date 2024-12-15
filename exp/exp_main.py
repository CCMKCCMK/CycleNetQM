from datetime import datetime
import pandas as pd
import torch.amp
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Linear,CycleNet,CycleNetMM,CycleNetQQ,CycleNetQM,GRU, LSTM
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')
def plot_comparison(training_data, weights, residual, save_dir='./results'):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 创建三个子图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    
    # 1. Training Data (蓝色)
    ax1.plot(training_data, color='blue', label='Normalized Training Data (Channel 320)')
    ax1.set_ylim(-3, 3)
    ax1.grid(True)
    ax1.legend()
    
    # 2. Weights (橙色)
    ax2.plot(weights, color='orange', label='Expanded Weights (Channel 320)')
    ax2.set_ylim(-3, 3)
    ax2.grid(True)
    ax2.legend()
    
    # 3. Residual (红色)
    ax3.plot(residual, color='red', label='Residual (Training Data - Weights)')
    ax3.set_ylim(-3, 3)
    ax3.grid(True)
    ax3.legend()
    
    # 设置x轴标签
    ax3.set_xlabel('Time Steps')
    
    # 设置y轴标签
    ax1.set_ylabel('Normalized value')
    ax2.set_ylabel('Normalized value')
    ax3.set_ylabel('Residual value')
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(os.path.join(save_dir, f'comparison_{timestamp}.png'))
    plt.close()
    
    print(f"Comparison plot saved as: comparison_{timestamp}.png")

def detailed_analysis(trues_remain, preds_remain, Q, trues_last, preds_last, Q_repeated, save_dir='./results'):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 创建多个子图
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Remain vs True Remain
    plt.subplot(3, 3, 1)
    plt.plot(preds_remain, label='Pred Remain', alpha=0.7)
    plt.plot(trues_remain, label='True Remain', alpha=0.7)
    plt.title('Pred vs True Remain')
    plt.legend()
    plt.grid(True)
    
    # 2. Q数据
    plt.subplot(3, 3, 2)
    plt.plot(Q[:, -1], label='Q', color='red', alpha=0.7)
    plt.title('Q Values (One Cycle)')
    plt.grid(True)
    
    # 3. True vs Pred (原始值)
    plt.subplot(3, 3, 3)
    plt.plot(preds_last[-len(Q_repeated):], label='Pred', alpha=0.7)
    plt.plot(trues_last[-len(Q_repeated):], label='True', alpha=0.7)
    plt.title('True vs Predicted Values')
    plt.legend()
    plt.grid(True)
    
    # 4. Remain和True Remain的直方图对比
    plt.subplot(3, 3, 4)
    plt.hist(preds_remain, bins=50, alpha=0.5, label='Pred Remain')
    plt.hist(trues_remain, bins=50, alpha=0.5, label='True Remain')
    plt.title('Remain Distribution Comparison')
    plt.legend()
    
    # 5. Box Plot比较
    plt.subplot(3, 3, 5)
    plt.boxplot([preds_remain, trues_remain, Q[:, -1]], 
                labels=['Pred Remain', 'True Remain', 'Q'])
    plt.title('Box Plot Comparison')
    
    # 6. 预测误差分析
    error = preds_remain - trues_remain
    plt.subplot(3, 3, 6)
    plt.plot(error, label='Prediction Error')
    plt.title('Prediction Error (Pred Remain - True Remain)')
    plt.grid(True)
    
    # 7. Rolling mean比较
    window = min(len(Q_repeated) // 10, 100)
    plt.subplot(3, 3, 7)
    plt.plot(pd.Series(preds_remain).rolling(window=window).mean(), 
            label='Pred Remain Rolling Mean')
    plt.plot(pd.Series(trues_remain).rolling(window=window).mean(), 
            label='True Remain Rolling Mean')
    plt.title(f'Rolling Mean Comparison (window={window})')
    plt.legend()
    plt.grid(True)
    
    # 8. Scatter plot: Pred Remain vs True Remain
    plt.subplot(3, 3, 8)
    plt.scatter(trues_remain, preds_remain, alpha=0.1)
    plt.plot([min(trues_remain), max(trues_remain)], 
             [min(trues_remain), max(trues_remain)], 'r--')
    plt.xlabel('True Remain')
    plt.ylabel('Pred Remain')
    plt.title('Pred vs True Remain Scatter')
    
    # 9. Error Distribution
    plt.subplot(3, 3, 9)
    plt.hist(error, bins=50)
    plt.title('Error Distribution')
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(os.path.join(save_dir, f'detailed_analysis_{timestamp}.png'))
    plt.close()

    # 新建一个图形用于展示truth和pred的对比
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Truth vs Truth Remain对比
    ax1.plot(trues_last[-len(Q_repeated):], label='Truth', alpha=0.7)
    ax1.plot(trues_remain, label='Truth Remain', alpha=0.7)
    ax1.plot(Q_repeated, label='Q', alpha=0.5, linestyle='--')
    ax1.set_title('Truth vs Truth Remain Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # Pred vs Pred Remain对比
    ax2.plot(preds_last[-len(Q_repeated):], label='Prediction', alpha=0.7)
    ax2.plot(preds_remain, label='Pred Remain', alpha=0.7)
    ax2.plot(Q_repeated, label='Q', alpha=0.5, linestyle='--')
    ax2.set_title('Prediction vs Pred Remain Comparison')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    # 保存新的对比图
    plt.savefig(os.path.join(save_dir, f'value_remain_comparison_{timestamp}.png'))
    plt.close()

    # 创建周期叠加图
    fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # 计算每个周期的数据点数
    cycle_length = len(Q[:, -1])
    num_cycles = len(Q_repeated) // cycle_length
    
    # Truth的周期叠加图
    for i in range(num_cycles):
        start_idx = i * cycle_length
        end_idx = (i + 1) * cycle_length
        ax1.plot(range(cycle_length), 
                trues_last[-len(Q_repeated):][start_idx:end_idx], 
                alpha=0.3, label=f'Cycle {i+1}' if i < 5 else None)
    ax1.plot(range(cycle_length), Q[:, -1], 'r--', label='Q', linewidth=2)
    ax1.set_title('Truth Values - Cycle Overlay')
    if num_cycles > 5:
        ax1.legend(['Cycles 1-5', 'Q'])
    else:
        ax1.legend()
    ax1.grid(True)
    
    # Prediction的周期叠加图
    for i in range(num_cycles):
        start_idx = i * cycle_length
        end_idx = (i + 1) * cycle_length
        ax2.plot(range(cycle_length), 
                preds_last[-len(Q_repeated):][start_idx:end_idx], 
                alpha=0.3, label=f'Cycle {i+1}' if i < 5 else None)
    ax2.plot(range(cycle_length), Q[:, -1], 'r--', label='Q', linewidth=2)
    ax2.set_title('Predicted Values - Cycle Overlay')
    if num_cycles > 5:
        ax2.legend(['Cycles 1-5', 'Q'])
    else:
        ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    # 保存周期叠加图
    plt.savefig(os.path.join(save_dir, f'cycle_overlay_{timestamp}.png'))
    plt.close()

    # 保存详细统计信息到文本文件
    stats_file = os.path.join(save_dir, f'statistics_{timestamp}.txt')
    with open(stats_file, 'w') as f:
        f.write("Statistical Analysis\n")
        f.write("==================\n\n")
        
        f.write("Pred Remain Statistics:\n")
        f.write(f"Mean: {np.mean(preds_remain):.4f}\n")
        f.write(f"Std: {np.std(preds_remain):.4f}\n")
        f.write(f"Min: {np.min(preds_remain):.4f}\n")
        f.write(f"Max: {np.max(preds_remain):.4f}\n")
        f.write(f"Median: {np.median(preds_remain):.4f}\n\n")
        
        f.write("True Remain Statistics:\n")
        f.write(f"Mean: {np.mean(trues_remain):.4f}\n")
        f.write(f"Std: {np.std(trues_remain):.4f}\n")
        f.write(f"Min: {np.min(trues_remain):.4f}\n")
        f.write(f"Max: {np.max(trues_remain):.4f}\n")
        f.write(f"Median: {np.median(trues_remain):.4f}\n\n")
        
        f.write("Q Statistics:\n")
        f.write(f"Mean: {np.mean(Q[:, -1]):.4f}\n")
        f.write(f"Std: {np.std(Q[:, -1]):.4f}\n")
        f.write(f"Min: {np.min(Q[:, -1]):.4f}\n")
        f.write(f"Max: {np.max(Q[:, -1]):.4f}\n")
        f.write(f"Median: {np.median(Q[:, -1]):.4f}\n\n")
        
        f.write("Error Statistics:\n")
        f.write(f"Mean Absolute Error: {np.mean(np.abs(error)):.4f}\n")
        f.write(f"Root Mean Square Error: {np.sqrt(np.mean(error**2)):.4f}\n")
        f.write(f"Error Std: {np.std(error):.4f}\n")
        
    # 保存数据到CSV
    data_dict = {
        'pred_remain': preds_remain,
        'true_remain': trues_remain,
        'Q': Q_repeated,
        'prediction_error': error
    }
    df = pd.DataFrame(data_dict)
    csv_path = os.path.join(save_dir, f'data_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"Results saved to {save_dir}")
    print(f"Image: detailed_analysis_{timestamp}.png")
    print(f"Data: data_{timestamp}.csv")
    print(f"Statistics: statistics_{timestamp}.txt")


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Linear': Linear,
            'CycleNet': CycleNet,
            'CycleNetMM': CycleNetMM,
            'CycleNetQQ': CycleNetQQ,
            'CycleNetQM': CycleNetQM,
            'GRU': GRU,
            'LSTM': LSTM
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'CycleNetMM', 'CycleNetQQ', 'CycleNetQM', 'CycleNet'}):
                            outputs, _ = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'GRU'}):
                            outputs, _ = self.model(batch_x)
                        
                else:
                    if any(substr in self.args.model for substr in {'CycleNetMM', 'CycleNetQQ', 'CycleNetQM', 'CycleNet'}):
                            outputs, _ = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear', 'GRU'}):
                        outputs, _ = self.model(batch_x)
    
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        if self.args.model in ['CycleNetMM','CycleNetQM']:
            return self.train_CycleNetMM_Q(setting)
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            # max_memory = 0
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'CycleNet', 'CycleNetQQ'}):
                            outputs, _ = self.model(batch_x, batch_cycle)[0]
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'GRU'}):
                            outputs, _ = self.model(batch_x)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if any(substr in self.args.model for substr in {'CycleNet', 'CycleNetQQ'}):
                        outputs, _ = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear', 'GRU'}):
                        outputs, _ = self.model(batch_x)
                    
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                # current_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
                # max_memory = max(max_memory, current_memory)

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = 1#self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # print(f"Max Memory (MB): {max_memory}")

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []        
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'CycleNetMM', 'CycleNetQQ', 'CycleNetQM', 'CycleNet'}):
                            outputs, remain = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'GRU'}):
                            outputs, _ = self.model(batch_x)
                        
                else:
                    if any(substr in self.args.model for substr in {'CycleNetMM', 'CycleNetQQ', 'CycleNetQM', 'CycleNet'}):
                        outputs, remain = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear', 'GRU'}):
                        outputs, _ = self.model(batch_x)
                   

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                remain = remain.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                # inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                    np.savetxt(os.path.join(folder_path, str(i) + '.txt'), pd)
                    np.savetxt(os.path.join(folder_path, str(i) + 'true.txt'), gt)

                    try:
                        # # x: (batch_size, seq_len, enc_in), cycle_index: (batch_size,)
                        # print(batch_x.shape, batch_cycle.shape)
                        # Q = self.model.cycleQueue.data.detach().cpu().numpy()  # 一个周期的数据
                        # preds_last = pred[0, :, -1].reshape(-1)  # 展平预测结果
                        # trues_last = true[0, :, -1].reshape(-1)

                        # # 计算需要多少个完整周期
                        # Q_len = Q.shape[0]  # 一个周期的长度
                        # total_len = len(preds_last)
                        # num_cycles = (total_len + Q_len - 1) // Q_len  # 向上取整得到需要的周期数

                        # Q = np.roll(Q, -self.args.seq_len-batch_cycle[0].detach().cpu().numpy(), axis=0)
                        # print(self.args.seq_len, batch_cycle[0].detach().cpu().numpy())

                        # # 将Q重复扩展到足够的长度
                        # Q_repeated = np.tile(Q[:, -1], num_cycles)[:total_len]

                        # # 计算remain
                        # pred_remain = preds_last[-len(Q_repeated):] - Q_repeated
                        # trues_remain = trues_last[-len(Q_repeated):] - Q_repeated

                        # detailed_analysis(trues_remain,pred_remain, Q, trues_last, preds_last, Q_repeated, folder_path)
                        # # 使用最后10个周期的数据                                     
                        # plot_comparison(
                        #     training_data=trues_last[-len(Q_repeated):],  # 原始训练数据
                        #     weights=Q_repeated,                           # 展开的周期性权重
                        #     residual=trues_remain,                         # 残差数据
                        #     save_dir=folder_path
                        # )
                        _time = datetime.now().strftime('%Y%m%d_%H%M%S')
                        plt.plot(remain[0, :, -1], color='blue', label='Remain Data')
                        plt.plot(true[0, :, -1], color='red', label='Origin Data')
                        plt.grid(True)
                        plt.legend()
                        plt.savefig(os.path.join(folder_path, f'remain_{_time}.png'))
                        plt.close()
                    except:
                        print('error')

        if self.args.test_flop:
            test_params_flop(self.model, (batch_x.shape[1], batch_x.shape[2]))
            exit()
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        # inputx = np.concatenate(inputx, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # # x: (batch_size, seq_len, enc_in), cycle_index: (batch_size,)
        # Q = self.model.cycleQueue.data.detach().cpu().numpy()  # 一个周期的数据
        # preds_last = preds[:, :, -1].reshape(-1)[:]  # 展平预测结果
        # trues_last = trues[:, :, -1].reshape(-1)[:]
        # # 计算需要多少个完整周期
        # Q_len = Q.shape[0]  # 一个周期的长度
        # total_len = len(preds_last)
        # num_cycles = (total_len + Q_len - 1) // Q_len  # 向上取整得到需要的周期数

        # Q = np.roll(Q, -96, axis=0)

        # # 将Q重复扩展到足够的长度
        # Q_repeated = np.tile(Q[:, -1], num_cycles)[:total_len]

        # # 计算remain
        # pred_remain = preds_last[-len(Q_repeated):] - Q_repeated
        # trues_remain = trues_last[-len(Q_repeated):] - Q_repeated

        # detailed_analysis(trues_remain,pred_remain, Q, trues_last, preds_last, Q_repeated, folder_path)
        # # 使用最后10个周期的数据                                     
        # plot_comparison(
        #     training_data=trues_last[-len(Q_repeated):],  # 原始训练数据
        #     weights=Q_repeated,                           # 展开的周期性权重
        #     residual=trues_remain,                         # 残差数据
        #     save_dir=folder_path
        # )
        
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
                    batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'CycleNet', 'CycleNetQQ', 'CycleNetMM', 'CycleNetQM'}):
                            outputs, _ = self.model(batch_x, batch_cycle)
                            print('Cyc style!')
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF', 'GRU'}):
                            outputs, _ = self.model(batch_x)
                
                else:
                    if any(substr in self.args.model for substr in {'CycleNet', 'CycleNetQQ', 'CycleNetMM', 'CycleNetQM'}):
                        outputs, _ = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear', 'GRU'}):
                        outputs, _ = self.model(batch_x)
                    
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
    
    def train_CycleNetMM_Q(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        train_epochs = self.args.train_epochs // 2
        learning_rate = self.args.learning_rate

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=train_epochs,
                                            max_lr=learning_rate)

        for epoch in range(train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            # max_memory = 0
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, remain = self.model(batch_x, batch_cycle, 1)[0]
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs, remain = self.model(batch_x, batch_cycle, 1)[0]
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                train_epochs += epoch
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=train_epochs,
                                            max_lr=learning_rate)
        
        for epoch in range(train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            # max_memory = 0
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, remain = self.model(batch_x, batch_cycle, 2)[0]
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs, remain = self.model(batch_x, batch_cycle, 2)[0]
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model