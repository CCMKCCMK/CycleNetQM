from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Linear, CycleNet, LSTM
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time
from tqdm import tqdm

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Linear': Linear,
            'CycleNet': CycleNet,
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
        vail_bar = tqdm(vali_loader, desc=f'Vali Epoch', position=0)
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast('cuda'):
                        if self.args.model == 'CycleNet':
                            outputs = self.model(batch_x, batch_cycle)
                        elif self.args.model == 'LSTM':
                            outputs = self.model(batch_x)
                        else:  # Linear
                            outputs = self.model(batch_x)
                else:
                    if self.args.model == 'CycleNet':
                        outputs = self.model(batch_x, batch_cycle)
                    elif self.args.model == 'LSTM':
                        outputs = self.model(batch_x)
                    else:  # Linear
                        outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
                vail_bar.set_postfix({'vali_loss': f'{loss.item():.7f}'})

                
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler('cuda')

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate
        )
        
        # Create progress bar for epochs
        epoch_bar = tqdm(range(self.args.train_epochs), desc='Training Epochs', position=0)

        for epoch in epoch_bar:
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            
            # Create progress bar for batches within each epoch
            batch_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.args.train_epochs}', 
                        position=1)
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast('cuda'):
                        if self.args.model == 'CycleNet':
                            outputs = self.model(batch_x, batch_cycle)
                        elif self.args.model == 'LSTM':
                            outputs = self.model(batch_x)
                        else:  # Linear
                            outputs = self.model(batch_x)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                else:
                    if self.args.model == 'CycleNet':
                        outputs = self.model(batch_x, batch_cycle)
                    elif self.args.model == 'LSTM':
                        outputs = self.model(batch_x)
                    else:  # Linear
                        outputs = self.model(batch_x)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)

                train_loss.append(loss.item())

                # Update batch progress bar
                batch_bar.set_postfix({'loss': f'{loss.item():.7f}'})

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
            
            # --------------------  HUGE Time delay???
            train_loss = np.average(train_loss) 
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            # ---------------------
            
            # Update epoch progress bar
            epoch_bar.set_postfix({
                'train_loss': f'{train_loss:.7f}',
                'vali_loss': f'{vali_loss:.7f}',
                'test_loss': f'{test_loss:.7f}',
                'time': f'{time.time() - epoch_time:.2f}s'
            })
            
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

                if self.args.use_amp:
                    with torch.cuda.amp.autocast('cuda'):
                        if self.args.model == 'CycleNet':
                            outputs = self.model(batch_x, batch_cycle)
                        elif self.args.model == 'LSTM':
                            outputs = self.model(batch_x)
                        else:  # Linear
                            outputs = self.model(batch_x)
                else:
                    if self.args.model == 'CycleNet':
                        outputs = self.model(batch_x, batch_cycle)
                    elif self.args.model == 'LSTM':
                        outputs = self.model(batch_x)
                    else:  # Linear
                        outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                    np.savetxt(os.path.join(folder_path, str(i) + '.txt'), pd)
                    np.savetxt(os.path.join(folder_path, str(i) + 'true.txt'), gt)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        
        with open("result.txt", 'a') as f:
            f.write(f"{setting}  \n")
            f.write(f'mse:{mse}, mae:{mae}\n\n')

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

                if self.args.use_amp:
                    with torch.cuda.amp.autocast('cuda'):
                        if self.args.model == 'CycleNet':
                            outputs = self.model(batch_x, batch_cycle)
                        elif self.args.model == 'LSTM':
                            outputs = self.model(batch_x)
                        else:  # Linear
                            outputs = self.model(batch_x)
                else:
                    if self.args.model == 'CycleNet':
                        outputs = self.model(batch_x, batch_cycle)
                    elif self.args.model == 'LSTM':
                        outputs = self.model(batch_x)
                    else:  # Linear
                        outputs = self.model(batch_x)
                        
                pred = outputs.detach().cpu().numpy()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
