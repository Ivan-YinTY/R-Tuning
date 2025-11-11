import os
import time
import warnings
import torch
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
warnings.filterwarnings('ignore')

def safe_print(*args, **kwargs):
    import argparse
    # 这里假设 args 是全局或可访问的
    if not hasattr(args, "ddp") or not args.ddp or (hasattr(args, "local_rank") and args.local_rank == 0):
        print(*args, **kwargs)

class Exp_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Forecast, self).__init__(args)
        self.model_core = self.model.module if hasattr(self.model, "module") else self.model

        
    # def _build_model(self):
    #     if self.args.ddp:
    #         self.device = torch.device('cuda:{}'.format(self.args.local_rank))
    #     else:
    #         # for methods that do not use ddp (e.g. finetuning-based LLM4TS models)
    #         self.device = self.args.gpu
        
    #     model = self.model_dict[self.args.model].Model(self.args)
        
    #     if self.args.ddp:
    #         model = DDP(model.cuda(), device_ids=[self.args.local_rank])
    #     elif self.args.dp:
    #         model = DataParallel(model, device_ids=self.args.device_ids).to(self.device)
    #     else:
    #         self.device = self.args.gpu
    #         model = model.to(self.device)
            
    #     if self.args.adaptation:
    #         model.load_state_dict(torch.load(self.args.pretrain_model_path))
    #     return model

    def _build_model(self):
        if self.args.ddp:
            torch.cuda.set_device(self.args.local_rank)
            self.device = torch.device("cuda", self.args.local_rank)
        else:
            self.device = self.args.gpu

        model = self.model_dict[self.args.model].Model(self.args).to(self.device)

        # =========================
        # 加载预训练权重（仅在 local_rank == 0）
        # =========================
        if self.args.adaptation:
            if not self.args.ddp or self.args.local_rank == 0:
                ckpt = torch.load(self.args.pretrain_model_path, map_location="cpu")
                model_keys = list(model.state_dict().keys())
                ckpt_keys = list(ckpt.keys())

                if model_keys[0].startswith("module") and not ckpt_keys[0].startswith("module"):
                    ckpt = {"module." + k: v for k, v in ckpt.items()}
                elif not model_keys[0].startswith("module") and ckpt_keys[0].startswith("module"):
                    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}

                model.load_state_dict(ckpt)
                print(f"[Rank {self.args.local_rank}] Loaded pretrained weights")

            # 等待0号GPU加载完再同步权重给其他GPU
            if self.args.ddp:
                torch.distributed.barrier()
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # =========================
        # DDP 封装
        # =========================
        if self.args.ddp:
            model = DDP(model.to(self.device), device_ids=[self.args.local_rank])
        elif self.args.dp:
            model = DataParallel(model, device_ids=self.args.device_ids).to(self.device)
        else:
            model = model.to(self.device)

        return model

    def _get_data(self, flag):
        # 训练集时传递模型用于生成合成数据
        if flag == 'train' and self.args.use_synthetic:
            return data_provider(self.args, flag, model=self.model_core)
        else:
            return data_provider(self.args, flag)

    def _select_optimizer(self):
        p_list = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            else:
                p_list.append(p)
        model_optim = optim.Adam([{'params': p_list}], lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
            print('next learning rate is {}'.format(self.args.learning_rate))
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def _evaluate(self, test_loader, setting_suffix):      
        preds = []
        trues = []
        folder_path = f'./test_results/{setting_suffix}/'
        os.makedirs(folder_path, exist_ok=True)
        time_now = time.time()

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                inference_steps = self.args.test_pred_len // self.args.output_token_len
                if self.args.test_pred_len % self.args.output_token_len != 0:
                    inference_steps += 1
                pred_y = []
                for j in range(inference_steps):
                    if len(pred_y) != 0:
                        batch_x = torch.cat([batch_x[:, self.args.input_token_len:, :], pred_y[-1]], dim=1)
                    outputs = self.model(batch_x, batch_x_mark, batch_y_mark)
                    pred_y.append(outputs[:, -self.args.output_token_len:, :])
                pred_y = torch.cat(pred_y, dim=1)
                if self.args.test_pred_len % self.args.output_token_len != 0:
                    dis = self.args.test_pred_len % self.args.output_token_len
                    pred_y = pred_y[:, :-self.args.output_token_len + dis, :]
                batch_y = batch_y[:, -self.args.test_pred_len:, :].to(self.device)

                preds.append(pred_y.detach().cpu())
                trues.append(batch_y.detach().cpu())

        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        if self.args.covariate:
            preds = preds[:, :, -1]
            trues = trues[:, :, -1]
        mae, mse, rmse, mape, mspe, smape = metric(preds, trues)

        result = {
            'setting': setting_suffix,
            'mae': mae, 'mse': mse, 'rmse': rmse,
            'mape': mape, 'mspe': mspe, 'smape': smape
        }

        print(f"\n========== [Continual Learning Evaluation: {setting_suffix}] ==========")
        print(f" MSE   : {mse:.5f}")
        print(f" MAE   : {mae:.5f}")
        print(f" RMSE  : {rmse:.5f}")
        print(f" MAPE  : {mape:.5f}")
        print(f" MSPE  : {mspe:.5f}")
        print(f" SMAPE : {smape:.5f}")
        print("=" * 55)

        return result
    
    def _print_compare(self, name, pre_result, post_result):
        print(f"\n==== [{name}] Pre- vs Post-Finetune Comparison ====")
        for k in ['mse', 'mae', 'rmse', 'mape', 'mspe', 'smape']:
            pre, post = pre_result[k], post_result[k]
            diff = (pre - post) / (pre + 1e-8) * 100
            print(f"{k.upper():<6} | Before: {pre:.5f} → After: {post:.5f} | Δ: {diff:+.2f}%")
    
    def _get_loader_from_path(self, root_path, data_path):
        old_root, old_data = self.args.root_path, self.args.data_path
        self.args.root_path, self.args.data_path = root_path, data_path
        _, loader = self._get_data('test')
        self.args.root_path, self.args.data_path = old_root, old_data
        return loader

    def continual_learning_protocol(self, setting):
        print("\n=== [Continual Learning] Step 1: Load Pretrained Model ===")
        self.model.load_state_dict(torch.load(self.args.pretrain_model_path), strict=False)

        # --- Step 2: Pre-finetune evaluation ---
        old_loader = self._get_loader_from_path(self.args.origin_root_path, self.args.origin_data_path)
        pre_old = self._evaluate(old_loader, setting + '_pre_origin')
        new_loader = self._get_data('test')[1]
        pre_new = self._evaluate(new_loader, setting + '_pre_new')

        # --- Step 3: Finetuning ---
        print("\n=== [Continual Learning] Step 3: Finetuning ===")
        self.train(setting)

        # --- Step 4: Post-finetune evaluation ---
        print("\n=== [Continual Learning] Step 4: Evaluation After Finetuning ===")
        self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')), strict=False)
        post_new = self._evaluate(new_loader, setting + '_post_new')
        post_old = self._evaluate(old_loader, setting + '_post_origin')

        # --- Step 5: Compare and report ---
        print("\n=== [Continual Learning Result Summary] ===")
        def fmt(m): return f"{m:.4f}"
        for k in ['mse', 'mae', 'rmse', 'mape', 'mspe', 'smape']:
            print(f"{k.upper():<6} | "
                f"Old {fmt(pre_old[k])} → {fmt(post_old[k])} ({((pre_old[k]-post_old[k])/pre_old[k]*100):+.2f}%) | "
                f"New {fmt(pre_new[k])} → {fmt(post_new[k])} ({((pre_new[k]-post_new[k])/pre_new[k]*100):+.2f}%)")

    def test_on_original_data(self, setting, test=0):
        origin_root = self.args.root_path
        origin_file = self.args.data_path

        self.args.root_path = self.args.origin_root_path
        self.args.data_path = self.args.origin_data_path

        data, loader = self._get_data(flag='test')

        self.args.root_path = origin_root
        self.args.data_path = origin_file

        print("\n>>>>>>>>>> Testing on ORIGINAL DATA (Pre-task) <<<<<<<<<<")
        return self._evaluate(loader, setting + '_origin')


    def vali(self, vali_data, vali_loader, criterion, is_test=False):
        total_loss = []
        total_count = []
        time_now = time.time()
        test_steps = len(vali_loader)
        iter_count = 0
        
        self.model.eval()    
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                outputs = self.model(batch_x, batch_x_mark, batch_y_mark)
                if is_test or self.args.nonautoregressive:
                        outputs = outputs[:, -self.args.output_token_len:, :]
                        batch_y = batch_y[:, -self.args.output_token_len:, :].to(self.device)
                else:
                    outputs = outputs[:, :, :]
                    batch_y = batch_y[:, :, :].to(self.device)

                if self.args.covariate:
                    if self.args.last_token:
                        outputs = outputs[:, -self.args.output_token_len:, -1]
                        batch_y = batch_y[:, -self.args.output_token_len:, -1]
                    else:
                        outputs = outputs[:, :, -1]
                        batch_y = batch_y[:, :, -1]
                loss = criterion(outputs, batch_y)

                loss = loss.detach().cpu()
                total_loss.append(loss)
                total_count.append(batch_x.shape[0])
                if (i + 1) % 100 == 0:
                    if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (test_steps - i)
                        print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(i + 1, speed, left_time))
                        iter_count = 0
                        time_now = time.time()
        if self.args.ddp:
            total_loss = torch.tensor(np.average(total_loss, weights=total_count)).to(self.device)
            dist.barrier()
            dist.reduce(total_loss, dst=0, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item() / dist.get_world_size()
        else:
            total_loss = np.average(total_loss, weights=total_count)
            
        if self.args.model == 'gpt4ts':
            # GPT4TS just requires to train partial layers
            # self.model.in_layer.train()
            self.model.module.in_layer.train()
            self.model.module.out_layer.train()
        else: 
            self.model.train()
            
        return total_loss

    def train(self, setting):
        # 加载旧任务模型并冻结权重
        old_model = None
        if self.args.use_lwf:
            safe_print("Loading teacher model for Learning Without Forgetting...")
            old_model = self._build_model()
            old_model.eval()
            for param in old_model.parameters():
                param.requires_grad = False
            old_model.to(self.device)

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
            if not os.path.exists(path):
                os.makedirs(path)

        # === Evaluation BEFORE Finetuning ===
        if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
            print("\n=== [Continual Learning] Pre-Finetuning Evaluation ===")
            if self.args.origin_root_path and self.args.origin_data_path:
                ori_loader = self._get_loader_from_path(self.args.origin_root_path, self.args.origin_data_path)
                self.pre_old_result = self._evaluate(ori_loader, setting + '_pre_origin')
            self.pre_new_result = self._evaluate(test_loader, setting + '_pre_new')

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.args, verbose=True)
        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)
        criterion = self._select_criterion()
        distillation_criterion = nn.KLDivLoss(reduction='batchmean')  # 知识蒸馏损失

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, batch_y_mark)

                if self.args.use_lwf:
                    with torch.no_grad():
                        old_outputs = old_model(batch_x, batch_x_mark, batch_y_mark)
                    task_loss = criterion(outputs, batch_y)
                    distillation_loss = distillation_criterion(
                        torch.log_softmax(outputs / self.args.temperature, dim=-1),
                        torch.softmax(old_outputs / self.args.temperature, dim=-1)
                    )
                    total_loss = task_loss + self.args.distillation_lambda * distillation_loss
                else:
                    total_loss = criterion(outputs, batch_y)

                total_loss.backward()
                model_optim.step()

                if (i + 1) % 100 == 0:
                    if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, total_loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

            if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            vali_loss = self.vali(vali_data, vali_loader, criterion, is_test=self.args.valid_last)
            test_loss = self.vali(test_data, test_loader, criterion, is_test=True)
            if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                print("Epoch: {}, Steps: {} | Vali Loss: {:.7f} Test Loss: {:.7f}".format(
                    epoch + 1, train_steps, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if self.args.cosine:
                scheduler.step()
                if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                    print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            if self.args.ddp:
                train_loader.sampler.set_epoch(epoch + 1)

        best_model_path = path + '/' + 'checkpoint.pth'
        if self.args.ddp:
            dist.barrier()
            self.model.load_state_dict(torch.load(best_model_path), strict=False)
        else:
            self.model.load_state_dict(torch.load(best_model_path), strict=False)

        if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
            print('[Train Done] Now evaluating post-finetuning performance...')
            if self.args.origin_root_path and self.args.origin_data_path:
                ori_loader = self._get_loader_from_path(self.args.origin_root_path, self.args.origin_data_path)
                post_old_result = self._evaluate(ori_loader, setting + '_post_origin')
                self._print_compare('ORIGIN', self.pre_old_result, post_old_result)
            post_new_result = self._evaluate(test_loader, setting + '_post_new')
            self._print_compare('NEW', self.pre_new_result, post_new_result)

        # DDP下所有进程最后同步并销毁
        if self.args.ddp:
            dist.barrier()
            dist.destroy_process_group()
            import sys
            sys.exit(0)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        print("info:", self.args.test_seq_len, self.args.input_token_len, self.args.output_token_len, self.args.test_pred_len)
        if test:
            print('loading model')
            setting = self.args.test_dir
            best_model_path = self.args.test_file_name
            print("loading model from {}".format(os.path.join(self.args.checkpoints, setting, best_model_path)))
            checkpoint = torch.load(os.path.join(self.args.checkpoints, setting, best_model_path))
            for name, param in self.model.named_parameters():
                if not param.requires_grad and name not in checkpoint:
                    checkpoint[name] = param
            self.model.load_state_dict(checkpoint)
            
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        time_now = time.time()
        test_steps = len(test_loader)
        iter_count = 0
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                inference_steps = self.args.test_pred_len // self.args.output_token_len
                dis = self.args.test_pred_len - inference_steps * self.args.output_token_len
                if dis != 0:
                    inference_steps += 1
                pred_y = []
                for j in range(inference_steps):  
                    if len(pred_y) != 0:
                        batch_x = torch.cat([batch_x[:, self.args.input_token_len:, :], pred_y[-1]], dim=1)
                    outputs = self.model(batch_x, batch_x_mark, batch_y_mark)
                    pred_y.append(outputs[:, -self.args.output_token_len:, :])
                pred_y = torch.cat(pred_y, dim=1)
                if dis != 0:
                    pred_y = pred_y[:, :-self.args.output_token_len+dis, :]
                batch_y = batch_y[:, -self.args.test_pred_len:, :].to(self.device)
                
                outputs = pred_y.detach().cpu()
                batch_y = batch_y.detach().cpu()
                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if (i + 1) % 100 == 0:
                    if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (test_steps - i)
                        print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(i + 1, speed, left_time))
                        iter_count = 0
                        time_now = time.time()
                # if self.args.visualize and i % 2 == 0:
                #     dir_path = folder_path + f'{self.args.test_pred_len}/'
                #     if not os.path.exists(dir_path):
                #         os.makedirs(dir_path)
                #     gt = np.array(true[0, :, -1])
                #     pd = np.array(pred[0, :, -1])
                #     visual(gt, pd, os.path.join(dir_path, f'{i}.pdf'))

                rank = dist.get_rank() if dist.is_initialized() else 0
                if self.args.visualize and rank == 0 and i % 2 == 0:
                    dir_path = folder_path + f'{self.args.test_pred_len}/'
                    os.makedirs(dir_path, exist_ok=True)
                    gt = np.array(true[0, :, -1])
                    pd = np.array(pred[0, :, -1])
                    file_name = os.path.join(dir_path, f'{i}_{int(time.time())}.pdf')
                    visual(gt, pd, name=file_name, rank=rank)

        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        print('preds shape:', preds.shape)
        print('trues shape:', trues.shape)
        if self.args.covariate:
            preds = preds[:, :, -1]
            trues = trues[:, :, -1]
        mae, mse, rmse, mape, mspe, smape = metric(preds, trues)
        # print('mse:{}, mae:{}'.format(mse, mae))
        print(f"MSE: {mse:.5f}, MAE: {mae:.5f}, RMSE: {rmse:.5f}, MAPE: {mape:.5f}, MSPE: {mspe:.5f}, SMAPE: {smape:.5f}")
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write(f"MSE: {mse:.5f}, MAE: {mae:.5f}, RMSE: {rmse:.5f}, MAPE: {mape:.5f}, MSPE: {mspe:.5f}, SMAPE: {smape:.5f}")
        f.write('\n')
        f.write('\n')
        f.close()
        return
