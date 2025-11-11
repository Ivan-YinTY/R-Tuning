from data_provider.data_loader import UnivariateDatasetBenchmark, MultivariateDatasetBenchmark, Global_Temp, Global_Wind, Dataset_ERA5_Pretrain, Dataset_ERA5_Pretrain_Test, UTSD, UTSD_Npy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import numpy as np
import pywt

data_dict = {
    'UnivariateDatasetBenchmark': UnivariateDatasetBenchmark,
    'MultivariateDatasetBenchmark': MultivariateDatasetBenchmark,
    'Global_Temp': Global_Temp,
    'Global_Wind': Global_Wind,
    'Era5_Pretrain': Dataset_ERA5_Pretrain,
    'Era5_Pretrain_Test': Dataset_ERA5_Pretrain_Test,
    'Utsd': UTSD,
    'Utsd_Npy': UTSD_Npy
}


def data_provider(args, flag, model=None):
    Data = data_dict[args.data]

    if flag in ['test', 'val']:
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size

    if flag in ['train', 'val']:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.input_token_len, args.output_token_len],
            nonautoregressive=args.nonautoregressive,
            test_flag=args.test_flag,
            subset_rand_ratio=args.subset_rand_ratio
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.test_seq_len, args.input_token_len, args.test_pred_len],
            nonautoregressive=args.nonautoregressive,
            test_flag=args.test_flag,
            subset_rand_ratio=args.subset_rand_ratio
        )

    # 合成数据集逻辑
    if flag == 'train' and args.use_synthetic and model is not None:
        print("Generating synthetic data...")
        synthetic_dataset = generate_synthetic_data(args, model, data_set)
        data_set = torch.utils.data.ConcatDataset([data_set, synthetic_dataset])

    print(flag, len(data_set))
    if args.ddp:
        train_datasampler = DistributedSampler(data_set, shuffle=shuffle_flag)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            sampler=train_datasampler,
            num_workers=args.num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=drop_last,
        )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=drop_last
        )
    return data_set, data_loader


def wavelet_denoise_tensor(data, wavelet='db4', level=1, detail_scale=1):
    # data: [N, seq_len, ...] or [N, seq_len]
    data_np = data.numpy()
    if data_np.ndim == 3:
        for i in range(data_np.shape[0]):
            for j in range(data_np.shape[2]):
                coeffs = pywt.swt(data_np[i, :, j], wavelet, level=level)
                # coeffs: list of (cA, cD) tuples
                new_coeffs = []
                for l, (cA, cD) in enumerate(coeffs):
                    if l == level - 1:
                        cD = cD * detail_scale  # 只抑制最高频细节
                    new_coeffs.append((cA, cD))
                data_np[i, :, j] = pywt.iswt(new_coeffs, wavelet)[:data_np.shape[1]]
    elif data_np.ndim == 2:
        for i in range(data_np.shape[0]):
            coeffs = pywt.swt(data_np[i, :], wavelet, level=level)
            new_coeffs = []
            for l, (cA, cD) in enumerate(coeffs):
                if l == level - 1:
                    cD = cD * detail_scale
                new_coeffs.append((cA, cD))
            data_np[i, :] = pywt.iswt(new_coeffs, wavelet)[:data_np.shape[1]]
    return torch.from_numpy(data_np).type_as(data)


def generate_synthetic_data(args, model, data_set):
    model.eval()
    device = next(model.parameters()).device
    synthetic_x = []
    synthetic_y = []
    synthetic_x_mark = []
    synthetic_y_mark = []
    with torch.no_grad():
        for _ in range(args.synthetic_size):
            idx = np.random.randint(0, len(data_set))
            x, y, x_mark, y_mark = data_set[idx]
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.float32)
            if not isinstance(x_mark, torch.Tensor):
                x_mark = torch.tensor(x_mark, dtype=torch.float32)
            if not isinstance(y_mark, torch.Tensor):
                y_mark = torch.tensor(y_mark, dtype=torch.float32)
            rand_x = torch.randn_like(x)
            rand_x_mark = torch.zeros_like(x_mark)
            rand_y_mark = torch.zeros_like(y_mark)
            pred_y = model(
                rand_x.unsqueeze(0).to(device),
                rand_x_mark.unsqueeze(0).to(device),
                rand_y_mark.unsqueeze(0).to(device)
            )
            pred_y = pred_y.squeeze(0).detach().cpu()
            pred_y = pred_y.float()
            synthetic_x.append(rand_x.float())
            synthetic_y.append(pred_y)
            synthetic_x_mark.append(rand_x_mark.float())
            synthetic_y_mark.append(rand_y_mark.float())
    synthetic_x = torch.stack(synthetic_x)
    synthetic_y = torch.stack(synthetic_y)
    synthetic_x_mark = torch.stack(synthetic_x_mark)
    synthetic_y_mark = torch.stack(synthetic_y_mark)

    # 小波去高频
    filtered_synthetic_x = wavelet_denoise_tensor(synthetic_x, wavelet='db4', level=1, detail_scale=0.7)
    filtered_synthetic_y = wavelet_denoise_tensor(synthetic_y, wavelet='db4', level=1, detail_scale=0.7)

    # 拼接原始数据
    if isinstance(data_set, torch.utils.data.TensorDataset):
        orig_x = data_set.tensors[0]
        orig_y = data_set.tensors[1]
        orig_x_mark = data_set.tensors[2]
        orig_y_mark = data_set.tensors[3]
    else:
        orig_x, orig_y, orig_x_mark, orig_y_mark = [], [], [], []
        for i in range(len(data_set)):
            x, y, x_mark, y_mark = data_set[i]
            orig_x.append(x)
            orig_y.append(y)
            orig_x_mark.append(x_mark)
            orig_y_mark.append(y_mark)
        orig_x = torch.stack(orig_x)
        orig_y = torch.stack(orig_y)
        orig_x_mark = torch.stack(orig_x_mark)
        orig_y_mark = torch.stack(orig_y_mark)

    # 拼接：原始 + 未滤波合成 + 小波去高频合成
    all_x = torch.cat([orig_x, synthetic_x, filtered_synthetic_x], dim=0)
    all_y = torch.cat([orig_y, synthetic_y, filtered_synthetic_y], dim=0)
    all_x_mark = torch.cat([orig_x_mark, synthetic_x_mark, synthetic_x_mark], dim=0)
    all_y_mark = torch.cat([orig_y_mark, synthetic_y_mark, synthetic_y_mark], dim=0)

    return torch.utils.data.TensorDataset(all_x, all_y, all_x_mark, all_y_mark)
