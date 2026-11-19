import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import nibabel as nib # 用于读取 .nii.gz，如果存的是 .npy 则直接用 np.load

class RealDGMMPDataset(Dataset):
    def __init__(self, csv_path, data_root_dir, is_train=True, max_patches=1000):
        """
        csv_path: 包含 patient_id, time, event, recurrence, tox_lung, tox_heart 等列的 CSV 文件
        data_root_dir: 存放 ct/, dose/, path/, clinical/ 文件夹的根目录
        """
        self.df = pd.read_csv(csv_path)
        self.data_root = data_root_dir
        self.is_train = is_train
        self.max_patches = max_patches

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = str(row['patient_id'])
        
        # 1. 加载 3D CT (假设已预处理为 128x128x32 的 .npy 或 .nii.gz)
        ct_path = os.path.join(self.data_root, 'ct', f'{pid}.npy')
        ct_vol = np.load(ct_path) # shape: (1, 128, 128, 32)
        ct_tensor = torch.tensor(ct_vol, dtype=torch.float32)
        
        # 2. 加载 3D Dose (已除以 50Gy 归一化到 [0, 1])
        dose_path = os.path.join(self.data_root, 'dose', f'{pid}.npy')
        dose_vol = np.load(dose_path) # shape: (1, 128, 128, 32)
        dose_tensor = torch.tensor(dose_vol, dtype=torch.float32)
        
        # 3. 加载病理 Patches
        path_path = os.path.join(self.data_root, 'path', f'{pid}.npy')
        patches = np.load(path_path) # shape: (N, 3, 256, 256)
        
        # 训练时随机采样，验证时取前 max_patches 个 (Supp A.2)
        if self.is_train:
            if len(patches) > self.max_patches:
                indices = np.random.choice(len(patches), self.max_patches, replace=False)
                patches = patches[indices]
        else:
            patches = patches[:self.max_patches]
            
        # 补齐或截断到固定的 max_patches (方便 DataLoader collate)
        if len(patches) < self.max_patches:
            pad_len = self.max_patches - len(patches)
            patches = np.pad(patches, ((0, pad_len), (0, 0), (0, 0), (0, 0)), mode='constant')
            
        path_tensor = torch.tensor(patches, dtype=torch.float32)
        
        # 4. 加载临床数据 (假设已做 Z-score 标准化和 One-hot 编码)
        clin_features = row[['age', 't_stage', 'n_stage', 'ki67', 'er', 'pr', 'her2']].values.astype(np.float32)
        clin_tensor = torch.tensor(clin_features, dtype=torch.float32)
        
        # 5. 标签
        time_tensor = torch.tensor(float(row['time']), dtype=torch.float32)
        event_tensor = torch.tensor(float(row['event']), dtype=torch.float32)
        rec_tensor = torch.tensor(int(row['recurrence_type']), dtype=torch.long)
        tox_tensor = torch.tensor([float(row['tox_lung']), float(row['tox_heart'])], dtype=torch.float32)
        
        return (ct_tensor, dose_tensor, path_tensor, clin_tensor, 
                time_tensor, event_tensor, rec_tensor, tox_tensor)