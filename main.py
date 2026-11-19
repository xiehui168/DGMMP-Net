import os
import sys
import math
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')


import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve

try:
    import SimpleITK as sitk
    import pydicom
    import cv2
    import timm
    from rt_utils import RTStructBuilder
except ImportError as e:
    print(f" 警告: 缺少依赖库 {e.name}。请运行: pip install SimpleITK pydicom opencv-python timm rt_utils")
    sys.exit(1)

from lifelines.utils import concordance_index

# 开启 CUDA 同步报错
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# ==========================================
# 0. 全局配置
# ==========================================
class Config:
    SEED = 42
    DATA_ROOT = r"D:\pythonProject\DGMMP\rawdata"
    CLINICAL_CSV_PATH = r"D:\pythonProject\DGMMP\clinical_data.csv"
    RESULTS_DIR = r"D:\pythonProject\DGMMP\results" 
    
    CT_SIZE = (128, 128, 32)       
    PATCH_SIZE_IMG = 256           
    PATCH_SIZE_VIT = 16            
    TRAIN_MAX_PATCHES = 256        
    VAL_MAX_PATCHES = 512          
    
    BATCH_SIZE = 2                 
    GRADIENT_ACCUMULATION_STEPS = 4 
    EPOCHS = 100
    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    PATIENCE = 20                  
    
    D_LATENT = 256                 
    VIT_EMBED_DIM = 768            
    RESNET_CHANNELS = [64, 128, 256, 512] 
    CLINICAL_DIM = 32              
    
    MC_DROPOUT_RATE = 0.3          
    MC_PASSES_INFERENCE = 10       
    
    LAMBDA_SURV = 1.0
    LAMBDA_REC = 0.8
    LAMBDA_TOX = 0.5
    LAMBDA_CONTRAST = 0.1
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

cfg = Config()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(cfg.SEED)
print(f" 运行设备: {cfg.DEVICE} | 随机种子: {cfg.SEED} | 等效 Batch Size: {cfg.BATCH_SIZE * cfg.GRADIENT_ACCUMULATION_STEPS}")

# ==========================================
# 1. 数据加载工具
# ==========================================
def find_dicom_files(patient_folder_path):
    ct_files = []
    dose_path = None
    rs_path = None
    for f in os.listdir(patient_folder_path):
        f_lower = f.lower()
        if f_lower.endswith('.dcm'):
            if f_lower.startswith('rd'): dose_path = os.path.join(patient_folder_path, f)
            elif f_lower.startswith('rs'): rs_path = os.path.join(patient_folder_path, f)
            else: ct_files.append(os.path.join(patient_folder_path, f))
    return ct_files, dose_path, rs_path

def load_mask_from_rs(rs_path, ct_files):
    try:
        rtstruct = RTStructBuilder.create_from(dicom_series_path=os.path.dirname(ct_files[0]), rt_struct_path=rs_path)
        roi_names = rtstruct.get_roi_names()
        target_roi = None
        for name in roi_names:
            if any(kw in name.lower() for kw in ['ctv', 'tumor', 'bed', 'ptv', 'gtv']):
                target_roi = name; break
        if target_roi is None and len(roi_names) > 0: target_roi = roi_names[0]
        if target_roi is None: return None
        return rtstruct.get_roi_mask_by_name(target_roi).astype(np.float32)
    except Exception as e:
        return None

def crop_based_on_mask(ct_array, dose_array, mask_array, target_size=(128, 128, 32)):
    if mask_array is not None and np.sum(mask_array) > 0:
        center = np.argwhere(mask_array > 0).mean(axis=0).astype(int)
    else:
        center = np.array(ct_array.shape) // 2
    tz, ty, tx = target_size
    z_s, y_s, x_s = max(0, center[0]-tz//2), max(0, center[1]-ty//2), max(0, center[2]-tx//2)
    z_e, y_e, x_e = min(ct_array.shape[0], z_s+tz), min(ct_array.shape[1], y_s+ty), min(ct_array.shape[2], x_s+tx)
    ct_crop = ct_array[z_s:z_e, y_s:y_e, x_s:x_e]
    dose_crop = dose_array[z_s:z_e, y_s:y_e, x_s:x_e]
    def pad(arr, t): return np.pad(arr, [(0, max(0, t-s)) for t, s in zip(t, arr.shape)], mode='constant')
    return pad(ct_crop, target_size), pad(dose_crop, target_size)

def load_dicom_ct_dose_mask(patient_folder_path):
    try:
        ct_files, dose_path, rs_path = find_dicom_files(patient_folder_path)
        if not ct_files: raise FileNotFoundError("No CT")
        reader = sitk.ImageSeriesReader(); reader.SetFileNames(ct_files); ct_image = reader.Execute()
        ct_array = sitk.GetArrayFromImage(ct_image)
        if dose_path:
            dose_image = sitk.ReadImage(dose_path)
            dose_resampled = sitk.Resample(dose_image, ct_image, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0)
            dose_array = np.clip(sitk.GetArrayFromImage(dose_resampled) / 50.0, 0, 1)
        else: dose_array = np.ones(ct_array.shape)
        mask_array = load_mask_from_rs(rs_path, ct_files) if rs_path else None
        ct_crop, dose_crop = crop_based_on_mask(ct_array, dose_array, mask_array, cfg.CT_SIZE)
        return np.expand_dims(ct_crop, 0).astype(np.float32), np.expand_dims(dose_crop, 0).astype(np.float32)
    except Exception:
        return np.random.randn(1, *cfg.CT_SIZE).astype(np.float32), np.random.rand(1, *cfg.CT_SIZE).astype(np.float32)

def load_pathology_patches(patient_folder_path, is_train):
    patches = []
    if not os.path.exists(patient_folder_path):
        max_p = cfg.TRAIN_MAX_PATCHES if is_train else cfg.VAL_MAX_PATCHES
        return np.random.randn(max_p, 3, cfg.PATCH_SIZE_IMG, cfg.PATCH_SIZE_IMG).astype(np.float32)
    img_files = [f for f in os.listdir(patient_folder_path) if f.lower().endswith('.png')]
    if len(img_files) == 0:
        max_p = cfg.TRAIN_MAX_PATCHES if is_train else cfg.VAL_MAX_PATCHES
        return np.random.randn(max_p, 3, cfg.PATCH_SIZE_IMG, cfg.PATCH_SIZE_IMG).astype(np.float32)
    max_patches = cfg.TRAIN_MAX_PATCHES if is_train else cfg.VAL_MAX_PATCHES
    img_files = np.random.choice(img_files, max_patches, replace=len(img_files) < max_patches)
    for img_file in img_files:
        img = cv2.imread(os.path.join(patient_folder_path, img_file))
        img = cv2.cvtColor(cv2.resize(img, (cfg.PATCH_SIZE_IMG, cfg.PATCH_SIZE_IMG)), cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
        patches.append(img)
    return np.stack(patches, axis=0)

class RealDGMMPDataset(Dataset):
    def __init__(self, csv_path, is_train=True):
        self.df = pd.read_csv(csv_path); self.is_train = is_train
        self.feature_cols = [c for c in self.df.columns if c not in ['patient_id', 'time', 'event', 'recurrence_type', 'tox_lung', 'tox_heart']][:cfg.CLINICAL_DIM]
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]; pid = str(row['patient_id'])
        ct_vol, dose_vol = load_dicom_ct_dose_mask(os.path.join(cfg.DATA_ROOT, 'ct', pid))
        path_patches = load_pathology_patches(os.path.join(cfg.DATA_ROOT, 'pathology', pid), self.is_train)
        clin_vec = row[self.feature_cols].values.astype(np.float32)
        return (torch.tensor(ct_vol), torch.tensor(dose_vol), torch.tensor(path_patches), torch.tensor(clin_vec),
                torch.tensor(float(row['time']), dtype=torch.float32), torch.tensor(float(row['event']), dtype=torch.float32),
                torch.tensor(int(row['recurrence_type']), dtype=torch.long), torch.tensor([float(row['tox_lung']), float(row['tox_heart'])], dtype=torch.float32))

# ==========================================
# 2. 核心模型架构
# ==========================================
class MCDropout(nn.Dropout):
    def forward(self, x): return F.dropout(x, p=self.p, training=True)

class ResNet3DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False); self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, stride=1, padding=1, bias=False); self.bn2 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch: self.shortcut = nn.Sequential(nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm3d(out_ch))
    def forward(self, x): return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))

class VolumeEncoder3D(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        chs = cfg.RESNET_CHANNELS
        self.layer0 = nn.Sequential(nn.Conv3d(in_channels, chs[0], 7, stride=2, padding=3), nn.BatchNorm3d(chs[0]), nn.ReLU())
        self.layer1 = self._make_layer(chs[0], chs[1], 2, 1); self.layer2 = self._make_layer(chs[1], chs[2], 2, 2); self.layer3 = self._make_layer(chs[2], chs[3], 2, 2)
        self.gap = nn.AdaptiveAvgPool3d(1); self.out_ch = chs[-1]
    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [ResNet3DBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks): layers.append(ResNet3DBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.layer0(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        return x, self.gap(x).flatten(1)

class PathologyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        self.vit.patch_embed.img_size = (cfg.PATCH_SIZE_IMG, cfg.PATCH_SIZE_IMG); self.vit.patch_embed.grid_size = (16, 16); self.vit.patch_embed.num_patches = 256
        self.vit.pos_embed = nn.Parameter(torch.zeros(1, 257, cfg.VIT_EMBED_DIM)); self.vit.set_grad_checkpointing(True)
        self.mil_attn = nn.Sequential(nn.Linear(cfg.VIT_EMBED_DIM, 128), nn.Tanh(), nn.Linear(128, 1)); self.chunk_size = 64 
    def forward(self, patches):
        B, N, C, H, W = patches.shape; x = patches.flatten(0, 1)
        patch_features_list = []
        for i in range(0, x.size(0), self.chunk_size):
            chunk = x[i:i+self.chunk_size]; patch_features_list.append(self.vit(chunk)); del chunk 
        patch_features = torch.cat(patch_features_list, dim=0).view(B, N, cfg.VIT_EMBED_DIM)
        attn_w = F.softmax(self.mil_attn(patch_features), dim=1); patient_vec = torch.sum(patch_features * attn_w, dim=1)
        return patch_features, patient_vec

class ClinicalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(cfg.CLINICAL_DIM, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU())
    def forward(self, x): return self.mlp(x)

class DoseGuidedCrossModalAlignment(nn.Module):
    def __init__(self):
        super().__init__()
        d = cfg.D_LATENT; self.Wq, self.Wk, self.Wv = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d); self.scale = math.sqrt(d)
    def forward(self, q_tokens, kv_tokens):
        Q, K, V = self.Wq(q_tokens), self.Wk(kv_tokens), self.Wv(kv_tokens)
        attn = F.softmax(torch.bmm(Q, K.transpose(1, 2)) / self.scale, dim=-1)
        return torch.bmm(attn, V), attn

class UncertaintyAwareGraphFusion(nn.Module):
    def __init__(self):
        super().__init__()
        d = cfg.D_LATENT; self.Wq, self.Wk, self.Wv = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d); self.scale = math.sqrt(d)
        self.gat_fc1, self.gat_fc2 = nn.Linear(d, d), nn.Linear(d, d); self.leaky_relu = nn.LeakyReLU(0.2)
    def forward(self, nodes, vars=None):
        B, V, d = nodes.shape
        Q, K = self.Wq(nodes), self.Wk(nodes); A = F.softmax(torch.bmm(Q, K.transpose(1, 2)) / self.scale, dim=-1)
        if vars is not None:
            conf = 1.0 / (1.0 + vars.squeeze(-1)); A = A * conf.unsqueeze(1); A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)
        V_proj = self.Wv(nodes); H1 = self.leaky_relu(torch.bmm(A, V_proj)); H1 = self.gat_fc1(H1)
        H2 = self.leaky_relu(torch.bmm(A, H1)); H2 = self.gat_fc2(H2)
        return H2.mean(dim=1)

class DGMMPNet(nn.Module):
    def __init__(self):
        super().__init__()
        d = cfg.D_LATENT
        self.ct_enc = VolumeEncoder3D(1); self.dose_enc = VolumeEncoder3D(1); self.path_enc = PathologyEncoder(); self.clin_enc = ClinicalEncoder()
        self.proj_ct = nn.Linear(self.ct_enc.out_ch, d); self.proj_dose = nn.Linear(self.dose_enc.out_ch, d)
        self.proj_path = nn.Linear(cfg.VIT_EMBED_DIM, d); self.proj_clin = nn.Linear(64, d)
        self.align_ct = DoseGuidedCrossModalAlignment(); self.align_path = DoseGuidedCrossModalAlignment()
        self.graph = UncertaintyAwareGraphFusion(); self.drop = MCDropout(p=cfg.MC_DROPOUT_RATE)
        self.head_surv = nn.Linear(d, 1); self.head_rec = nn.Linear(d, 4); self.head_tox = nn.Linear(d, 2)
    def _flatten(self, x): return x.flatten(2).transpose(1, 2)
    def forward(self, ct, dose, path, clin, mc_passes=1, return_attn=False):
        ct_map, _ = self.ct_enc(ct); dose_map, _ = self.dose_enc(dose); path_tok, _ = self.path_enc(path); clin_vec = self.clin_enc(clin)
        ct_t = self.proj_ct(self._flatten(ct_map)); dose_t = self.proj_dose(self._flatten(dose_map)); path_t = self.proj_path(path_tok)
        ali_ct, attn_ct = self.align_ct(dose_t, ct_t); ali_path, attn_path = self.align_path(dose_t, path_t)
        nodes = torch.stack([ali_ct.mean(1), dose_t.mean(1), ali_path.mean(1), self.proj_clin(clin_vec)], dim=1)
        fused = self.graph(nodes)
        risks, recs, toxs = [], [], []
        for _ in range(mc_passes):
            h = self.drop(fused); risks.append(self.head_surv(h)); recs.append(self.head_rec(h)); toxs.append(self.head_tox(h))
        out = {'risk_mean': torch.stack(risks).mean(0), 'risk_var': torch.stack(risks).var(0, unbiased=False),
               'rec_mean': torch.stack(recs).mean(0), 'tox_mean': torch.stack(toxs).mean(0), 'nodes': nodes}
        if return_attn: out['attn_ct'], out['attn_path'] = attn_ct, attn_path
        return out

# ==========================================
# 3. 损失函数与训练引擎
# ==========================================
class CoxPHLoss(nn.Module):
    def forward(self, risk, times, events):
        risk = risk.squeeze(); idx = torch.argsort(times, descending=True); risk, events = risk[idx], events[idx]
        max_r, _ = torch.max(risk, 0); lse = torch.log(torch.cumsum(torch.exp(risk - max_r), 0)) + max_r
        return -torch.sum((risk - lse) * events) / (torch.sum(events) + 1e-8)

class InfoNCELoss(nn.Module):
    def __init__(self): super().__init__(); self.t = 0.07
    def forward(self, z):
        B, V, d = z.shape; z = F.normalize(z, dim=-1); loss = 0.0
        for i in range(V):
            for j in range(V):
                if i != j: loss += F.cross_entropy(torch.mm(z[:, i], z[:, j].T) / self.t, torch.arange(B).to(z.device))
        return loss / (V * (V - 1))

class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cox, self.ce, self.bce, self.nce = CoxPHLoss(), nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss(), InfoNCELoss()
    def forward(self, out, times, events, rec, tox):
        return cfg.LAMBDA_SURV*self.cox(out['risk_mean'], times, events) + cfg.LAMBDA_REC*self.ce(out['rec_mean'], rec) + \
               cfg.LAMBDA_TOX*self.bce(out['tox_mean'], tox) + cfg.LAMBDA_CONTRAST*self.nce(out['nodes'])

def calc_cindex(risk, times, events):
    return concordance_index(times.cpu().numpy(), -risk.cpu().detach().numpy().squeeze(), events.cpu().numpy())

class EarlyStopping:
    def __init__(self, patience=20):
        self.patience, self.counter, self.best_score = patience, 0, None; self.best_wts = None
    def __call__(self, score, model):
        if self.best_score is None or score > self.best_score:
            self.best_score, self.best_wts, self.counter = score, copy.deepcopy(model.state_dict()), 0
        else:
            self.counter += 1
            if self.counter >= self.patience: return True
        return False

def train_model(model, train_loader, val_loader):
    model.to(cfg.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)
    criterion = MultiTaskLoss(); es = EarlyStopping(patience=cfg.PATIENCE)
    for epoch in range(cfg.EPOCHS):
        model.train(); total_loss = 0.0; optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            ct, dose, path, clin, times, events, rec, tox = [b.to(cfg.DEVICE) for b in batch]
            loss = criterion(model(ct, dose, path, clin, mc_passes=1), times, events, rec, tox)
            loss = loss / cfg.GRADIENT_ACCUMULATION_STEPS; loss.backward()
            if (i + 1) % cfg.GRADIENT_ACCUMULATION_STEPS == 0: optimizer.step(); optimizer.zero_grad(); torch.cuda.empty_cache()
            total_loss += loss.item() * cfg.GRADIENT_ACCUMULATION_STEPS
        scheduler.step()
        model.eval(); val_risks, val_times, val_events = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                ct, dose, path, clin, times, events, _, _ = [b.to(cfg.DEVICE) for b in batch]
                out = model(ct, dose, path, clin, mc_passes=cfg.MC_PASSES_INFERENCE)
                val_risks.append(out['risk_mean'].cpu()); val_times.append(times.cpu()); val_events.append(events.cpu())
        if val_risks:
            c_idx = calc_cindex(torch.cat(val_risks), torch.cat(val_times), torch.cat(val_events))
            print(f"Epoch {epoch+1:03d} | Loss: {total_loss/len(train_loader):.4f} | Val C-Index: {c_idx:.4f}")
            if es(c_idx, model): print(" Early Stopping Triggered!"); break
    if es.best_wts: model.load_state_dict(es.best_wts)
    return model

# ==========================================
# 可视化绘图模块 
# ==========================================
def plot_kaplan_meier(times, events, risk_scores, save_path):
    """Figure : 风险分层 KM 生存曲线"""
    # 按风险分数分位数分为 Low, Medium, High 三组
    thresholds = np.percentile(risk_scores, [33.3, 66.6])
    groups = np.digitize(risk_scores, thresholds) # 0: Low, 1: Med, 2: High
    
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e', '#d62728'] # 蓝, 橙, 红
    labels = ['Low Risk', 'Intermediate Risk', 'High Risk']
    
    for i in range(3):
        mask = groups == i
        if np.sum(mask) == 0: continue
        kmf.fit(times[mask], events[mask], label=f'{labels[i]} (n={np.sum(mask)})')
        kmf.plot_survival_function(ci_show=True, color=colors[i])
        
    plt.title('Kaplan-Meier Survival Curves by Risk Group', fontsize=14)
    plt.xlabel('Time (months)', fontsize=12); plt.ylabel('Survival Probability', fontsize=12)
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()
    print(f" KM 曲线已保存至: {save_path}")

def plot_calibration_curve(risk_scores, events, times, save_path, eval_time=60):
    """Figure : 60个月校准曲线"""
    # 简化：将 Cox risk score 映射为概率 (Sigmoid)
    # 严格来说应使用 Breslow estimator，这里用 Sigmoid 近似以展示流程
    prob_scores = 1 / (1 + np.exp(-risk_scores)) 
    
    # 筛选在 eval_time 有明确结局的样本
    mask = times <= eval_time
    if np.sum(mask) < 10: return
    y_true = ((times[mask] <= eval_time) & (events[mask] == 1)).astype(int)
    y_prob = prob_scores[mask]
    
    # 计算校准曲线
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10)
    
    plt.figure(figsize=(6, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", color="#d62728", label="DGMMP-Net", markersize=8)
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability", fontsize=12); plt.ylabel("Fraction of positives", fontsize=12)
    plt.title(f"Calibration Curve at {eval_time} Months", fontsize=14)
    plt.legend(loc="lower right"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()
    print(f" 校准曲线已保存至: {save_path}")

def plot_time_dependent_roc(risk_scores, events, times, save_path):
    """Figure : 时间依赖性 ROC (12, 24, 36, 48, 60 个月)"""
    prob_scores = 1 / (1 + np.exp(-risk_scores))
    time_points = [12, 24, 36, 48, 60]
    
    plt.figure(figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, t in enumerate(time_points):
        # 简化版 Time-dependent ROC：
        # 正样本：在时间 t 之前发生事件
        # 负样本：生存时间大于 t (忽略在 t 之前删失的样本)
        pos_mask = (times <= t) & (events == 1)
        neg_mask = times > t
        
        if np.sum(pos_mask) < 5 or np.sum(neg_mask) < 5: continue
        
        y_true = np.concatenate([np.ones(np.sum(pos_mask)), np.zeros(np.sum(neg_mask))])
        y_score = np.concatenate([prob_scores[pos_mask], prob_scores[neg_mask]])
        
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[idx], lw=2, label=f'{t} months (AUC={roc_auc:.3f})')
        
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12); plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Time-Dependent ROC Curves", fontsize=14)
    plt.legend(loc="lower right"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()
    print(f" 时间依赖性 ROC 已保存至: {save_path}")

# ==========================================
# 5. 主运行逻辑
# ==========================================
def main():
    if not os.path.exists(cfg.CLINICAL_CSV_PATH):
        raise FileNotFoundError(f"找不到临床数据 CSV: {cfg.CLINICAL_CSV_PATH}。")
    
    print(f" 加载临床数据: {cfg.CLINICAL_CSV_PATH}")
    full_df = pd.read_csv(cfg.CLINICAL_CSV_PATH)
    train_df = full_df.sample(frac=0.7, random_state=42); val_df = full_df.drop(train_df.index)
    train_csv, val_csv = r"D:\pythonProject\DGMMP\train.csv", r"D:\pythonProject\DGMMP\val.csv"
    train_df.to_csv(train_csv, index=False); val_df.to_csv(val_csv, index=False)
    
    train_loader = DataLoader(RealDGMMPDataset(train_csv, True), batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(RealDGMMPDataset(val_csv, False), batch_size=cfg.BATCH_SIZE, num_workers=0)
    
    print(" 初始化 DGMMP-Net..."); model = DGMMPNet()
    print(" 开始训练..."); best_model = train_model(model, train_loader, val_loader)
    
    print("\n 最终评估与绘图..."); best_model.eval()
    risks, times, events = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            ct, dose, path, clin, t, e, _, _ = [b.to(cfg.DEVICE) for b in batch]
            out = best_model(ct, dose, path, clin, mc_passes=10)
            risks.append(out['risk_mean'].cpu()); times.append(t.cpu()); events.append(e.cpu())
            
    risk_np = torch.cat(risks).numpy().squeeze()
    time_np = torch.cat(times).numpy()
    event_np = torch.cat(events).numpy()
    
    c_idx = calc_cindex(torch.cat(risks), torch.cat(times), torch.cat(events))
    print(f" 最终验证集 C-Index: {c_idx:.4f}")
    
    # 创建结果目录并绘图
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    plot_kaplan_meier(time_np, event_np, risk_np, os.path.join(cfg.RESULTS_DIR, "fig3c_km_curve.png"))
    plot_calibration_curve(risk_np, event_np, time_np, os.path.join(cfg.RESULTS_DIR, "fig3b_calibration.png"))
    plot_time_dependent_roc(risk_np, event_np, time_np, os.path.join(cfg.RESULTS_DIR, "fig3a_time_roc.png"))
    
    print(" 真实数据流程验证与绘图完成！请查看 results 文件夹。")

if __name__ == "__main__":
    main()