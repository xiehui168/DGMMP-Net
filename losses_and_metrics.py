import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score

# ==========================================
# 1. 损失函数 
# ==========================================
class CoxPHLoss(nn.Module):
    """DeepSurv 风格的 Cox 偏对数似然损失"""
    def forward(self, risk_scores, times, events):
        risk = risk_scores.squeeze()
        idx = torch.argsort(times, descending=True)
        risk, events = risk[idx], events[idx]
        
        # 数值稳定的 log-sum-exp 计算
        max_risk, _ = torch.max(risk, dim=0)
        log_sum_exp = torch.log(torch.cumsum(torch.exp(risk - max_risk), dim=0)) + max_risk
        loss = -torch.sum((risk - log_sum_exp) * events) / (torch.sum(events) + 1e-8)
        return loss

class MultiModalInfoNCELoss(nn.Module):
    """跨模态对比学习损失"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, node_features):
        B, V, d = node_features.shape
        z = F.normalize(node_features, dim=-1) 
        loss = 0.0
        for i in range(V):
            for j in range(V):
                if i == j: continue
                logits = torch.mm(z[:, i, :], z[:, j, :].T) / self.temperature
                labels = torch.arange(B).to(logits.device)
                loss += F.cross_entropy(logits, labels)
        return loss / (V * (V - 1))

class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cox_loss = CoxPHLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.contrast_loss = MultiModalInfoNCELoss()
        # 权重 (Section 2.4)
        self.l1, self.l2, self.l3, self.l4 = 1.0, 0.8, 0.5, 0.1

    def forward(self, outputs, times, events, rec_labels, tox_labels):
        loss_surv = self.cox_loss(outputs['risk_mean'], times, events)
        loss_rec = self.ce_loss(outputs['rec_mean'], rec_labels)
        loss_tox = self.bce_loss(outputs['tox_mean'], tox_labels)
        loss_contrast = self.contrast_loss(outputs['node_features'])
        
        total = self.l1*loss_surv + self.l2*loss_rec + self.l3*loss_tox + self.l4*loss_contrast
        return total, {'surv': loss_surv.item(), 'rec': loss_rec.item(), 
                       'tox': loss_tox.item(), 'contrast': loss_contrast.item()}

# ==========================================
# 2. 评估指标 
# ==========================================
def calculate_c_index(risk_scores, times, events):
    """Harrell's C-index"""
    risk_np = risk_scores.cpu().numpy().squeeze()
    time_np = times.cpu().numpy()
    event_np = events.cpu().numpy()
    return concordance_index(time_np, -risk_np, event_np)

def calculate_ece(probs, labels, n_bins=10):
    """Expected Calibration Error (Section 2.6)"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i+1])
        if np.sum(mask) > 0:
            avg_conf = np.mean(probs[mask])
            avg_acc = np.mean(labels[mask])
            ece += (np.sum(mask) / len(probs)) * np.abs(avg_acc - avg_conf)
    return ece

def bootstrap_ci(data, metric_func, n_boot=1000, ci=0.95):
    """计算 95% 置信区间 """
    n = len(data)
    boot_metrics = []
    for _ in range(n_boot):
        sample = data[np.random.randint(0, n, n)]
        boot_metrics.append(metric_func(sample))
    lower = np.percentile(boot_metrics, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_metrics, (1 + ci) / 2 * 100)
    return np.mean(boot_metrics), lower, upper