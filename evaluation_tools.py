import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
from sklearn.metrics import roc_auc_score, brier_score_loss
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. Bootstrap 95% Confidence Intervals 
# ==========================================
def bootstrap_cindex_ci(risk_scores, times, events, n_bootstraps=1000, confidence_level=0.95):
    """计算 C-index 的 95% Bootstrap 置信区间"""
    rng = np.random.RandomState(42)
    boot_cindices = []
    n = len(risk_scores)
    
    for _ in range(n_bootstraps):
        indices = rng.randint(0, n, n)
        if len(np.unique(events[indices])) < 2: continue
        c_idx = concordance_index(times[indices], -risk_scores[indices], events[indices])
        boot_cindices.append(c_idx)
        
    boot_cindices = np.array(boot_cindices)
    median = np.median(boot_cindices)
    lower = np.percentile(boot_cindices, (1 - confidence_level) / 2 * 100)
    upper = np.percentile(boot_cindices, (1 + confidence_level) / 2 * 100)
    return median, lower, upper

# ==========================================
# 2. Time-Dependent AUC 
# ==========================================
def time_dependent_auc(risk_scores, times, events, eval_time=60):
    """
    计算特定时间点 的 AUC
    注意：这里使用简化的累积/动态 AUC 逻辑。
    严格复现建议使用 scikit-survival 的 CumulativeDynamicAUC
    """
    # 筛选在 eval_time 之前发生事件或删失时间大于 eval_time 的样本
    mask = (times <= eval_time) & (events == 1) | (times > eval_time)
    if np.sum(mask) < 10: return 0.5
    
    y_true = (times[mask] <= eval_time) & (events[mask] == 1)
    y_score = risk_scores[mask]
    
    if len(np.unique(y_true)) < 2: return 0.5
    return roc_auc_score(y_true, y_score)

# ==========================================
# 3. Calibration Metrics: Brier Score & ECE (Table 4, Figure 3B)
# ==========================================
def calculate_brier_score(probabilities, times, events, eval_time=60):
    """计算特定时间点的 Brier Score"""
    # 简化版：仅考虑在 eval_time 有明确结局的样本
    mask = times <= eval_time
    if np.sum(mask) == 0: return 0.0
    y_true = (times[mask] <= eval_time) & (events[mask] == 1)
    return brier_score_loss(y_true.astype(int), probabilities[mask])

def calculate_ece(probabilities, labels, n_bins=10):
    """Expected Calibration Error (ECE)"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probabilities >= bin_boundaries[i]) & (probabilities < bin_boundaries[i+1])
        if np.sum(mask) > 0:
            avg_conf = np.mean(probabilities[mask])
            avg_acc = np.mean(labels[mask])
            ece += (np.sum(mask) / len(probabilities)) * np.abs(avg_acc - avg_conf)
    return ece