import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.metrics import roc_curve

# ==========================================
# 1. 两步 Youden Index 阈值优化 
# ==========================================
def optimize_risk_thresholds(risk_scores_train, times_train, events_train, eval_time=60):
    """
    在训练集上优化两个阈值，将患者分为低、中、高风险组
    """
    # 定义结局：在 eval_time 之前发生事件为 1，否则为 0
    y_true = ((times_train <= eval_time) & (events_train == 1)).astype(int)
    
    # 第一步：寻找 低风险 vs (中+高)风险 的阈值
    fpr, tpr, thresholds = roc_curve(y_true, risk_scores_train)
    youden_idx = tpr - fpr
    best_idx_1 = np.argmax(youden_idx)
    threshold_1 = thresholds[best_idx_1]
    
    # 第二步：在 (中+高)风险 子集中寻找 中风险 vs 高风险 的阈值
    high_risk_mask = risk_scores_train >= threshold_1
    if np.sum(high_risk_mask) > 10:
        fpr2, tpr2, thresholds2 = roc_curve(y_true[high_risk_mask], risk_scores_train[high_risk_mask])
        youden_idx2 = tpr2 - fpr2
        best_idx_2 = np.argmax(youden_idx2)
        threshold_2 = thresholds2[best_idx_2]
    else:
        threshold_2 = np.percentile(risk_scores_train, 90) # 备选方案
        
    return threshold_1, threshold_2

def stratify_risk(risk_scores, threshold_1, threshold_2):
    """根据阈值将风险分数分为 0:Low, 1:Medium, 2:High"""
    groups = np.zeros(len(risk_scores), dtype=int)
    groups[risk_scores >= threshold_1] = 1
    groups[risk_scores >= threshold_2] = 2
    return groups

# ==========================================
# 2. Kaplan-Meier 绘图与 Log-rank 检验 
# ==========================================
def plot_kaplan_meier(times, events, risk_groups, save_path='km_curve.png'):
    """绘制三组风险的 KM 曲线"""
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8, 6))
    
    colors = ['blue', 'orange', 'red']
    labels = ['Low Risk', 'Intermediate Risk', 'High Risk']
    
    for i in range(3):
        mask = risk_groups == i
        if np.sum(mask) == 0: continue
        kmf.fit(times[mask], events[mask], label=f'{labels[i]} (n={np.sum(mask)})')
        kmf.plot_survival_function(ci_show=True, color=colors[i])
        
    plt.title('Kaplan-Meier Survival Curves by Risk Group')
    plt.xlabel('Time (months)')
    plt.ylabel('Survival Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300)
    print(f" KM 曲线已保存至: {save_path}")
    
    # Log-rank 检验 (Low vs High)
    mask_low = risk_groups == 0
    mask_high = risk_groups == 2
    if np.sum(mask_low) > 0 and np.sum(mask_high) > 0:
        results = logrank_test(times[mask_low], times[mask_high], 
                               event_observed_A=events[mask_low], 
                               event_observed_B=events[mask_high])
        print(f"Log-rank Test (Low vs High) P-value: {results.p_value:.4e}")