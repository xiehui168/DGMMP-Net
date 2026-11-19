import numpy as np
import torch

# ==========================================
# 1. 推理与不确定性量化 (Section 3.5, Fig S1)
# ==========================================
@torch.no_grad()
def inference_with_uncertainty(model, dataloader, device, mc_iterations=10):
    model.eval()
    all_risk_means, all_risk_vars = [], []
    all_rec_probs, all_tox_probs = [], []
    
    for batch in dataloader:
        ct, dose, path, clin = [b.to(device) for b in batch[:4]]
        outputs = model(ct, dose, path, clin, mc_passes=mc_iterations)
        
        all_risk_means.append(outputs['risk_mean'].cpu())
        all_risk_vars.append(outputs['risk_var'].cpu())
        all_rec_probs.append(torch.softmax(outputs['rec_mean'], dim=1).cpu())
        all_tox_probs.append(torch.sigmoid(outputs['tox_mean']).cpu())
        
    return {
        'risk_mean': torch.cat(all_risk_means).numpy(),
        'risk_var': torch.cat(all_risk_vars).numpy(),
        'rec_probs': torch.cat(all_rec_probs).numpy(),
        'tox_probs': torch.cat(all_tox_probs).numpy()
    }

# ==========================================
# 2. 风险分层 (Youden Index 优化)
# ==========================================
def stratify_risk(risk_scores, times, events):
    """基于训练集优化的阈值进行低/中/高风险分层"""
    # 实际应用中需使用 lifelines 或 sksurv 寻找最佳截断值
    # 这里使用简单的三分位数作为 Mock 示例
    thresholds = np.percentile(risk_scores, [33.3, 66.6])
    groups = np.digitize(risk_scores, thresholds) # 0:Low, 1:Med, 2:High
    return groups

# ==========================================
# 3. 可解释性: 剂量引导注意力提取 (Section 3.5, Fig S3)
# ==========================================
@torch.no_grad()
def extract_dose_attention_maps(model, dataloader, device):
    """提取 Dose-guided cross-modal attention weights"""
    model.eval()
    attn_maps_ct, attn_maps_path = [], []
    
    for batch in dataloader:
        ct, dose, path, clin = [b.to(device) for b in batch[:4]]
        # return_attn=True 返回注意力矩阵
        outputs = model(ct, dose, path, clin, mc_passes=1, return_attn=True)
        attn_maps_ct.append(outputs['attn_ct'].cpu().numpy())
        attn_maps_path.append(outputs['attn_path'].cpu().numpy())
        
    return np.concatenate(attn_maps_ct, axis=0), np.concatenate(attn_maps_path, axis=0)

# ==========================================
# 4. 可解释性: 排列特征扰动 (Section 3.5, Table 4)
# ==========================================
def permutation_feature_ablation(model, dataloader, device, modality='dose'):
    """评估单一模态的重要性 (Delta C-index)"""
    model.eval()
    baseline_risks, baseline_times, baseline_events = [], [], []
    
    # 1. 获取 Baseline 预测
    with torch.no_grad():
        for batch in dataloader:
            ct, dose, path, clin, times, events = [b.to(device) for b in batch[:6]]
            out = model(ct, dose, path, clin, mc_passes=10)
            baseline_risks.append(out['risk_mean'].cpu())
            baseline_times.append(times.cpu())
            baseline_events.append(events.cpu())
            
    baseline_c = calculate_c_index(torch.cat(baseline_risks), torch.cat(baseline_times), torch.cat(baseline_events))
    
    # 2. 扰动指定模态
    perturbed_risks = []
    with torch.no_grad():
        for batch in dataloader:
            ct, dose, path, clin, _, _ = [b.to(device) for b in batch[:6]]
            if modality == 'dose':
                dose = torch.randn_like(dose) # 随机打乱 Dose
            elif modality == 'ct':
                ct = torch.randn_like(ct)
            # ... 其他模态同理
            
            out = model(ct, dose, path, clin, mc_passes=10)
            perturbed_risks.append(out['risk_mean'].cpu())
            
    perturbed_c = calculate_c_index(torch.cat(perturbed_risks), torch.cat(baseline_times), torch.cat(baseline_events))
    
    return baseline_c, perturbed_c, baseline_c - perturbed_c # 返回 Delta C-index