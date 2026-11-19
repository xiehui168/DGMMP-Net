import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.vision_transformer import VisionTransformer

# ==========================================
# 1. 基础模块: MC Dropout 
# ==========================================
class MCDropout(nn.Dropout):
    """强制在 eval 模式下也保持 Dropout 激活，用于推理时计算不确定性"""
    def forward(self, x):
        return F.dropout(x, p=self.p, training=True)

# ==========================================
# 2. 模态特异性编码器 (Supp A.2)
# ==========================================
class ResNet3DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)

class VolumeEncoder3D(nn.Module):
    """CT 和 Dose 的 3D ResNet-18 编码器)"""
    def __init__(self, in_channels=1):
        super().__init__()
        self.layer0 = nn.Sequential(nn.Conv3d(in_channels, 64, 7, stride=2, padding=3), nn.BatchNorm3d(64), nn.ReLU())
        self.layer1 = self._make_layer(64, 64, 2, 1)
        self.layer2 = self._make_layer(64, 128, 2, 2)
        self.layer3 = self._make_layer(128, 256, 2, 2)
        self.layer4 = self._make_layer(256, 512, 2, 2)
        self.gap = nn.AdaptiveAvgPool3d(1)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [ResNet3DBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks): layers.append(ResNet3DBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat_map = self.layer4(x) # (B, 512, D, H, W)
        global_vec = self.gap(feat_map).flatten(1) # (B, 512)
        return feat_map, global_vec

class PathologyEncoder(nn.Module):
    """病理 WSI 编码器: ViT-Base/16 + Attention MIL (Supp A.2)"""
    def __init__(self, img_size=256, patch_size=16, embed_dim=768):
        super().__init__()
        self.vit = VisionTransformer(img_size=img_size, patch_size=patch_size, 
                                     embed_dim=embed_dim, depth=12, num_heads=12)
        self.mil_attention = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.Tanh(), nn.Linear(128, 1)
        )

    def forward(self, patches):
        B, N, C, H, W = patches.shape
        x = patches.flatten(0, 1) # (B*N, C, H, W)
        x = self.vit.patch_embed(x)
        for blk in self.vit.blocks: x = blk(x)
        tokens = self.vit.norm(x) # (B*N, Num_Tokens, 768)
        tokens = tokens.view(B, N, -1, tokens.shape[-1]).flatten(1, 2) # (B, M, 768)
        
        attn_weights = self.mil_attention(tokens)
        attn_weights = F.softmax(attn_weights, dim=1)
        patient_vec = torch.sum(tokens * attn_weights, dim=1) # (B, 768)
        return tokens, patient_vec

class ClinicalEncoder(nn.Module):
    """临床表格数据编码器 """
    def __init__(self, input_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU()
        )
    def forward(self, x): return self.mlp(x)

# ==========================================
# 3. 剂量引导的交叉模态对齐 
# ==========================================
class DoseGuidedCrossModalAlignment(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.Wq = nn.Linear(d, d)
        self.Wk = nn.Linear(d, d)
        self.Wv = nn.Linear(d, d)
        self.scale = math.sqrt(d)

    def forward(self, dose_tokens, target_tokens):
        """Dose 作为 Query, CT/Path 作为 Key/Value"""
        Q = self.Wq(dose_tokens)
        K = self.Wk(target_tokens)
        V = self.Wv(target_tokens)
        attn = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attn = F.softmax(attn, dim=-1)
        return torch.bmm(attn, V), attn # 返回对齐特征和注意力权重(用于可解释性)

# ==========================================
# 4. 不确定性感知的异构图融合 
# ==========================================
class UncertaintyAwareGraphFusion(nn.Module):
    def __init__(self, d=256, num_nodes=4):
        super().__init__()
        self.Wq = nn.Linear(d, d)
        self.Wk = nn.Linear(d, d)
        self.Wv = nn.Linear(d, d)
        self.scale = math.sqrt(d)
        self.gat_conv1 = nn.Linear(d, d)
        self.gat_conv2 = nn.Linear(d, d)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, node_features, uncertainty_vars=None):
        """动态边更新 + 2层 GAT"""
        Q = self.Wq(node_features)
        K = self.Wk(node_features)
        A = F.softmax(torch.bmm(Q, K.transpose(1, 2)) / self.scale, dim=-1) # (B, V, V)
        
        if uncertainty_vars is not None:
            conf_weight = 1.0 / (1.0 + uncertainty_vars.squeeze(-1))
            A = A * conf_weight.unsqueeze(1)
            A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)

        H_v = self.Wv(node_features)
        H_1 = self.leaky_relu(torch.bmm(A, H_v))
        H_1 = self.gat_conv1(H_1)
        H_2 = self.leaky_relu(torch.bmm(A, H_1))
        H_2 = self.gat_conv2(H_2)
        return H_2.mean(dim=1) # Readout: (B, d)

# ==========================================
# 5. 完整模型封装 (DGMMP-Net)
# ==========================================
class DGMMPNet(nn.Module):
    def __init__(self, clinical_dim=32, d_latent=256, mc_dropout_rate=0.3):
        super().__init__()
        self.d = d_latent
        # Encoders
        self.ct_encoder = VolumeEncoder3D(1)
        self.dose_encoder = VolumeEncoder3D(1)
        self.path_encoder = PathologyEncoder()
        self.clin_encoder = ClinicalEncoder(clinical_dim)
        
        # Projections to d=256
        self.proj_ct = nn.Linear(512, d_latent)
        self.proj_dose = nn.Linear(512, d_latent)
        self.proj_path = nn.Linear(768, d_latent)
        self.proj_clin = nn.Linear(64, d_latent)
        
        # Alignment
        self.align_ct = DoseGuidedCrossModalAlignment(d_latent)
        self.align_path = DoseGuidedCrossModalAlignment(d_latent)
        
        # Graph Fusion
        self.graph_fusion = UncertaintyAwareGraphFusion(d_latent, num_nodes=4)
        
        # Prediction Head with MC Dropout
        self.head_dropout = MCDropout(p=mc_dropout_rate)
        self.survival_head = nn.Linear(d_latent, 1)
        self.recurrence_head = nn.Linear(d_latent, 3) # 3种复发模式
        self.toxicity_head = nn.Linear(d_latent, 2)   # 肺/心脏毒性

    def _flatten_spatial(self, feat_map):
        B, C, D, H, W = feat_map.shape
        return feat_map.flatten(2).transpose(1, 2)

    def forward(self, ct_vol, dose_vol, path_patches, clin_data, mc_passes=1, return_attn=False):
        # 1. 提取特征
        ct_map, _ = self.ct_encoder(ct_vol)
        dose_map, _ = self.dose_encoder(dose_vol)
        path_tokens, _ = self.path_encoder(path_patches)
        clin_vec = self.clin_encoder(clin_data)
        
        # 投影
        ct_tokens = self.proj_ct(self._flatten_spatial(ct_map))
        dose_tokens = self.proj_dose(self._flatten_spatial(dose_map))
        path_tokens = self.proj_path(path_tokens)
        
        # 2. 剂量引导对齐 (Eq. 1)
        aligned_ct, attn_ct = self.align_ct(dose_tokens, ct_tokens)
        aligned_path, attn_path = self.align_path(dose_tokens, path_tokens)
        
        # 池化为节点
        nodes = torch.stack([
            aligned_ct.mean(dim=1), dose_tokens.mean(dim=1), 
            aligned_path.mean(dim=1), self.proj_clin(clin_vec)
        ], dim=1) # (B, 4, d)
        
        # 3. 图融合
        fused_repr = self.graph_fusion(nodes)
        
        # 4. MC Dropout 预测
        risk_scores, rec_probs, tox_probs = [], [], []
        for _ in range(mc_passes):
            h = self.head_dropout(fused_repr)
            risk_scores.append(self.survival_head(h))
            rec_probs.append(self.recurrence_head(h))
            tox_probs.append(self.toxicity_head(h))
            
        risk_scores = torch.stack(risk_scores, dim=0)
        rec_probs = torch.stack(rec_probs, dim=0)
        tox_probs = torch.stack(tox_probs, dim=0)
        
        out = {
            'risk_mean': risk_scores.mean(dim=0),
            'risk_var': torch.var(risk_scores, dim=0),
            'rec_mean': rec_probs.mean(dim=0),
            'tox_mean': tox_probs.mean(dim=0),
            'node_features': nodes
        }
        if return_attn:
            out['attn_ct'] = attn_ct
            out['attn_path'] = attn_path
        return out