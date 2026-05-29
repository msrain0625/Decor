"""
═══════════════════════════════════════════════════════════════════════════
任务 B v1.5: v1 (4 簇 5 路融合) + 三处关键修复
═══════════════════════════════════════════════════════════════════════════

vs v1 / v2 的修复:
  v1/v2 失败模式 (从 80k 训练曲线诊断):
    - l21_cluster_* 全程 ≈ 0  → 簇内原型从未分散过
    - loss_cluster_refine 持续上涨 (0.10 → 0.30+)  → 监督方向错
    - wearing/wears 周期性互斥 (15k 共存,之后翻转)
    - 根因: supervised_mask = (gt == p_hat) 形成自我强化循环

  v1.5 三处修复:
    [Fix 1] 解耦 supervised_mask
      旧: (gt_cluster == hat_cluster) & (gt_cluster != -1)  ← 只在 base 已对时监督
      新: gt_cluster != -1                                    ← 任何 GT 落入簇就监督
      训练时 cluster_head 按 GT 路由,推理时按 p_hat 路由

    [Fix 2] cluster_head 不污染 backbone
      新: rel_feat 进 cluster_head 前 detach
      理由: anchor/refine loss 通过 rel_feat 反传会扰动 PE-net 的特征分布
            cluster_head 应是纯粹的"事后修正者"

    [Fix 3] 双层监督学开簇内原型
      新增 loss_anchor: InfoNCE on fused_emb ↔ cluster_protos
        让 wearing 样本的 fused_emb 拉向 wearing proto, 推离 wears proto
        这是"数据驱动的方向分离",真正学到判别轴
      新增 GloVe 初始化 cluster_protos (而非 random)
        给原型先验上正确的方向
      l21_cluster 改为 margin hinge (cos_sim > -0.1 才有梯度)
        仅作为防塌缩兜底,不主导分离方向

数据流:
  rel_feat (PE-net 输出)
    ├─→ 5 套 vis 原型 max → vis_dist_full
    ├─→ 全局语义 → sem_dist
    └─→ base = sem_dist + vis_dist_full           ← backbone 唯一通路

  rel_feat.detach() + union + sub_vis + obj_vis + sub_sem + obj_sem + geo
    └─→ cluster_head[c] (5 路融合)
          ├─→ fused_emb (mlp_dim, L2 norm) → loss_anchor
          └─→ cluster_logits (|c| 维) → refine_correction (51 维稀疏)

  final = base + 0.5 * refine_correction   → loss_rel (CE)

簇定义 (4 个最关键的, 共 10 个谓词):
  C0: wearing(48), wears(49)               ← 主要靠 sub_sem (人/衣) 与方向
  C1: on(31), over(33), above(1)           ← 主要靠 geometry (垂直 + IoU)
  C2: holding(21), carrying(11)            ← 主要靠 sub_vis (姿态)
  C3: sitting on(40), laying on(24), lying on(26)  ← 主要靠 sub_vis (姿态)
═══════════════════════════════════════════════════════════════════════════
"""

import torch
from math import sqrt

from torch import nn
from torch.nn import functional as F
import numpy as np

from hetsgg.config import cfg
from hetsgg.data import get_dataset_statistics
from hetsgg.modeling import registry
from hetsgg.modeling.roi_heads.relation_head.classifier import build_classifier
from hetsgg.modeling.roi_heads.relation_head.model_kern import (
    to_onehot,
)

from hetsgg.modeling.utils import cat
from hetsgg.structures.boxlist_ops import squeeze_tensor
from .model_motifs import FrequencyBias

from hetsgg.modeling.roi_heads.relation_head.model_HetSGG import HetSGG
from hetsgg.modeling.roi_heads.relation_head.model_HetSGGplus import HetSGGplus_Context

from .rel_proposal_network.loss import (
    RelAwareLoss,
)
from .utils_relation import obj_prediction_nms
from .utils_motifs import obj_edge_vectors, rel_vectors, encode_box_info
from .utils_relation import layer_init, get_box_info, get_box_pair_info, obj_prediction_nms
from hetsgg.modeling.make_layers import make_fc
import h5py
from .utils_relation import nms_overlaps


# ════════════════════════════════════════════════════════════════════════════
# 混淆簇定义
# ════════════════════════════════════════════════════════════════════════════
CONFUSION_CLUSTERS = [
    [48, 49],            # C0: wearing, wears
    [31, 33, 1],         # C1: on, over, above
    [21, 11],            # C2: holding, carrying
    [40, 24, 26],        # C3: sitting on, laying on, lying on
]
NUM_CLUSTERS = len(CONFUSION_CLUSTERS)


def build_predicate_to_cluster():
    p2c = torch.full((51,), -1, dtype=torch.long)
    for c_id, members in enumerate(CONFUSION_CLUSTERS):
        for p in members:
            assert p2c[p] == -1, f"predicate {p} appears in multiple clusters"
            p2c[p] = c_id
    return p2c


# ════════════════════════════════════════════════════════════════════════════
# 几何特征 (9 维, 已做 NaN/Inf 防御)
# ════════════════════════════════════════════════════════════════════════════
def compute_pair_geometry(sub_box, obj_box):
    eps = 1e-6
    sx = (sub_box[:, 0] + sub_box[:, 2]) / 2
    sy = (sub_box[:, 1] + sub_box[:, 3]) / 2
    ox = (obj_box[:, 0] + obj_box[:, 2]) / 2
    oy = (obj_box[:, 1] + obj_box[:, 3]) / 2

    sw = (sub_box[:, 2] - sub_box[:, 0]).clamp(min=eps)
    sh = (sub_box[:, 3] - sub_box[:, 1]).clamp(min=eps)
    ow = (obj_box[:, 2] - obj_box[:, 0]).clamp(min=eps)
    oh = (obj_box[:, 3] - obj_box[:, 1]).clamp(min=eps)

    rel_dx = (sx - ox) / (ow + eps)
    rel_dy = (sy - oy) / (oh + eps)
    rel_w = torch.log((sw / ow).clamp(min=eps))
    rel_h = torch.log((sh / oh).clamp(min=eps))

    ix1 = torch.max(sub_box[:, 0], obj_box[:, 0])
    iy1 = torch.max(sub_box[:, 1], obj_box[:, 1])
    ix2 = torch.min(sub_box[:, 2], obj_box[:, 2])
    iy2 = torch.min(sub_box[:, 3], obj_box[:, 3])
    iw = (ix2 - ix1).clamp(min=0)
    ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih
    sub_area = sw * sh
    obj_area = ow * oh
    union = sub_area + obj_area - inter + eps
    iou = inter / union

    sub_in_obj = inter / (sub_area + eps)
    obj_in_sub = inter / (obj_area + eps)
    vert_above = (obj_box[:, 1] - sub_box[:, 1]) / (oh + eps)
    log_area_ratio = torch.log((sub_area / (obj_area + eps)).clamp(min=eps))

    geo = torch.stack([
        rel_dx, rel_dy, rel_w, rel_h, iou,
        sub_in_obj, obj_in_sub, vert_above, log_area_ratio
    ], dim=-1)

    geo = torch.nan_to_num(geo, nan=0.0, posinf=10.0, neginf=-10.0)
    geo = torch.clamp(geo, min=-10.0, max=10.0)
    return geo


# ════════════════════════════════════════════════════════════════════════════
# Cluster Refine Head
#   5 路独立投影 → 拼接 → 融合层 → 簇内原型相似度
#   forward 可选返回 fused_emb 用于 anchor loss
# ════════════════════════════════════════════════════════════════════════════
class ClusterRefineHead(nn.Module):
    def __init__(self, cluster_size,
                 rel_dim=2048, union_dim=4096, obj_vis_dim=2048,
                 obj_emb_dim=300, geo_dim=9,
                 mlp_dim=2048, hidden_dim=1024,
                 init_protos=None):
        """
        init_protos: [cluster_size, 300] GloVe 向量, 用于初始化 cluster_protos
                     None 时回退到 random init
        """
        super().__init__()
        self.cluster_size = cluster_size
        self.mlp_dim = mlp_dim

        # 5 路独立投影
        self.proj_rel = nn.Linear(rel_dim, hidden_dim)
        self.proj_union = nn.Linear(union_dim, hidden_dim)
        self.proj_subobj_vis = nn.Linear(obj_vis_dim * 2, hidden_dim)
        self.proj_subobj_sem = nn.Linear(obj_emb_dim * 2, hidden_dim)
        self.proj_geo = nn.Sequential(
            nn.Linear(geo_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, hidden_dim),
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 5, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(mlp_dim, mlp_dim),
        )

        # 簇内原型 — GloVe 初始化优先
        if init_protos is not None:
            assert init_protos.size(0) == cluster_size
            with torch.no_grad():
                proj = torch.randn(init_protos.size(1), mlp_dim) * (
                    1.0 / sqrt(init_protos.size(1))
                )
                init = init_protos.float() @ proj
                init = init + torch.randn_like(init) * 0.01
                init = F.normalize(init, p=2, dim=-1) * 0.1
            self.protos = nn.Parameter(init)
        else:
            self.protos = nn.Parameter(torch.randn(cluster_size, mlp_dim) * 0.02)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        for m in [self.proj_rel, self.proj_union,
                  self.proj_subobj_vis, self.proj_subobj_sem]:
            layer_init(m, xavier=True)
        for m in self.proj_geo:
            if isinstance(m, nn.Linear):
                layer_init(m, xavier=True)
        for m in self.fusion:
            if isinstance(m, nn.Linear):
                layer_init(m, xavier=True)

    def forward(self, rel_feat, union_feat,
                sub_vis, obj_vis, sub_sem, obj_sem, geometry,
                return_fused=False):
        h_rel = torch.relu(self.proj_rel(rel_feat))
        h_union = torch.relu(self.proj_union(union_feat))
        h_subobj_vis = torch.relu(self.proj_subobj_vis(
            torch.cat([sub_vis, obj_vis], dim=-1)))
        h_subobj_sem = torch.relu(self.proj_subobj_sem(
            torch.cat([sub_sem, obj_sem], dim=-1)))
        h_geo = self.proj_geo(geometry)

        fused = torch.cat([h_rel, h_union, h_subobj_vis, h_subobj_sem, h_geo],
                          dim=-1)
        out = self.fusion(fused)

        out_norm = F.normalize(out, p=2, dim=-1, eps=1e-6)
        protos_norm = F.normalize(self.protos, p=2, dim=-1, eps=1e-6)
        scale = self.logit_scale.clamp(max=4.0).exp()
        logits = (out_norm @ protos_norm.t()) * scale

        if return_fused:
            return logits, out_norm
        return logits


@registry.ROI_RELATION_PREDICTOR.register("PrototypeEmbeddingNetwork")
class PrototypeEmbeddingNetwork(nn.Module):
    def __init__(self, config, in_channels):
        super(PrototypeEmbeddingNetwork, self).__init__()

        self.num_obj_cls = 151
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = 51
        self.cfg = config

        assert in_channels is not None
        self.in_channels = in_channels
        self.obj_dim = in_channels

        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)

        self.hidden_dim = 512
        self.pooling_dim = 4096
        self.mlp_dim = 2048
        self.post_emb = nn.Linear(self.obj_dim, self.mlp_dim * 2)
        self.embed_dim = 300
        dropout_p = 0.2

        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        rel_embed_vecs = rel_vectors(rel_classes, wv_dir=config.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.rel_embed = nn.Embedding(self.num_rel_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)

        self.W_sub = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_obj = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_pred = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)

        self.gate_sub = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        self.gate_obj = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        self.gate_pred = nn.Linear(self.mlp_dim * 2, self.mlp_dim)

        self.vis2sem = nn.Sequential(*[
            nn.Linear(self.mlp_dim, self.mlp_dim * 2), nn.ReLU(True),
            nn.Dropout(dropout_p), nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        ])

        self.project_head = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim * 2, 2)

        self.linear_sub = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_obj = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_pred = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_rel_rep = nn.Linear(self.mlp_dim, self.mlp_dim)

        self.norm_sub = nn.LayerNorm(self.mlp_dim)
        self.norm_obj = nn.LayerNorm(self.mlp_dim)
        self.norm_rel_rep = nn.LayerNorm(self.mlp_dim)

        self.dropout_sub = nn.Dropout(dropout_p)
        self.dropout_obj = nn.Dropout(dropout_p)
        self.dropout_rel_rep = nn.Dropout(dropout_p)
        self.dropout_rel = nn.Dropout(dropout_p)
        self.dropout_pred = nn.Dropout(dropout_p)

        self.down_samp = MLP(self.pooling_dim, self.mlp_dim, self.mlp_dim, 2)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # ----- refine object labels -----
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum=0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])

        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.obj_dim = in_channels
        self.out_obj = make_fc(self.hidden_dim, self.num_obj_classes)
        self.lin_obj_cyx = make_fc(self.obj_dim + self.embed_dim + 128, self.hidden_dim)

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        self.nms_thresh = 0.5
        self.post_cat = MLP(self.mlp_dim, self.mlp_dim * 2, self.mlp_dim, 2)

        # ════════════════════════════════════════════════════════════════════
        # 5 套视觉原型
        # ════════════════════════════════════════════════════════════════════
        proto = 'proto/muil_sgl_p.h5'
        f = h5py.File(proto, 'r')

        self.vis_proto_0 = torch.from_numpy(f['proto_0'][:])
        self.vis_W_0 = MLP(self.mlp_dim, self.mlp_dim * 2, self.mlp_dim, 2)
        self.norm_vis_rep_0 = nn.LayerNorm(self.mlp_dim)
        self.dropout_vis_rep_0 = nn.Dropout(dropout_p)
        self.linear_vis_rep_0 = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.dropout_vis_0 = nn.Dropout(dropout_p)
        self.project_head_vis_0 = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim * 2, 2)
        self.dropout_pred_vis_0 = nn.Dropout(dropout_p)
        self.logit_scale_vis_0 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.vis_embed_0 = nn.Embedding(51, 2048)
        with torch.no_grad():
            self.vis_embed_0.weight.copy_(self.vis_proto_0, non_blocking=True)

        self.vis_proto_1 = torch.from_numpy(f['proto_1'][:])
        self.vis_W_1 = MLP(self.mlp_dim, self.mlp_dim * 2, self.mlp_dim, 2)
        self.norm_vis_rep_1 = nn.LayerNorm(self.mlp_dim)
        self.dropout_vis_rep_1 = nn.Dropout(dropout_p)
        self.linear_vis_rep_1 = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.dropout_vis_1 = nn.Dropout(dropout_p)
        self.project_head_vis_1 = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim * 2, 2)
        self.dropout_pred_vis_1 = nn.Dropout(dropout_p)
        self.logit_scale_vis_1 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.vis_embed_1 = nn.Embedding(51, 2048)
        with torch.no_grad():
            self.vis_embed_1.weight.copy_(self.vis_proto_1, non_blocking=True)

        self.vis_proto_2 = torch.from_numpy(f['proto_2'][:])
        self.vis_W_2 = MLP(self.mlp_dim, self.mlp_dim * 2, self.mlp_dim, 2)
        self.norm_vis_rep_2 = nn.LayerNorm(self.mlp_dim)
        self.dropout_vis_rep_2 = nn.Dropout(dropout_p)
        self.linear_vis_rep_2 = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.dropout_vis_2 = nn.Dropout(dropout_p)
        self.project_head_vis_2 = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim * 2, 2)
        self.dropout_pred_vis_2 = nn.Dropout(dropout_p)
        self.logit_scale_vis_2 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.vis_embed_2 = nn.Embedding(51, 2048)
        with torch.no_grad():
            self.vis_embed_2.weight.copy_(self.vis_proto_2, non_blocking=True)

        self.vis_proto_3 = torch.from_numpy(f['proto_3'][:])
        self.vis_W_3 = MLP(self.mlp_dim, self.mlp_dim * 2, self.mlp_dim, 2)
        self.norm_vis_rep_3 = nn.LayerNorm(self.mlp_dim)
        self.dropout_vis_rep_3 = nn.Dropout(dropout_p)
        self.linear_vis_rep_3 = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.dropout_vis_3 = nn.Dropout(dropout_p)
        self.project_head_vis_3 = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim * 2, 2)
        self.dropout_pred_vis_3 = nn.Dropout(dropout_p)
        self.logit_scale_vis_3 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.vis_embed_3 = nn.Embedding(51, 2048)
        with torch.no_grad():
            self.vis_embed_3.weight.copy_(self.vis_proto_3, non_blocking=True)

        self.vis_proto_4 = torch.from_numpy(f['proto_4'][:])
        self.vis_W_4 = MLP(self.mlp_dim, self.mlp_dim * 2, self.mlp_dim, 2)
        self.norm_vis_rep_4 = nn.LayerNorm(self.mlp_dim)
        self.dropout_vis_rep_4 = nn.Dropout(dropout_p)
        self.linear_vis_rep_4 = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.dropout_vis_4 = nn.Dropout(dropout_p)
        self.project_head_vis_4 = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim * 2, 2)
        self.dropout_pred_vis_4 = nn.Dropout(dropout_p)
        self.logit_scale_vis_4 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.vis_embed_4 = nn.Embedding(51, 2048)
        with torch.no_grad():
            self.vis_embed_4.weight.copy_(self.vis_proto_4, non_blocking=True)

        layer_init(self.linear_vis_rep_0, xavier=True)
        layer_init(self.linear_vis_rep_1, xavier=True)
        layer_init(self.linear_vis_rep_2, xavier=True)
        layer_init(self.linear_vis_rep_3, xavier=True)
        layer_init(self.linear_vis_rep_4, xavier=True)
        f.close()

        self.context_layer = HetSGG(config, in_channels)
        self.ent_post = MLP(128, 1024, 4096, 2)
        self.rel_post = MLP(128, 1024, 2048, 2)

        # ----- het 相关 -----
        self.use_bias = cfg.MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS
        self.use_obj_recls_logits = config.MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS
        self.n_reltypes = cfg.MODEL.ROI_RELATION_HEAD.HETSGG.NUM_RELATION
        self.n_dim = cfg.MODEL.ROI_RELATION_HEAD.HETSGG.H_DIM
        self.n_ntypes = int(sqrt(self.n_reltypes))
        self.obj2rtype = {(i, j): self.n_ntypes * j + i for j in range(self.n_ntypes) for i in range(self.n_ntypes)}
        self.rel_aware_model_on = config.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON
        self.rel_classifier = build_classifier(self.num_rel_cls, self.num_rel_cls)
        self.obj_classifier = build_classifier(self.num_obj_cls, self.num_obj_cls)

        if self.use_bias:
            self.freq_bias = FrequencyBias(config, statistics)
            self.freq_lambda = nn.Parameter(torch.Tensor([1.0]), requires_grad=False)

        # ════════════════════════════════════════════════════════════════════
        # [v1.5] Cluster Refine Heads (4 簇,5 路融合,GloVe 初始化原型)
        # ════════════════════════════════════════════════════════════════════
        self.num_clusters = NUM_CLUSTERS
        predicate_to_cluster = build_predicate_to_cluster()
        self.register_buffer("predicate_to_cluster", predicate_to_cluster)

        for c_id, members in enumerate(CONFUSION_CLUSTERS):
            self.register_buffer(
                f"cluster_members_{c_id}",
                torch.tensor(members, dtype=torch.long)
            )

        self.cluster_heads = nn.ModuleList()
        for c_id, members in enumerate(CONFUSION_CLUSTERS):
            init_vecs = rel_embed_vecs[torch.tensor(members)]   # [|c|, 300]
            self.cluster_heads.append(
                ClusterRefineHead(
                    cluster_size=len(members),
                    rel_dim=self.mlp_dim,
                    union_dim=self.pooling_dim,
                    obj_vis_dim=self.mlp_dim,
                    obj_emb_dim=self.embed_dim,
                    geo_dim=9,
                    mlp_dim=self.mlp_dim,
                    hidden_dim=1024,
                    init_protos=init_vecs,
                )
            )

        # 融合权重与 loss 权重 (与 v1 一致, 不引入 warmup)
        self.refine_alpha = 0.35
        self.refine_loss_weight = 0.3
        self.anchor_loss_weight = 0.3
        self.cluster_push_margin = -0.1
        self.cluster_push_weight = 0.1

    def init_classifier_weight(self):
        self.obj_classifier.reset_parameters()
        for i in range(self.n_reltypes):
            self.rel_classifier[i].reset_parameters()

    def get_cluster_members(self, c_id):
        return getattr(self, f"cluster_members_{c_id}")

    # ════════════════════════════════════════════════════════════════════════
    # [v1.5 核心] 计算 refine_correction
    #   routing_labels: 训练时传 GT 标签, 推理时传 p_hat
    #   return_fused: 训练时为 True, 同时返回各簇 fused_emb 用于 anchor loss
    # ════════════════════════════════════════════════════════════════════════
    def compute_refine_correction(self, routing_labels,
                                  rel_feat, union_feat,
                                  sub_vis, obj_vis,
                                  sub_sem, obj_sem,
                                  geometry,
                                  return_fused=False):
        N = rel_feat.shape[0]
        cluster_id = self.predicate_to_cluster[routing_labels]   # [N], -1 或簇 idx
        refine_correction = rel_feat.new_zeros(N, self.num_rel_cls)
        fused_per_cluster = {} if return_fused else None

        if (cluster_id >= 0).sum() == 0:
            if return_fused:
                return refine_correction, fused_per_cluster
            return refine_correction

        for c in range(self.num_clusters):
            relevant = (cluster_id == c)
            if not relevant.any():
                continue

            # [Fix 2] rel_feat 在进 cluster_head 前 detach,断开对 backbone 的梯度
            rel_in = rel_feat[relevant].detach()

            if return_fused:
                cluster_logits, fused_emb = self.cluster_heads[c](
                    rel_in,
                    union_feat[relevant],
                    sub_vis[relevant], obj_vis[relevant],
                    sub_sem[relevant], obj_sem[relevant],
                    geometry[relevant],
                    return_fused=True,
                )
                rows = torch.where(relevant)[0]
                fused_per_cluster[c] = (rows, fused_emb)
            else:
                cluster_logits = self.cluster_heads[c](
                    rel_in,
                    union_feat[relevant],
                    sub_vis[relevant], obj_vis[relevant],
                    sub_sem[relevant], obj_sem[relevant],
                    geometry[relevant],
                )

            members = self.get_cluster_members(c)
            scatter_buf = rel_feat.new_zeros(int(relevant.sum()), self.num_rel_cls)
            scatter_buf[:, members] = cluster_logits
            refine_correction[relevant] = scatter_buf

        if return_fused:
            return refine_correction, fused_per_cluster
        return refine_correction

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None,
                is_training=True):

        add_losses = {}
        add_data = {}

        entity_dists, entity_preds = self.refine_obj_labels(roi_features, proposals)

        ent_feat, rel_feat = self.context_layer(roi_features, union_features, proposals, rel_pair_idxs, rel_binarys,
                                                logger, is_training)

        rel_feat = self.rel_post(rel_feat)
        entity_rep = self.post_emb(roi_features)
        entity_rep = entity_rep.view(entity_rep.size(0), 2, self.mlp_dim)

        sub_rep = entity_rep[:, 1].contiguous().view(-1, self.mlp_dim)
        obj_rep = entity_rep[:, 0].contiguous().view(-1, self.mlp_dim)

        entity_embeds = self.obj_embed(entity_preds)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        sub_reps = sub_rep.split(num_objs, dim=0)
        obj_reps = obj_rep.split(num_objs, dim=0)
        entity_preds_split = entity_preds.split(num_objs, dim=0)
        entity_embeds_split = entity_embeds.split(num_objs, dim=0)

        fusion_so = []
        pair_preds = []

        # 收集供 cluster_head 使用的 5 路 per-pair 信号
        pair_sub_vis_list = []
        pair_obj_vis_list = []
        pair_sub_sem_list = []
        pair_obj_sem_list = []
        pair_sub_box_list = []
        pair_obj_box_list = []

        for pair_idx, sub_rep_i, obj_rep_i, entity_pred_i, entity_embed_i, proposal in zip(
                rel_pair_idxs, sub_reps, obj_reps, entity_preds_split, entity_embeds_split, proposals):
            s_embed = self.W_sub(entity_embed_i[pair_idx[:, 0]])
            o_embed = self.W_obj(entity_embed_i[pair_idx[:, 1]])

            sem_sub = self.vis2sem(sub_rep_i[pair_idx[:, 0]])
            sem_obj = self.vis2sem(obj_rep_i[pair_idx[:, 1]])

            gate_sem_sub = torch.sigmoid(self.gate_sub(cat((s_embed, sem_sub), dim=-1)))
            gate_sem_obj = torch.sigmoid(self.gate_obj(cat((o_embed, sem_obj), dim=-1)))

            sub = s_embed + sem_sub * gate_sem_sub
            obj = o_embed + sem_obj * gate_sem_obj

            sub = self.norm_sub(self.dropout_sub(torch.relu(self.linear_sub(sub))) + sub)
            obj = self.norm_obj(self.dropout_obj(torch.relu(self.linear_obj(obj))) + obj)

            fusion_so.append(fusion_func(sub, obj))
            pair_preds.append(torch.stack((entity_pred_i[pair_idx[:, 0]], entity_pred_i[pair_idx[:, 1]]), dim=1))

            # cluster_head 输入 — 只收集,不参与 backbone 计算
            pair_sub_vis_list.append(sub_rep_i[pair_idx[:, 0]])
            pair_obj_vis_list.append(obj_rep_i[pair_idx[:, 1]])
            pair_sub_sem_list.append(entity_embed_i[pair_idx[:, 0]])
            pair_obj_sem_list.append(entity_embed_i[pair_idx[:, 1]])
            boxes = proposal.bbox
            pair_sub_box_list.append(boxes[pair_idx[:, 0]])
            pair_obj_box_list.append(boxes[pair_idx[:, 1]])

        fusion_so = cat(fusion_so, dim=0)
        fusion_so = self.post_cat(fusion_so)

        pair_sub_vis = cat(pair_sub_vis_list, dim=0)
        pair_obj_vis = cat(pair_obj_vis_list, dim=0)
        pair_sub_sem = cat(pair_sub_sem_list, dim=0)
        pair_obj_sem = cat(pair_obj_sem_list, dim=0)
        pair_sub_box = cat(pair_sub_box_list, dim=0)
        pair_obj_box = cat(pair_obj_box_list, dim=0)

        pair_geometry = compute_pair_geometry(pair_sub_box, pair_obj_box)

        sem_pred = self.vis2sem(self.down_samp(union_features))
        gate_sem_pred = torch.sigmoid(self.gate_pred(cat((fusion_so, sem_pred), dim=-1)))

        # ════════════════════════════════════════════════════════════════════
        # 5 套视觉原型并行 max
        # ════════════════════════════════════════════════════════════════════
        vis_proto_0_w = self.vis_W_0(self.vis_embed_0.weight)
        vis_rep_0 = self.norm_vis_rep_0(
            self.dropout_vis_rep_0(torch.relu(self.linear_vis_rep_0(rel_feat))) + rel_feat
        )
        vis_rep_0 = self.project_head_vis_0(self.dropout_vis_0(torch.relu(vis_rep_0)))
        predicate_proto_vis_0 = self.project_head_vis_0(
            self.dropout_pred_vis_0(torch.relu(vis_proto_0_w))
        )
        vis_rep_norm_0 = vis_rep_0 / vis_rep_0.norm(dim=1, keepdim=True)
        predicate_proto_vis_norm_0 = predicate_proto_vis_0 / predicate_proto_vis_0.norm(dim=1, keepdim=True)
        vis_dist_0 = vis_rep_norm_0 @ predicate_proto_vis_norm_0.t() * self.logit_scale_vis_0.exp()

        vis_proto_1_w = self.vis_W_1(self.vis_embed_1.weight)
        vis_rep_1 = self.norm_vis_rep_1(
            self.dropout_vis_rep_1(torch.relu(self.linear_vis_rep_1(rel_feat))) + rel_feat
        )
        vis_rep_1 = self.project_head_vis_1(self.dropout_vis_1(torch.relu(vis_rep_1)))
        predicate_proto_vis_1 = self.project_head_vis_1(
            self.dropout_pred_vis_1(torch.relu(vis_proto_1_w))
        )
        vis_rep_norm_1 = vis_rep_1 / vis_rep_1.norm(dim=1, keepdim=True)
        predicate_proto_vis_norm_1 = predicate_proto_vis_1 / predicate_proto_vis_1.norm(dim=1, keepdim=True)
        vis_dist_1 = vis_rep_norm_1 @ predicate_proto_vis_norm_1.t() * self.logit_scale_vis_1.exp()

        vis_proto_2_w = self.vis_W_2(self.vis_embed_2.weight)
        vis_rep_2 = self.norm_vis_rep_2(
            self.dropout_vis_rep_2(torch.relu(self.linear_vis_rep_2(rel_feat))) + rel_feat
        )
        vis_rep_2 = self.project_head_vis_2(self.dropout_vis_2(torch.relu(vis_rep_2)))
        predicate_proto_vis_2 = self.project_head_vis_2(
            self.dropout_pred_vis_2(torch.relu(vis_proto_2_w))
        )
        vis_rep_norm_2 = vis_rep_2 / vis_rep_2.norm(dim=1, keepdim=True)
        predicate_proto_vis_norm_2 = predicate_proto_vis_2 / predicate_proto_vis_2.norm(dim=1, keepdim=True)
        vis_dist_2 = vis_rep_norm_2 @ predicate_proto_vis_norm_2.t() * self.logit_scale_vis_2.exp()

        vis_proto_3_w = self.vis_W_3(self.vis_embed_3.weight)
        vis_rep_3 = self.norm_vis_rep_3(
            self.dropout_vis_rep_3(torch.relu(self.linear_vis_rep_3(rel_feat))) + rel_feat
        )
        vis_rep_3 = self.project_head_vis_3(self.dropout_vis_3(torch.relu(vis_rep_3)))
        predicate_proto_vis_3 = self.project_head_vis_3(
            self.dropout_pred_vis_3(torch.relu(vis_proto_3_w))
        )
        vis_rep_norm_3 = vis_rep_3 / vis_rep_3.norm(dim=1, keepdim=True)
        predicate_proto_vis_norm_3 = predicate_proto_vis_3 / predicate_proto_vis_3.norm(dim=1, keepdim=True)
        vis_dist_3 = vis_rep_norm_3 @ predicate_proto_vis_norm_3.t() * self.logit_scale_vis_3.exp()

        vis_proto_4_w = self.vis_W_4(self.vis_embed_4.weight)
        vis_rep_4 = self.norm_vis_rep_4(
            self.dropout_vis_rep_4(torch.relu(self.linear_vis_rep_4(rel_feat))) + rel_feat
        )
        vis_rep_4 = self.project_head_vis_4(self.dropout_vis_4(torch.relu(vis_rep_4)))
        predicate_proto_vis_4 = self.project_head_vis_4(
            self.dropout_pred_vis_4(torch.relu(vis_proto_4_w))
        )
        vis_rep_norm_4 = vis_rep_4 / vis_rep_4.norm(dim=1, keepdim=True)
        predicate_proto_vis_norm_4 = predicate_proto_vis_4 / predicate_proto_vis_4.norm(dim=1, keepdim=True)
        vis_dist_4 = vis_rep_norm_4 @ predicate_proto_vis_norm_4.t() * self.logit_scale_vis_4.exp()

        vis_dist_stack = torch.stack([vis_dist_0, vis_dist_1, vis_dist_2,
                                      vis_dist_3, vis_dist_4], dim=0)
        vis_dist_full = vis_dist_stack.max(dim=0)[0]

        # ════════════════════════════════════════════════════════════════════
        # 语义分支
        # ════════════════════════════════════════════════════════════════════
        rel_rep = fusion_so - sem_pred * gate_sem_pred
        predicate_proto = self.W_pred(self.rel_embed.weight)

        rel_rep = self.norm_rel_rep(self.dropout_rel_rep(torch.relu(self.linear_rel_rep(rel_rep))) + rel_rep)
        rel_rep = self.project_head(self.dropout_rel(torch.relu(rel_rep)))
        predicate_proto = self.project_head(self.dropout_pred(torch.relu(predicate_proto)))

        rel_rep_norm = rel_rep / rel_rep.norm(dim=1, keepdim=True)
        predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)

        sem_dist = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()

        # ════════════════════════════════════════════════════════════════════
        # Stage 1: base 全局判别
        # ════════════════════════════════════════════════════════════════════
        base_logits = sem_dist + vis_dist_full

        # ════════════════════════════════════════════════════════════════════
        # Stage 2: cluster_head 二阶段加性修正
        #   [Fix 1] 训练时按 GT 路由,推理时按 p_hat 路由
        # ════════════════════════════════════════════════════════════════════
        p_hat = base_logits.argmax(dim=-1).detach()

        if self.training:
            rel_labels_cat = cat(rel_labels, dim=0)
            routing_labels = rel_labels_cat
            refine_correction, fused_per_cluster = self.compute_refine_correction(
                routing_labels,
                rel_feat=rel_feat, union_feat=union_features,
                sub_vis=pair_sub_vis, obj_vis=pair_obj_vis,
                sub_sem=pair_sub_sem, obj_sem=pair_obj_sem,
                geometry=pair_geometry,
                return_fused=True,
            )
        else:
            routing_labels = p_hat
            refine_correction = self.compute_refine_correction(
                routing_labels,
                rel_feat=rel_feat, union_feat=union_features,
                sub_vis=pair_sub_vis, obj_vis=pair_obj_vis,
                sub_sem=pair_sub_sem, obj_sem=pair_obj_sem,
                geometry=pair_geometry,
                return_fused=False,
            )
            fused_per_cluster = None

        final_logits = base_logits + self.refine_alpha * refine_correction

        entity_dists = entity_dists.split(num_objs, dim=0)
        rel_dists_split = final_logits.split(num_rels, dim=0)

        if self.training:
            # ════════════════════════════════════════════════════════════════
            # 主干 prototype 正则化 (与 v1 一致)
            # ════════════════════════════════════════════════════════════════
            l21 = self.get_l2loss(predicate_proto_norm)
            add_losses.update({"l21_loss": l21 * 0.8})

            l21_vis_0 = self.get_l2loss(predicate_proto_vis_norm_0)
            add_losses.update({"l21_vis_loss_0": l21_vis_0 * 0.2})
            l21_vis_1 = self.get_l2loss(predicate_proto_vis_norm_1)
            add_losses.update({"l21_vis_loss_1": l21_vis_1 * 0.2})
            l21_vis_2 = self.get_l2loss(predicate_proto_vis_norm_2)
            add_losses.update({"l21_vis_loss_2": l21_vis_2 * 0.2})
            l21_vis_3 = self.get_l2loss(predicate_proto_vis_norm_3)
            add_losses.update({"l21_vis_loss_3": l21_vis_3 * 0.2})
            l21_vis_4 = self.get_l2loss(predicate_proto_vis_norm_4)
            add_losses.update({"l21_vis_loss_4": l21_vis_4 * 0.2})

            dist_loss = self.get_guss_loss(predicate_proto)
            add_losses.update({"dist_loss2": dist_loss * 0.8})
            dist_loss_vis_0 = self.get_guss_loss(vis_proto_0_w)
            add_losses.update({"dist_loss_vis_0": dist_loss_vis_0 * 0.2})
            dist_loss_vis_1 = self.get_guss_loss(vis_proto_1_w)
            add_losses.update({"dist_loss_vis_1": dist_loss_vis_1 * 0.2})
            dist_loss_vis_2 = self.get_guss_loss(vis_proto_2_w)
            add_losses.update({"dist_loss_vis_2": dist_loss_vis_2 * 0.2})
            dist_loss_vis_3 = self.get_guss_loss(vis_proto_3_w)
            add_losses.update({"dist_loss_vis_3": dist_loss_vis_3 * 0.2})
            dist_loss_vis_4 = self.get_guss_loss(vis_proto_4_w)
            add_losses.update({"dist_loss_vis_4": dist_loss_vis_4 * 0.2})

            # 语义 loss_dis (rep ↔ proto triplet)
            rel_labels_cat = cat(rel_labels, dim=0)
            gamma1 = 1.0
            rel_rep_expand = rel_rep.unsqueeze(dim=1).expand(-1, 51, -1)
            predicate_proto_expand = predicate_proto.unsqueeze(dim=0).expand(rel_labels_cat.size(0), -1, -1)
            distance_set = (rel_rep_expand - predicate_proto_expand).norm(dim=2) ** 2

            mask_neg = torch.ones(rel_labels_cat.size(0), 51).cuda()
            mask_neg[torch.arange(rel_labels_cat.size(0)), rel_labels_cat] = 0
            distance_set_neg = distance_set * mask_neg
            distance_set_pos = distance_set[torch.arange(rel_labels_cat.size(0)), rel_labels_cat]

            sorted_distance_set_neg, _ = torch.sort(distance_set_neg, dim=1)
            topK_sorted_distance_set_neg = sorted_distance_set_neg[:, :11].sum(dim=1) / 10
            loss_sum = torch.max(torch.zeros(rel_labels_cat.size(0)).cuda(),
                                 distance_set_pos - topK_sorted_distance_set_neg + gamma1).mean()
            add_losses.update({"loss_dis": loss_sum * 0.8})

            # 视觉 loss_dis (5 套)
            loss_dis_vis_0 = self.get_rep_pro_loss(vis_rep_0, predicate_proto_vis_0, rel_labels_cat)
            add_losses.update({"loss_dis_vis_0": loss_dis_vis_0 * 0.2})
            loss_dis_vis_1 = self.get_rep_pro_loss(vis_rep_1, predicate_proto_vis_1, rel_labels_cat)
            add_losses.update({"loss_dis_vis_1": loss_dis_vis_1 * 0.2})
            loss_dis_vis_2 = self.get_rep_pro_loss(vis_rep_2, predicate_proto_vis_2, rel_labels_cat)
            add_losses.update({"loss_dis_vis_2": loss_dis_vis_2 * 0.2})
            loss_dis_vis_3 = self.get_rep_pro_loss(vis_rep_3, predicate_proto_vis_3, rel_labels_cat)
            add_losses.update({"loss_dis_vis_3": loss_dis_vis_3 * 0.2})
            loss_dis_vis_4 = self.get_rep_pro_loss(vis_rep_4, predicate_proto_vis_4, rel_labels_cat)
            add_losses.update({"loss_dis_vis_4": loss_dis_vis_4 * 0.2})

            # ════════════════════════════════════════════════════════════════
            # [Fix 1] Cluster Refinement Loss
            #   监督 mask: 只要 GT 落在任何混淆簇就监督 (不再要求 base 已对)
            #   注意: 训练时 routing 用 GT, 所以 refine_correction[i] 对应的 cluster
            #         恰好是 gt_cluster[i],无需额外 mask 簇
            # ════════════════════════════════════════════════════════════════
            with torch.no_grad():
                gt_cluster = self.predicate_to_cluster[rel_labels_cat]
                supervised_mask = (gt_cluster != -1)

            if supervised_mask.any():
                sup_logits = refine_correction[supervised_mask]      # [n_sup, 51]
                sup_labels = rel_labels_cat[supervised_mask]
                sup_clusters = gt_cluster[supervised_mask]

                # 构造每个样本的 cluster mask (只在该样本所属 cluster 的 members 上算 softmax)
                cluster_keep_mask = torch.zeros_like(sup_logits, dtype=torch.bool)
                for c in range(self.num_clusters):
                    in_c = (sup_clusters == c)
                    if not in_c.any():
                        continue
                    members = self.get_cluster_members(c)
                    rows = torch.where(in_c)[0]
                    cluster_keep_mask[rows[:, None], members[None, :]] = True

                very_neg = sup_logits.min().detach() - 8.0
                masked_logits = torch.where(cluster_keep_mask, sup_logits, very_neg)

                log_prob = F.log_softmax(masked_logits, dim=-1)
                gt_log_prob = log_prob[torch.arange(sup_labels.size(0)), sup_labels]
                gt_log_prob = torch.nan_to_num(gt_log_prob, nan=0.0, posinf=0.0, neginf=-15.0)
                gt_log_prob = torch.clamp(gt_log_prob, min=-10.0)

                loss_refine = -gt_log_prob.mean()
                add_losses["loss_cluster_refine"] = loss_refine * self.refine_loss_weight
            else:
                add_losses["loss_cluster_refine"] = rel_feat.new_zeros(())

            # ════════════════════════════════════════════════════════════════
            # [Fix 3a] Anchor Loss (InfoNCE on fused_emb ↔ cluster.protos)
            #   让 wearing 样本的 fused_emb 拉向 wearing proto, 推离 wears proto
            #   这是数据驱动的方向分离, 真正的判别轴学习
            # ════════════════════════════════════════════════════════════════
            anchor_losses = []
            if fused_per_cluster is not None:
                for c, (rows, fused_emb) in fused_per_cluster.items():
                    sample_labels = rel_labels_cat[rows]                # [n_in_c]
                    members = self.get_cluster_members(c)               # [|c|]
                    # 真值谓词在簇内的 index
                    # members 形如 tensor([48,49]), sample_labels[i] ∈ members
                    # 用 broadcast 找 index
                    target_idx = (sample_labels.unsqueeze(1) == members.unsqueeze(0)).long().argmax(dim=1)

                    proto_c = F.normalize(self.cluster_heads[c].protos, p=2, dim=-1)
                    scale = self.cluster_heads[c].logit_scale.clamp(max=4.0).exp()
                    logits_anchor = (fused_emb @ proto_c.t()) * scale   # [n_in_c, |c|]
                    anchor_loss = F.cross_entropy(logits_anchor, target_idx)
                    anchor_losses.append(anchor_loss)
            if len(anchor_losses) > 0:
                add_losses["loss_anchor"] = (sum(anchor_losses) / len(anchor_losses)) * self.anchor_loss_weight
            else:
                add_losses["loss_anchor"] = rel_feat.new_zeros(())

            # ════════════════════════════════════════════════════════════════
            # [Fix 3b] Cluster proto push-apart (margin hinge, 仅防塌缩)
            # ════════════════════════════════════════════════════════════════
            for c in range(self.num_clusters):
                proto_c = self.cluster_heads[c].protos
                n = proto_c.size(0)
                if n < 2:
                    continue
                proto_c_norm = F.normalize(proto_c, p=2, dim=-1)
                sim = proto_c_norm @ proto_c_norm.t()
                eye = torch.eye(n, device=proto_c.device)
                mask = (1 - eye).bool()
                off_diag = sim[mask]
                # cos_sim > margin 时才有梯度,否则原型已经分开
                push_loss = F.relu(off_diag - self.cluster_push_margin).mean()
                add_losses[f"l21_cluster_{c}"] = push_loss * self.cluster_push_weight

        return entity_dists, rel_dists_split, add_losses

    def get_rep_pro_loss(self, rep, proto, ans):
        gamma1_2 = 1.0
        rep_expand = rep.unsqueeze(dim=1).expand(-1, 51, -1)
        proto_expand = proto.unsqueeze(dim=0).expand(ans.size(0), -1, -1)
        distance_set = (rep_expand - proto_expand).norm(dim=2) ** 2
        mask_neg_2 = torch.ones(ans.size(0), 51).cuda()
        mask_neg_2[torch.arange(ans.size(0)), ans] = 0
        distance_set_neg = distance_set * mask_neg_2
        distance_set_pos = distance_set[torch.arange(ans.size(0)), ans]

        sorted_distance_set_neg, _ = torch.sort(distance_set_neg, dim=1)
        topK_sorted_distance_set_neg = sorted_distance_set_neg[:, :11].sum(dim=1) / 10
        loss_sum_2 = torch.max(torch.zeros(ans.size(0)).cuda(),
                               distance_set_pos - topK_sorted_distance_set_neg + gamma1_2).mean()
        return loss_sum_2

    def get_guss_loss(self, tensor):
        gamma2 = 7.0
        predicate_proto_a = tensor.unsqueeze(dim=1).expand(-1, 51, -1)
        predicate_proto_b = tensor.detach().unsqueeze(dim=0).expand(51, -1, -1)
        proto_dis_mat = (predicate_proto_a - predicate_proto_b).norm(dim=2) ** 2
        sorted_proto_dis_mat, _ = torch.sort(proto_dis_mat, dim=1)
        topK_proto_dis = sorted_proto_dis_mat[:, :2].sum(dim=1) / 1
        dist_loss = torch.max(torch.zeros(51).cuda(), -topK_proto_dis + gamma2).mean()
        return dist_loss

    def get_l2loss(self, tensor):
        target_rpredicate_proto_norm = tensor.clone().detach()
        simil_mat = tensor @ target_rpredicate_proto_norm.t()
        l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (51 * 51)
        return l21

    def refine_obj_labels(self, roi_features, proposals):
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None
        pos_embed = self.pos_embed(encode_box_info(proposals))

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = obj_labels.long()
            obj_embed = self.obj_embed1(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        assert proposals[0].mode == 'xyxy'

        pos_embed = self.pos_embed(encode_box_info(proposals))
        num_objs = [len(p) for p in proposals]
        obj_pre_rep_for_pred = self.lin_obj_cyx(cat([roi_features, obj_embed, pos_embed], -1))

        if self.mode == 'predcls':
            obj_labels = obj_labels.long()
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_classes)
        else:
            obj_dists = self.out_obj(obj_pre_rep_for_pred)
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs).long()
            else:
                obj_preds = (obj_dists[:, 1:].max(1)[1] + 1).long()

        return obj_dists, obj_preds

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for i in range(len(num_objs)):
            is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh

            out_dists_sampled = F.softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1

            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for i in range(num_objs[i]):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind, :, cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0

            obj_preds.append(out_label.long())
        obj_preds = torch.cat(obj_preds, dim=0)
        return obj_preds


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

        for i, layer in enumerate(self.layers):
            layer_init(layer, xavier=True)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def fusion_func(x, y):
    return F.relu(x + y) - (x - y) ** 2


@registry.ROI_RELATION_PREDICTOR.register("HetSGG_Predictor")
class HetSGG_Predictor(nn.Module):
    def __init__(self, config, in_channels):
        super(HetSGG_Predictor, self).__init__()
        self.num_obj_cls = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.use_bias = cfg.MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS
        self.rel_aware_model_on = config.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON
        self.use_obj_recls_logits = config.MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS

        self.rel_aware_loss_eval = None

        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            self.mode = "predcls" if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL else "sgcls"
        else:
            self.mode = "sgdet"

        self.obj_recls_logits_update_manner = (
            cfg.MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_MANNER
        )

        self.n_reltypes = cfg.MODEL.ROI_RELATION_HEAD.HETSGG.NUM_RELATION
        self.n_dim = cfg.MODEL.ROI_RELATION_HEAD.HETSGG.H_DIM
        self.n_ntypes = int(sqrt(self.n_reltypes))
        self.obj2rtype = {(i, j): self.n_ntypes * j + i for j in range(self.n_ntypes) for i in range(self.n_ntypes)}

        self.rel_classifier = build_classifier(self.num_rel_cls, self.num_rel_cls)
        self.obj_classifier = build_classifier(self.num_obj_cls, self.num_obj_cls)

        self.context_layer = HetSGG(config, in_channels)

        if self.use_bias:
            statistics = get_dataset_statistics(config)
            self.freq_bias = FrequencyBias(config, statistics)
            self.freq_lambda = nn.Parameter(torch.Tensor([1.0]), requires_grad=False)

    def init_classifier_weight(self):
        self.obj_classifier.reset_parameters()
        for i in range(self.n_reltypes):
            self.rel_classifier[i].reset_parameters()

    def forward(self, inst_proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None,
                is_training=True):
        obj_feats, rel_feats = self.context_layer(roi_features, union_features, inst_proposals, rel_pair_idxs,
                                                   rel_binarys, logger, is_training)

        if self.mode == "predcls":
            obj_labels = cat([proposal.get_field("labels") for proposal in inst_proposals], dim=0)
            refined_obj_logits = to_onehot(obj_labels, self.num_obj_cls)
        else:
            refined_obj_logits = self.obj_classifier(obj_feats)

        rel_cls_logits = self.rel_classifier(rel_feats)

        num_objs = [len(b) for b in inst_proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)
        obj_pred_logits = cat([each_prop.get_field("predict_logits") for each_prop in inst_proposals], dim=0)

        if self.use_obj_recls_logits:
            boxes_per_cls = cat([proposal.get_field("boxes_per_cls") for proposal in inst_proposals], dim=0)
            if self.obj_recls_logits_update_manner == "add":
                obj_pred_logits = refined_obj_logits + obj_pred_logits
            if self.obj_recls_logits_update_manner == "replace":
                obj_pred_logits = refined_obj_logits
            refined_obj_pred_labels = obj_prediction_nms(boxes_per_cls, obj_pred_logits, nms_thresh=0.5)
            obj_pred_labels = refined_obj_pred_labels
        else:
            obj_pred_labels = cat([each_prop.get_field("pred_labels") for each_prop in inst_proposals], dim=0)

        if self.use_bias:
            obj_pred_labels = obj_pred_labels.split(num_objs, dim=0)
            pair_preds = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_pred_labels):
                pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
            pair_pred = cat(pair_preds, dim=0)
            rel_cls_logits = rel_cls_logits + self.freq_bias.index_with_labels(pair_pred.long())

        obj_pred_logits = obj_pred_logits.split(num_objs, dim=0)
        rel_cls_logits = rel_cls_logits.split(num_rels, dim=0)
        add_losses = {}

        return obj_pred_logits, rel_cls_logits, add_losses


def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)


@registry.ROI_RELATION_PREDICTOR.register("HetSGGplus_Predictor")
class HetSGGplus_Predictor(nn.Module):
    def __init__(self, config, in_channels):
        super(HetSGGplus_Predictor, self).__init__()
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS

        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = "predcls"
            else:
                self.mode = "sgcls"
        else:
            self.mode = "sgdet"

        assert in_channels is not None
        self.pooling_dim = cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.input_dim = in_channels
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.GRAPH_HIDDEN_DIM

        self.split_context_model4inst_rel = (
            config.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.SPLIT_GRAPH4OBJ_REL
        )
        if self.split_context_model4inst_rel:
            self.obj_context_layer = HetSGGplus_Context(
                config, self.input_dim, hidden_dim=self.hidden_dim,
                num_iter=config.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.GRAPH_ITERATION_NUM,
            )
            self.rel_context_layer = HetSGGplus_Context(
                config, self.input_dim, hidden_dim=self.hidden_dim,
                num_iter=config.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.GRAPH_ITERATION_NUM,
            )
        else:
            self.context_layer = HetSGGplus_Context(
                config, self.input_dim, hidden_dim=self.hidden_dim,
                num_iter=config.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.GRAPH_ITERATION_NUM,
            )

        self.rel_feature_type = config.MODEL.ROI_RELATION_HEAD.EDGE_FEATURES_REPRESENTATION

        self.use_obj_recls_logits = config.MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS
        self.obj_recls_logits_update_manner = (
            config.MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_MANNER
        )
        assert self.obj_recls_logits_update_manner in ["replace", "add"]

        self.rel_classifier = build_classifier(self.hidden_dim, self.num_rel_cls)
        self.obj_classifier = build_classifier(self.hidden_dim, self.num_obj_cls)

        self.rel_aware_model_on = config.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON

        if self.rel_aware_model_on:
            self.rel_aware_loss_eval = RelAwareLoss(config)

        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        if self.use_bias:
            statistics = get_dataset_statistics(config)
            self.freq_bias = FrequencyBias(config, statistics)
            self.freq_lambda = nn.Parameter(
                torch.Tensor([1.0]), requires_grad=False
            )

        self.init_classifier_weight()
        self.forward_time = 0

    def init_classifier_weight(self):
        self.rel_classifier.reset_parameters()
        self.obj_classifier.reset_parameters()

    def start_preclser_relpn_pretrain(self):
        self.context_layer.set_pretrain_pre_clser_mode()

    def end_preclser_relpn_pretrain(self):
        self.context_layer.set_pretrain_pre_clser_mode(False)

    def forward(self, inst_proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features,
                logger=None, is_training=None):
        obj_feats, rel_feats, pre_cls_logits, relatedness = self.context_layer(
            roi_features, union_features, inst_proposals, rel_pair_idxs, rel_binarys, logger
        )

        if relatedness is not None:
            for idx, prop in enumerate(inst_proposals):
                prop.add_field("relness_mat", relatedness[idx])

        if self.mode == "predcls":
            obj_labels = cat([proposal.get_field("labels") for proposal in inst_proposals], dim=0)
            refined_obj_logits = to_onehot(obj_labels, self.num_obj_cls)
        else:
            refined_obj_logits = self.obj_classifier(obj_feats)

        rel_cls_logits = self.rel_classifier(rel_feats)

        num_objs = [len(b) for b in inst_proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)
        obj_pred_logits = cat([each_prop.get_field("predict_logits") for each_prop in inst_proposals], dim=0)

        if self.use_obj_recls_logits:
            if (self.mode == "sgdet") | (self.mode == "sgcls"):
                boxes_per_cls = cat([proposal.get_field("boxes_per_cls") for proposal in inst_proposals], dim=0)
                if self.obj_recls_logits_update_manner == "add":
                    obj_pred_logits = refined_obj_logits + obj_pred_logits
                if self.obj_recls_logits_update_manner == "replace":
                    obj_pred_logits = refined_obj_logits
                refined_obj_pred_labels = obj_prediction_nms(boxes_per_cls, obj_pred_logits, nms_thresh=0.5)
                obj_pred_labels = refined_obj_pred_labels
        else:
            obj_pred_labels = cat([each_prop.get_field("pred_labels") for each_prop in inst_proposals], dim=0)

        if self.use_bias:
            obj_pred_labels = obj_pred_labels.split(num_objs, dim=0)
            pair_preds = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_pred_labels):
                pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
            pair_pred = cat(pair_preds, dim=0)
            rel_cls_logits = (
                    rel_cls_logits + self.freq_lambda * self.freq_bias.index_with_labels(pair_pred.long())
            )

        obj_pred_logits = obj_pred_logits.split(num_objs, dim=0)
        rel_cls_logits = rel_cls_logits.split(num_rels, dim=0)

        add_losses = {}
        if pre_cls_logits is not None and self.training:
            rel_labels = cat(rel_labels, dim=0)
            for iters, each_iter_logit in enumerate(pre_cls_logits):
                if len(squeeze_tensor(torch.nonzero(rel_labels != -1))) == 0:
                    loss_rel_pre_cls = None
                else:
                    loss_rel_pre_cls = self.rel_aware_loss_eval(each_iter_logit, rel_labels)
                add_losses[f"pre_rel_classify_loss_iter-{iters}"] = loss_rel_pre_cls

        return obj_pred_logits, rel_cls_logits, add_losses

