import math
import torch
import torch.nn as nn
from collections import OrderedDict


class PrototypeDistanceScheduler(nn.Module):
    """
    Prototype-Distance-Guided Progressive Re-weighting Scheduler

    独立训练调度模块, 不修改 PENET 内部逻辑。

    核心思路:
      训练初期 → 原型尚未分离 → λ 接近 λ_0 → CB 重加权强
      训练后期 → 尾类原型已分离 → λ 衰减 → CB 重加权弱

    调度公式:
      d_tail = mean(1 - max_neighbor_cosine_sim)  across all tail class prototypes
      λ      = λ_0 * exp(-alpha * d_tail)

    优势:
      - 按实际表示学习进度自适应, 而非固定 epoch 数
      - 利用 PE-Net 原生 prototype 机制, 零额外开销
      - 仅在更新 step 时计算距离, 其余 step 直接返回缓存值
    """

    def __init__(self, cfg):
        super().__init__()

        self.lambda_0 = cfg.MODEL.ROI_RELATION_HEAD.CURRICULUM.LAMBDA_0
        self.alpha = cfg.MODEL.ROI_RELATION_HEAD.CURRICULUM.ALPHA
        self.update_every = cfg.MODEL.ROI_RELATION_HEAD.CURRICULUM.UPDATE_EVERY
        self.warmup_iters = cfg.MODEL.ROI_RELATION_HEAD.CURRICULUM.WARMUP_ITERS

        longtail_dict = cfg.MODEL.ROI_RELATION_HEAD.LONGTAIL_PART_DICT
        self.tail_classes = [i for i, tag in enumerate(longtail_dict) if tag == 't']

        self.register_buffer('current_lambda', torch.tensor(self.lambda_0))
        self.register_buffer('last_proto_distance', torch.tensor(0.0))

        self._step = 0
        self._num_tail = len(self.tail_classes)

    def step(self, predicate_proto_norm):
        """
        Args:
            predicate_proto_norm: [num_rel_cls, embed_dim]  L2-normalized 原型向量

        Returns:
            float: 当前的 λ 值, 用于 scale CB loss 贡献
        """
        self._step += 1

        if self._step < self.warmup_iters:
            return self.current_lambda.item()

        if self._step % self.update_every != 0 and self._step != self.warmup_iters:
            return self.current_lambda.item()

        if predicate_proto_norm is None:
            return self.current_lambda.item()

        d_tail = self._compute_tail_separation(predicate_proto_norm)

        new_lambda = self.lambda_0 * math.exp(-self.alpha * d_tail)
        new_lambda = max(new_lambda, 0.05)

        self.current_lambda.fill_(new_lambda)
        self.last_proto_distance.fill_(d_tail)

        return self.current_lambda.item()

    def _compute_tail_separation(self, proto):
        """
        计算所有尾类原型之间的平均最小类间 cosine 距离

        对于每个尾类, 找到与其 cosine 相似度最高的那个其他尾类,
        取 1-similarity 为距离, 然后对全体尾类取均值

        d_tail ∈ [0, 2], 越大说明尾类之间越容易区分
        """
        if self._num_tail <= 1:
            return 0.0

        tail_proto = proto[self.tail_classes]

        sim = tail_proto @ tail_proto.t()

        eye_mask = ~torch.eye(self._num_tail, dtype=torch.bool, device=sim.device)
        sim_inter = sim[eye_mask].view(self._num_tail, self._num_tail - 1)

        max_neighbor_sim = sim_inter.max(dim=1).values

        d_tail = (1.0 - max_neighbor_sim).mean().item()
        d_tail = max(d_tail, 0.0)

        return d_tail

    def get_state(self):
        return OrderedDict([
            ('step', self._step),
            ('num_tail_classes', self._num_tail),
            ('current_lambda', round(self.current_lambda.item(), 6)),
            ('last_proto_distance', round(self.last_proto_distance.item(), 6)),
        ])


def build_curriculum_scheduler(cfg):
    if not cfg.MODEL.ROI_RELATION_HEAD.CURRICULUM.ENABLED:
        return None
    return PrototypeDistanceScheduler(cfg)