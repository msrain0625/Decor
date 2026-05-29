from hetsgg import _C

try:
    from apex import amp
    nms = amp.float_function(_C.nms)
except (ImportError, AttributeError):
    # apex.amp 不可用，直接使用原始函数
    # 新版本的 PyTorch 使用 torch.cuda.amp 进行自动混合精度
    nms = _C.nms
