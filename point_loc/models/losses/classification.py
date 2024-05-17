from mmpretrain.models.losses import CrossEntropyLoss as CEL
from point_loc.registry import MODELS

@MODELS.register_module()
class CrossEntropyLoss(CEL):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
