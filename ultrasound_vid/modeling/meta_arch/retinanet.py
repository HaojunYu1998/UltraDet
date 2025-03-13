from itertools import chain
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling import RetinaNet


@META_ARCH_REGISTRY.register()
class SingleFrameRetinaNet(RetinaNet):

    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, batched_inputs):
        if self.training:
            batched_inputs = list(chain.from_iterable(batched_inputs))
            losses = super().forward(batched_inputs)
            return losses
        else:
            batched_inputs = [batched_inputs]
            results = super().forward(batched_inputs)
            assert len(results) == 1, f"{len(results)} != 1 for inference"
            # print(len(results), type(results), type(results[0]), results[0])
            # print(results[0]["instances"])
            return results[0]["instances"]

    def reset(self):
        pass