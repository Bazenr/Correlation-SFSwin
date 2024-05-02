import pytest
import torch

# from mmdet.models.backbones.swin import Divide_SKConv
# from mmdet.models.backbones.swin_double_SKNet_old import SKConv
from mmdet.models.backbones.swin import SwinTransformer

if __name__ == '__main__':
    # 测试代码用
    temp1 =  torch.ones((2, 3, 192, 192))
    temp2 =  torch.ones((2, 3, 192, 192))

    temp = torch.concat([temp1, temp2], axis=1)
    model = SwinTransformer(in_chans1=3, in_chans2=3,
                                        depths=(2, 2, 18, 2))
    outs = model(temp)
for i, out in enumerate(outs):
        print(out.shape)