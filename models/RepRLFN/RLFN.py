# -*- coding: utf-8 -*-
# Copyright 2022 ByteDance
import copy

import torch.nn as nn
from models.RepRLFN import block
import torch
import random
import numpy as np
import os

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)



class RLFN(nn.Module):
    """
    Residual Local Feature Network (RLFN)
    Model definition of RLFN in `Residual Local Feature Network for 
    Efficient Super-Resolution`
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 # feature_channels=52,
                 # feature_channels=48,
                 feature_channels=45,
                 upscale=4,
                 deploy=False):
        super(RLFN, self).__init__()

        self.conv_1 = block.conv_layer(in_channels,
                                       feature_channels,
                                       kernel_size=3)

        self.block_1 = block.RLFB(in_channels=feature_channels, act_type='lrelu', deploy=deploy)
        self.block_2 = block.RLFB(in_channels=feature_channels, act_type='lrelu', deploy=deploy)
        self.block_3 = block.RLFB(in_channels=feature_channels, act_type='lrelu', deploy=deploy)
        self.block_4 = block.RLFB(in_channels=feature_channels, act_type='lrelu', deploy=deploy)
        # self.block_5 = block.RLFB(in_channels=feature_channels, act_type='lrelu', deploy=deploy)
        # self.block_6 = block.RLFB(in_channels=in_channels, act_type='lrelu', deploy=deploy)

        self.conv_2 = block.conv_layer(feature_channels,
                                       feature_channels,
                                       kernel_size=3)

        self.upsampler = block.pixelshuffle_block(feature_channels,
                                                  out_channels,
                                                  upscale_factor=upscale)

    def forward(self, x):
        out_feature = self.conv_1(x)

        out_b1 = self.block_1(out_feature)
        out_b2 = self.block_2(out_b1)
        out_b3 = self.block_3(out_b2)
        out_b4 = self.block_4(out_b3)
        # out_b5 = self.block_5(out_b4)
        # out_b6 = self.block_6(out_b5)

        # out_low_resolution = self.conv_2(out_b6) + out_feature
        # out_low_resolution = self.conv_2(out_b5) + out_feature
        out_low_resolution = self.conv_2(out_b4) + out_feature
        output = self.upsampler(out_low_resolution)

        return output


def get_RLFN(checkpoint=None, deploy=True):
    model = RLFN(in_channels=3, out_channels=3, deploy=deploy)

    # param_key_g = 'params'
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=True)

    return model

def repvgg_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model


if __name__ == '__main__':
    seed_everything(2024)


    mynet = get_RLFN(checkpoint=None, deploy=False)
    img = torch.ones((1,3,40,40))
    out = mynet(img)
    state_dict = mynet.state_dict()
    torch.save(state_dict, 'mynet01.pth')
    # print("out: {}".format(out))
    print("size: {}".format(out.size()))
    print(out[0][0][0][:4]) # [ 0.0732, -0.0637,  0.0166, -0.1979]


    mynet = get_RLFN(checkpoint='mynet01.pth', deploy=False)
    img = torch.ones((1,3,40,40))
    out = mynet(img)
    print("size: {}".format(out.size()))
    print(out[0][0][0][:4]) # [ 0.0732, -0.0637,  0.0166, -0.1979]
    # # 模型转为部署模式保存
    deploy_model = repvgg_model_convert(mynet, save_path='mynet01_deploy.pth')


    mynet = get_RLFN(checkpoint='mynet01_deploy.pth', deploy=True)
    img = torch.ones((1,3,40,40))
    out = mynet(img)
    print("size: {}".format(out.size()))
    print(out[0][0][0][:4]) # [ 0.0732, -0.0637,  0.0166, -0.1979]


    from utils.model_summary import get_model_flops, get_model_activation
    model = get_RLFN(checkpoint=None, deploy=False)
    # model = get_RLFN(checkpoint='mynet01_deploy.pth', deploy=True)
    input_dim = (3, 256, 256)  # set the input dimension
    activations, num_conv = get_model_activation(model, input_dim)
    activations = activations / 10 ** 6
    print("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
    print("{:>16s} : {:<d}".format("#Conv2d", num_conv))
    flops = get_model_flops(model, input_dim, False)
    flops = flops / 10 ** 9
    print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))
    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    num_parameters = num_parameters / 10 ** 6
    print("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))

    # before -- #Activations : 183.1276 [M], #Conv2d : 75, #FLOPs : 29.1872 [G], #Params : 0.4612 [M]
    # after -- #Activations : 76.9593 [M], #Conv2d : 39, #FLOPs : 18.0395 [G], #Params : 0.2895 [M]

    # RLFN_Prune(baseline) -- #Activations : 80.0452 [M], #Conv2d : 39, FLOPs: 19.6953 [G], Params : 0.3172 [M]
    # valid: 26.90dB, test: 26.99dB