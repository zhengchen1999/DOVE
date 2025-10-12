import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _pair, _single
import math

import torchvision
import warnings
from einops import rearrange

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)

def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

class ConvResidualBlocks(nn.Module):
    """Conv and residual block used in BasicVSR.

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch))

    def forward(self, fea):
        return self.main(fea)


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x), indexing='ij')
    grid = torch.stack((grid_x, grid_y), 2)  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    vgrid_scaled = vgrid_scaled.to(x)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    return output

def length_sq(x):
    return torch.sum(torch.square(x), dim=1, keepdim=True)

def fbConsistencyCheck(flow_fw, flow_bw, alpha1=0.01, alpha2=0.5): 
    flow_bw_warped = flow_warp(flow_bw, flow_fw.permute(0, 2, 3, 1))  # wb(wf(x))
    flow_diff_fw = flow_fw + flow_bw_warped  # wf + wb(wf(x))

    mag_sq_fw = length_sq(flow_fw) + length_sq(flow_bw_warped)  # |wf| + |wb(wf(x))|
    occ_thresh_fw = alpha1 * mag_sq_fw + alpha2

    # fb_valid_fw = (length_sq(flow_diff_fw) < occ_thresh_fw).float()
    fb_valid_fw = (length_sq(flow_diff_fw) < occ_thresh_fw).to(flow_fw)
    return fb_valid_fw

class EmptyPropagation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feats_in, flows_forward, flows_backward):
        return feats_in

class Propagation(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels = 256,
        max_residue_magnitude = 10,
        num_blocks = 2,
        learnable=True,
    ):
        super().__init__()

        self.learnable = learnable
        self.module = ['backward_prop', 'forward_prop']
        if self.learnable:
            if mid_channels != in_channels:
                self.input_layer = nn.Conv3d(in_channels, mid_channels, kernel_size=(3,1,1), stride=(1,1,1), padding=(1,0,0))
                self.output_layer = nn.Conv3d(mid_channels, in_channels, kernel_size=(3,1,1), stride=(1,1,1), padding=(1,0,0))

            self.mid_channels = mid_channels
            # propagation branches
            self.deform_align = nn.ModuleDict()
            self.backbone = nn.ModuleDict()
            for i, module in enumerate(self.module):
                if torch.cuda.is_available():
                    self.deform_align[module] = DeformableAlignment(
                        mid_channels,
                        mid_channels,
                        3,
                        padding=1,
                        deformable_groups=16,
                        max_residue_magnitude=max_residue_magnitude)
                self.backbone[module] = ConvResidualBlocks(2 * mid_channels, mid_channels, num_blocks)

            self.fuse = ConvResidualBlocks(3 * mid_channels, mid_channels, 2)


    def forward(self, x, flows_forward, flows_backward, interpolation='bilinear', mode='fuse', fuse_scale=0.5,
                alpha1=0.01, alpha2=0.5):
        """
        x shape : [b, c, t, h, w]
        return [b, c, t, h, w]
        """

        # For backward warping
        # pred_flows_forward for backward feature propagation
        # pred_flows_backward for forward feature propagation

        b, c, t, h, w = x.shape
        w_f = flows_forward.shape[-1]
        s = 1.0*w/w_f
        flows_forward = F.interpolate(flows_forward, (t-1, h, w), mode='area') * s
        flows_backward = F.interpolate(flows_backward, (t-1, h, w), mode='area') * s
        x_orig = x.clone()
        if hasattr(self, "input_layer"):
            x = self.input_layer(x)

        feats = {}
        feats['input'] = [x[:, :, i, :, :] for i in range(0, t)]

        cache_list = ['input'] +  self.module

        for p_i, module_name in enumerate(self.module):
            feats[module_name] = []

            if 'backward' in module_name:
                frame_idx = range(0, t)
                frame_idx = frame_idx[::-1]
                flow_idx = frame_idx
                flows_for_prop = flows_forward
                flows_for_check = flows_backward
            else:
                frame_idx = range(0, t)
                flow_idx = range(-1, t - 1)
                flows_for_prop = flows_backward
                flows_for_check = flows_forward

            for i, idx in enumerate(frame_idx):
                feat_current = feats[cache_list[p_i]][idx]

                if i == 0:
                    feat_prop = feat_current
                else:
                    flow_prop = flows_for_prop[:, :, flow_idx[i], :, :]
                    flow_check = flows_for_check[:, :, flow_idx[i], :, :]
                    flow_vaild_mask = fbConsistencyCheck(flow_prop, flow_check, alpha1, alpha2)

                    feat_warped = flow_warp(feat_prop, flow_prop.permute(0, 2, 3, 1), interpolation)

                    if self.learnable:
                        cond = torch.cat([feat_current, feat_warped, flow_prop, flow_vaild_mask], dim=1)
                        feat_prop = self.deform_align[module_name](feat_prop, cond, flow_prop)
                    else:
                        if mode == 'fuse': # choice 1: blur
                            feat_warped = feat_warped * fuse_scale + feat_current * (1-fuse_scale)
                        elif mode == 'copy': # choice 2: alignment
                            feat_warped = feat_warped 
                        feat_prop = flow_vaild_mask * feat_warped + (1-flow_vaild_mask) * feat_current

                # refine
                if self.learnable:
                    feat = torch.cat([feat_current, feat_prop], dim=1)
                    feat_prop = feat_prop + self.backbone[module_name](feat)
                    # feat_prop = self.backbone[module_name](feat_prop)

                feats[module_name].append(feat_prop)

            # end for
            if 'backward' in module_name:
                feats[module_name] = feats[module_name][::-1]

        outputs_b = torch.stack(feats['backward_prop'], dim=2) # bcthw
        outputs_f = torch.stack(feats['forward_prop'], dim=2)  # bcthw

        if self.learnable:
            cat_feat = rearrange(torch.cat([x, outputs_b, outputs_f], dim=1), "b c t h w -> (b t) c h w").contiguous()
            fuse_feat = self.fuse(cat_feat)
            if hasattr(self, "out_layer"):
                fuse_feat = self.output_layer(fuse_feat)
            fuse_feat = rearrange(fuse_feat, "(b t) c h w -> b c t h w", t=t).contiguous()
            outputs = fuse_feat + x_orig
        else:
            outputs = outputs_f
        
        return outputs
               

class ModulatedDeformConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 bias=True):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, x, offset, mask):
        pass


class DeformableAlignment(ModulatedDeformConv):
    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 3)

        super(DeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(2*self.out_channels + 2 + 1, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deformable_groups, 3, 1, 1),
        )
        self.init_offset()

    def init_offset(self):
        def _constant_init(module, val, bias=0):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.constant_(module.weight, val)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, bias)

        _constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, cond_feat, flow):        
        out = self.conv_offset(cond_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset = offset + flow.flip(1).repeat(1, offset.size(1) // 2, 1, 1)

        # mask
        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, 
                                             self.stride, self.padding,
                                             self.dilation, mask)

def optical_flow_process(x, flows_forward, flows_backward, 
                            interpolation='bilinear', mode='fuse', fuse_scale=0.5,
                            alpha1=0.01, alpha2=0.5):
    """
    x shape : [b, c, t, h, w]
    return [b, c, t, h, w]
    """
    b, c, t, h, w = x.shape
    w_f = flows_forward.shape[-1]
    s = 1.0 * w / w_f
    flows_forward = F.interpolate(flows_forward, (t-1, h, w), mode='trilinear') * s
    flows_backward = F.interpolate(flows_backward, (t-1, h, w), mode='trilinear') * s

    feats = {}
    feats['input'] = [x[:, :, i, :, :] for i in range(0, t)]

    _module = ['backward_prop', 'forward_prop']

    cache_list = ['input'] +  _module

    for p_i, module_name in enumerate(_module):
        feats[module_name] = []
        if 'backward' in module_name:
            frame_idx = range(0, t)
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx
            flows_for_prop = flows_forward
            flows_for_check = flows_backward
        else:
            frame_idx = range(0, t)
            flow_idx = range(-1, t - 1)
            flows_for_prop = flows_backward
            flows_for_check = flows_forward

        for i, idx in enumerate(frame_idx):
            feat_current = feats[cache_list[p_i]][idx]

            if i == 0:
                feat_prop = feat_current
            else:
                flow_prop = flows_for_prop[:, :, flow_idx[i], :, :]
                flow_check = flows_for_check[:, :, flow_idx[i], :, :]
                flow_vaild_mask = fbConsistencyCheck(flow_prop, flow_check, alpha1, alpha2)
                # print('flow_vaild_mask', flow_vaild_mask.shape, torch.sum(flow_vaild_mask == 1).item())

                feat_warped = flow_warp(feat_prop, flow_prop.permute(0, 2, 3, 1), interpolation)
                if mode == 'fuse': # choice 1: blur
                    feat_warped = feat_warped * fuse_scale + feat_current * (1-fuse_scale)
                elif mode == 'copy': # choice 2: alignment
                    feat_warped = feat_warped 
                feat_prop = flow_vaild_mask * feat_warped + (1-flow_vaild_mask) * feat_current

            feats[module_name].append(feat_prop)

        # end for
        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

    # outputs_b = torch.stack(feats['backward_prop'], dim=2) # bcthw
    outputs_f = torch.stack(feats['forward_prop'], dim=2)  # bcthw

    outputs = outputs_f
    
    return outputs

def optical_flow_feature(x, flows_forward, flows_backward, 
                            interpolation='bilinear', mode='copy', fuse_scale=0.5,
                            alpha1=0.01, alpha2=0.5):
    """
    x shape : [b, c, t, h, w]
    return [b, c, t, h, w]
    """
    b, c, t, h, w = x.shape
    w_f = flows_forward.shape[-1]
    s = 1.0 * w / w_f
    flows_forward = F.interpolate(flows_forward, (t-1, h, w), mode='trilinear') * s
    flows_backward = F.interpolate(flows_backward, (t-1, h, w), mode='trilinear') * s

    feats = {}
    feats['input'] = [x[:, :, i, :, :] for i in range(0, t)]

    _module = ['backward_prop', 'forward_prop']

    for p_i, module_name in enumerate(_module):
        feats[module_name] = []
        if 'backward' in module_name:
            frame_idx = range(0, t)
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx
            flows_for_prop = flows_forward
            flows_for_check = flows_backward
        else:
            frame_idx = range(0, t)
            flow_idx = range(-1, t - 1)
            flows_for_prop = flows_backward
            flows_for_check = flows_forward

        for i, idx in enumerate(frame_idx):
            feat_current = feats['input'][idx]

            if i == 0:
                feat_prop = feat_current
            else:
                flow_prop = flows_for_prop[:, :, flow_idx[i], :, :]
                flow_check = flows_for_check[:, :, flow_idx[i], :, :]
                flow_vaild_mask = fbConsistencyCheck(flow_prop, flow_check, alpha1, alpha2)
                # print('flow_vaild_mask', flow_vaild_mask.shape, torch.sum(flow_vaild_mask == 1).item())

                feat_warped = flow_warp(feat_prop, flow_prop.permute(0, 2, 3, 1), interpolation)
                if mode == 'fuse': # choice 1: blur
                    feat_warped = feat_warped * fuse_scale + feat_current * (1-fuse_scale)
                elif mode == 'copy': # choice 2: alignment
                    feat_warped = feat_warped 
                feat_prop = flow_vaild_mask * feat_warped + (1-flow_vaild_mask) * feat_current

            feats[module_name].append(feat_prop)

        # end for
        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

    outputs_b = torch.stack(feats['backward_prop'], dim=2) # bcthw
    outputs_f = torch.stack(feats['forward_prop'], dim=2)  # bcthw
    
    return outputs_b, outputs_f


class Flow_ResBlock3D(nn.Module):
    def __init__(self, cin, cout, num_groups=8):
        super().__init__()
        self.cin = cin
        self.cout = cout

        self.conv1 = nn.Sequential(
            nn.GroupNorm(num_groups, cin),
            nn.SiLU(),
            nn.Conv3d(cin, cout, kernel_size=3, stride=1, padding=1),
        )
        
        self.conv2 = nn.Sequential(
            nn.GroupNorm(num_groups, cout),
            nn.SiLU(),
            nn.Conv3d(cout, cout, kernel_size=3, stride=1, padding=1),
        )

        if cin != cout:
            self.shortcut = nn.Sequential(
                nn.Conv3d(cin, cout, kernel_size=1, stride=1, padding=0),
                nn.GroupNorm(num_groups, cout)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # x: (B, C, F, H, W)
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        
        out = self.conv2(out)
        
        out = out + identity
        
        return out  # (B, Cout, F, H, W)


class OpticalFlowFusionModule(nn.Module):
    """
    Optical Flow Feature Fusion Module
    Fuses original features with flow-warped features to enhance temporal coherence
    """
    def __init__(
        self, 
        latent_channels,
        mid_channels=None,
        num_res_blocks=2, 
        num_groups=8, 
        final_zero_init=True
    ):
        super().__init__()
        
        self.latent_channels = latent_channels
        self.mid_channels = mid_channels or latent_channels * 2
        
        # Input channels: original + backward flow + forward flow
        input_channels = latent_channels * 3

        # Initial projection to intermediate feature space
        self.initial_conv = nn.Sequential(
            nn.Conv3d(input_channels, self.mid_channels, kernel_size=1, padding=0),
            nn.GroupNorm(num_groups, self.mid_channels),
            nn.SiLU()
        )
    
        # Residual blocks for feature processing
        self.res_blocks = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.res_blocks.append(
                Flow_ResBlock3D(cin=self.mid_channels, cout=self.mid_channels, num_groups=num_groups)
            )

        # Final projection back to original feature dimensions
        self.final_proj = nn.Conv3d(self.mid_channels, latent_channels, kernel_size=1, padding=0)
        
        self._init_weights(final_zero_init)
        
    def _init_weights(self, final_zero_init=True):
        """
        Initialize weights - only the final projection layer is zero-initialized
        to ensure the module has no effect at the beginning of training
        """
        # Standard initialization for initial convolution and residual blocks
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if m is self.final_proj and final_zero_init:
                    # Zero initialization for final layer
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    # Kaiming initialization for other layers
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(self, x, backward_flow_feature, forward_flow_feature):
        """
        Forward pass for optical flow fusion
        
        Args:
            x (Tensor): Original features [B, C, T, H, W]
            backward_flow_feature (Tensor): Backward warped features [B, C, T, H, W]
            forward_flow_feature (Tensor): Forward warped features [B, C, T, H, W]
            
        Returns:
            Tensor: Enhanced features (original + residual) [B, C, T, H, W]
        """
        # Concatenate features along channel dimension
        features = torch.cat([x, backward_flow_feature, forward_flow_feature], dim=1)
        
        # Process through fusion network
        out = self.initial_conv(features)
        
        for res_block in self.res_blocks:
            out = res_block(out)
        
        # Project back to original feature dimension (as residual)
        residual = self.final_proj(out)
        
        # Add residual to original features
        return x + residual