import functools
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

# try:
#     from codes.models.modules.DCNv2.dcn_v2 import DCN_sep
# except ImportError:
#     raise ImportError('Failed to import DCNv2 module.')

from my_upsampling import MyUpsampling 

from DCNv2.dcn_v2 import DCN_sep

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())

    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class pcd_align(nn.Module):
    def __init__(self, nf=64, groups=8):
        super(pcd_align, self).__init__()
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L3_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L2_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L2_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)
        self.L2_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        # L1: level 1, original spatial size
        self.L1_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L1_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.L1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)
        self.L1_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        # Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.cas_dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                   deformable_groups=groups)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.up1 = MyUpsampling(scale_factor=(1,1), align_corners=False)
        self.up2 = MyUpsampling(scale_factor=(1,1), align_corners=False)
        self.up3 = MyUpsampling(scale_factor=(1,1), align_corners=False)
        self.up4 = MyUpsampling(scale_factor=(1,1), align_corners=False)

    def forward(self, nbr_fea_l, ref_fea_l, return_off=False):
        # L3
        L3_offset = torch.cat([nbr_fea_l[2], ref_fea_l[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack(nbr_fea_l[2], L3_offset))
        # L2
        L2_offset = torch.cat([nbr_fea_l[1], ref_fea_l[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
        self.up1.set_scale((2,2)) 
        self.up2.set_scale((2,2)) 
        self.up3.set_scale((2,2)) 
        self.up4.set_scale((2,2)) 
        #L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L3_offset = self.up1(L3_offset)

        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        L2_fea = self.L2_dcnpack(nbr_fea_l[1], L2_offset)

        #L3_fea_up = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L3_fea_up = self.up2(L3_fea) #F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L3_fea_up], dim=1)))
        # L1
        L1_offset = torch.cat([nbr_fea_l[0], ref_fea_l[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        #L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.up3(L2_offset) 
        L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
        L1_fea = self.L1_dcnpack(nbr_fea_l[0], L1_offset)
        #L2_fea_up = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea_up = self.up4(L2_fea)
        L1_fea = self.L1_fea_conv(torch.cat([L1_fea, L2_fea_up], dim=1))
        # Cascading
        offset = torch.cat([L1_fea, ref_fea_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea = self.lrelu(self.cas_dcnpack(L1_fea, offset))

        if return_off:
            return L1_fea, L1_offset
        else:
            return L1_fea


class easy_fuse(nn.Module):
    def __init__(self, nf=64, nframes=3, has_relu=True):
        super(easy_fuse, self).__init__()
        self.has_relu = has_relu
        self.fea_fusion = nn.Conv2d(nf * nframes, nf, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, aligned_fea):
        B, N, C, H, W = aligned_fea.size()
        aligned_fea = aligned_fea.view(B, N * C, H, W)
        out = self.fea_fusion(aligned_fea)

        if self.has_relu:
            out = self.lrelu(out)

        return out


class encoder(nn.Module):
    def __init__(self, nf=64, N_RB=5):
        super(encoder, self).__init__()
        RB_f = functools.partial(ResidualBlock_noBN, nf=nf)

        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.rbs = make_layer(RB_f, N_RB)

        self.d2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.d2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.d4_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.d4_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        fea_lr = self.rbs(fea)

        fea_d2 = self.lrelu(self.d2_conv2(self.lrelu(self.d2_conv1(fea_lr))))
        fea_d4 = self.lrelu(self.d4_conv2(self.lrelu(self.d4_conv1(fea_d2))))

        return [fea_lr, fea_d2, fea_d4]


class REAL(nn.Module):
    def __init__(self, nf=64, front_RB=5, back_RB=10, nbr=2, groups=8,
            interpolation='bilinear'):
        super(REAL, self).__init__()
        self.nbr = nbr
        self.nframes = 2 * nbr + 1
        self.interpolation = interpolation

        RB_f = functools.partial(ResidualBlock_noBN, nf=nf)

        self.fea_extract = encoder(nf=nf, N_RB=front_RB)
        self.align = pcd_align(nf=nf, groups=groups)
        self.fuse = easy_fuse(nf=nf, nframes=self.nframes)
        self.recon = make_layer(RB_f, back_RB)
        self.up_conv1 = nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True)
        self.up_conv2 = nn.Conv2d(64, 64 * 4, 3, 1, 1, bias=True)
        self.hr_conv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.out_conv = nn.Conv2d(64, 3, 1, 1, bias=True)
        self.ps = nn.PixelShuffle(upscale_factor=2)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self._init_var()
        self._initialize_weights()
        self.up1 = MyUpsampling(scale_factor=(1,1), align_corners=False)

    def _init_var(self):
        self.h_buf = []
        self.f_buf = []
        self.cur_pt = 0 

    def _initialize_weights(self, scale=0.1):
        # for residual block
        for M in [self.fea_extract.rbs, self.recon]:
            for m in M.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                    m.weight.data *= scale
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, x):
        self._init_var()
        outputs = []

        B, N, C, H, W = x.size()
        # prepare for buffers
        for i in range(self.nbr):
            self.h_buf.append(self.fea_extract(x[:, i, ...].contiguous()))
        for i in range(self.nbr, self.nbr * 2):
            self.f_buf.append(self.fea_extract(x[:, i, ...].contiguous()))
        self.cur_pt = self.nbr * 2

        while(self.cur_pt < N):
            cur_x = x[:, self.cur_pt, ...].contiguous()
            base = x[:, self.cur_pt - self.nbr, ...].contiguous()
            output = self.process(cur_x, base)
            outputs.append(output)

        return torch.stack(outputs, dim=1)
            
    def process_zk(self,ref_fea,nbr_l,base):
        
        # print(ref_fea[0].size())
        B, C, H, W = ref_fea[0].size()
        # print('shape',B,C,H,W,base.max(),base.min(),base.mean())

        nbr_lr = torch.cat([fea[0] for fea in nbr_l], dim=0)
        nbr_d2 = torch.cat([fea[1] for fea in nbr_l], dim=0)
        nbr_d4 = torch.cat([fea[2] for fea in nbr_l], dim=0)
        nbr_mul_fea = [nbr_lr, nbr_d2, nbr_d4]

        ref_mul_fea = [ref_fea[0].repeat(self.nbr * 2, 1, 1, 1),
                       ref_fea[1].repeat(self.nbr * 2, 1, 1, 1),
                       ref_fea[2].repeat(self.nbr * 2, 1, 1, 1)]
        al_nbr = self.align(nbr_mul_fea, ref_mul_fea)
        al_fea = torch.cat([al_nbr, ref_fea[0]], dim=0)
        al_fea = al_fea.view(self.nframes, B, C, H, W).permute(1, 0, 2, 3, 4).contiguous()
        
        fuse_fea = self.fuse(al_fea)
        recon_fea = self.recon(fuse_fea)
        mr_fea = self.lrelu(self.ps(self.up_conv1(recon_fea)))
        hr_fea = self.lrelu(self.ps(self.up_conv2(mr_fea)))
        out = self.out_conv(self.lrelu(self.hr_conv(hr_fea)))

        #base = F.interpolate(base, scale_factor=4, mode=self.interpolation, align_corners=False)
        self.up1.set_scale((4, 4))
        base = self.up1(base)
        out += base
        return out
    def process(self, x, base):
        if x is not None:
            self.f_buf.append(self.fea_extract(x))
            self.cur_pt += 1

        ref_fea = self.f_buf.pop(0)
        B, C, H, W = ref_fea[0].size()

        nbr_l = []
        cnt_h = len(self.h_buf)
        cnt_f = len(self.f_buf)
        if cnt_h < self.nbr:
            take_h = cnt_h
            take_f = self.nbr * 2 - take_h
        elif cnt_f < self.nbr:
            take_h = self.nbr * 2 - cnt_f
            take_f = cnt_f
        else:
            take_h = self.nbr
            take_f = self.nbr
        nbr_l = self.h_buf[-take_h:] + self.f_buf[:take_f]
        nbr_lr = torch.cat([fea[0] for fea in nbr_l], dim=0)
        nbr_d2 = torch.cat([fea[1] for fea in nbr_l], dim=0)
        nbr_d4 = torch.cat([fea[2] for fea in nbr_l], dim=0)
        nbr_mul_fea = [nbr_lr, nbr_d2, nbr_d4]
        ref_mul_fea = [ref_fea[0].repeat(self.nbr * 2, 1, 1, 1),
                       ref_fea[1].repeat(self.nbr * 2, 1, 1, 1),
                       ref_fea[2].repeat(self.nbr * 2, 1, 1, 1)]
        al_nbr = self.align(nbr_mul_fea, ref_mul_fea)
        al_fea = torch.cat([al_nbr, ref_fea[0]], dim=0)
        al_fea = al_fea.view(self.nframes, B, C, H, W).permute(1, 0, 2, 3, 4).contiguous()
        
        fuse_fea = self.fuse(al_fea)
        recon_fea = self.recon(fuse_fea)
        mr_fea = self.lrelu(self.ps(self.up_conv1(recon_fea)))
        hr_fea = self.lrelu(self.ps(self.up_conv2(mr_fea)))
        out = self.out_conv(self.lrelu(self.hr_conv(hr_fea)))
        #base = F.interpolate(base, scale_factor=4, mode=self.interpolation, align_corners=False)
        self.up1.set_scale((4, 4))
        base = self.up1(base)
        out += base

        # add ref_fea to h_buf
        self.h_buf.append(ref_fea)
        if len(self.h_buf) > self.nbr * 2:
            self.h_buf.pop(0)

        return out


if __name__ == '__main__':
    # from thop import profile, clever_format
    import time

    torch.cuda.set_device(0)
    device = torch.device('cuda')
    net = REAL(nbr=2)
    net.eval()
    net.to(device)

    input = torch.rand((1, 5, 3, 64, 64))
    input = input.to(device)

    with torch.no_grad():
        out = net(input)
    print(out.shape)
    # t1 = time.time()
    # flops, params = profile(net, inputs=(input, ))
    # t2 = time.time()
    # print('time:', (t2 - t1) / input.size(1))

    # print(flops // input.size(1), params)
