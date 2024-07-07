import numbers
import torch
from torch import nn
from einops import rearrange
from VIT import SwinT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, in_ch=6, out_ch=48):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, 1, padding=0)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 16, 5, padding=2)
        self.conv4 = nn.Conv2d(16, 16, 7, padding=3)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.relu(x1)
        x2 = self.conv2(x1)
        x2 = self.relu(x2)
        x3 = self.conv3(x2)
        x3 = self.relu(x3)
        x4 = self.conv4(x3)
        x4 = self.relu(x4)
        out = torch.cat([x1, x2, x3, x4], dim=1)
        return out


class RB(nn.Module):
    def __init__(self, in_ch, depth):
        super(RB, self).__init__()
        self.conv_Decoder_1x1_1 = nn.Conv2d(in_ch, 64, kernel_size=1, padding=0, bias=True)
        self.conv_Decoder_1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_Decoder_2 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv_Decoder_3 = nn.Conv2d(64, 64, 7, padding=3)
        self.vit_Decoder = SwinT(n_feats=64, depth=depth)
        self.conv_Decoder_5 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu_Decoder = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x1 = self.conv_Decoder_1x1_1(x)
        x2 = self.relu_Decoder(self.conv_Decoder_1(x1))
        x3 = self.relu_Decoder(self.conv_Decoder_2(x2))
        x4 = self.relu_Decoder(self.conv_Decoder_3(x3)) * x1
        x5 = self.vit_Decoder(x4)
        x5 = x1 + x5
        x = self.conv_Decoder_5(x5)
        x = self.relu_Decoder(x)

        return x


class IB(nn.Module):
    def __init__(self, in_ch, depth):
        super(IB, self).__init__()
        self.conv1x1_total_1 = nn.Conv2d(in_ch, 64, kernel_size=1, padding=0, bias=True)
        self.conv1x1_total_2 = nn.Conv2d(in_ch, 64, kernel_size=1, padding=0, bias=True)
        self.conv1x1_total_3 = nn.Conv2d(128, 64, kernel_size=1, padding=0, bias=True)
        self.conv_s_1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_m_1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_total_1 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv_total_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_total_3 = nn.Conv2d(64, 64, 1, padding=0)
        self.vit_total = SwinT(n_feats=64, depth=depth)
        self.conv_total_5 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu_total = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x_s, x_m):
        x_s = self.relu_total(self.conv_s_1(self.conv1x1_total_1(x_s)))
        x_m = self.relu_total(self.conv_m_1(self.conv1x1_total_2(x_m)))
        x1 = torch.cat([x_s, x_m], dim=1)
        x1 = self.conv1x1_total_3(x1)
        x2 = self.relu_total(self.conv_total_1(x1))
        x3 = self.relu_total(self.conv_total_2(x2))
        x4 = self.relu_total(self.conv_total_3(x3)) * x1
        x5 = self.vit_total(x4)
        x5 = x1 + x5
        x = self.conv_total_5(x5)
        x = self.relu_total(x)

        return x


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class MR_HOIB(nn.Module):
    def __init__(self, dim=128, num_heads=8, bias=False, LayerNorm_type='WithBias'):
        super(MR_HOIB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)

        self.kv_1 = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_2 = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv_1 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.kv_dwconv_2 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv_1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.q_dwconv_2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

    def forward(self, x_s, x_m):
        x_s_1 = self.norm1(x_s)
        x_m_1 = self.norm2(x_m)
        b, c, h, w = x_s.shape
        kv_m = self.kv_dwconv_1(self.kv_1(x_m_1))
        k_m, v_m = kv_m.chunk(2, dim=1)
        q_s = self.q_dwconv_1(self.q_1(x_s_1))
        q_s = rearrange(q_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_m = rearrange(k_m, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_m = rearrange(v_m, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q_s = torch.nn.functional.normalize(q_s, dim=-1)
        k_m = torch.nn.functional.normalize(k_m, dim=-1)
        attn_s = (q_s @ k_m.transpose(-2, -1)) * self.temperature
        attn_s = attn_s.softmax(dim=-1)
        out_s = (attn_s @ v_m)
        out_s = rearrange(out_s, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_s = self.project_out_1(out_s)
        out_s = out_s + x_s

        kv_s = self.kv_dwconv_2(self.kv_2(x_s_1))
        k_s, v_s = kv_s.chunk(2, dim=1)
        q_m = self.q_dwconv_2(self.q_2(x_m_1))
        q_m = rearrange(q_m, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_s = rearrange(k_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_s = rearrange(v_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q_m = torch.nn.functional.normalize(q_m, dim=-1)
        k_s = torch.nn.functional.normalize(k_s, dim=-1)
        attn_m = (q_m @ k_s.transpose(-2, -1)) * self.temperature
        attn_m = attn_m.softmax(dim=-1)
        out_m = (attn_m @ v_s)
        out_m = rearrange(out_m, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_m = self.project_out_2(out_m)
        out_m = out_m + x_m

        return out_s, out_m


class MR_HEIB(nn.Module):
    def __init__(self, dim=128, num_heads=8, bias=False, LayerNorm_type='WithBias'):
        super(MR_HEIB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.kv_1 = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv_1 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv_1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.conv_MC1_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv_MC1_2 = nn.Conv2d(64, 128, 3, padding=1)
        self.MC1_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x2, x_other):
        x2 = self.MC1_relu(self.conv_MC1_1(x2))
        x_other = self.MC1_relu(self.conv_MC1_2(x_other))
        x2_1 = self.norm1(x2)
        x_other_1 = self.norm2(x_other)
        b, c, h, w = x2.shape
        kv = self.kv_dwconv_1(self.kv_1(x_other_1))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv_1(self.q_1(x2_1))
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out_1(out)
        out = out + x2
        return out


class DE_HEIB(nn.Module):
    def __init__(self):
        super(DE_HEIB, self).__init__()
        self.encoder = Encoder(in_ch=6)
        self.IP = MR_HEIB()

    def forward(self, x1, x2, x3, x2_S):
        x_avg = (x1 + x2 + x3) / 3
        x12 = x1 / x_avg
        x32 = x3 / x_avg
        wight_12 = x12
        wight_32 = x32
        zero = torch.zeros_like(wight_12)
        ones = torch.ones_like(wight_12)
        wight_12_1 = torch.where(0.5 < wight_12, ones, zero)
        wight_12_2 = torch.where(wight_12 < 1.5, ones, zero)
        wight_12 = wight_12_1 + wight_12_2 - ones
        wight_32_1 = torch.where(0.50 < wight_32, ones, zero)
        wight_32_2 = torch.where(wight_32 < 1.5, ones, zero)
        wight_32 = wight_32_1 + wight_32_2 - ones
        x1 = x1 * wight_12
        x3 = x3 * wight_32
        x = (x1 + x3) / 2
        x = self.encoder(x)
        out = self.IP(x2_S, x)

        return out


class HOIB(nn.Module):
    def __init__(self):
        super(HOIB, self).__init__()
        self.conv_IBS_1_1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_IBS_1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_IBS_1_3 = nn.Conv2d(64, 3, 3, padding=1)
        self.conv_IBS_1_4 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv_IBS_1_5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv_IBM_1_1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_IBM_1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_IBM_1_3 = nn.Conv2d(64, 3, 3, padding=1)
        self.conv_IBM_1_4 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv_IBM_1_5 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu_IB_1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.cross_a_1 = MR_HOIB()

    def forward(self, x_s, x_m):
        g_s = self.relu_IB_1(self.conv_IBS_1_1(x_s))
        g_m = self.relu_IB_1(self.conv_IBM_1_1(x_m))
        g_s = self.relu_IB_1(self.conv_IBS_1_2(g_s))
        g_m = self.relu_IB_1(self.conv_IBM_1_2(g_m))
        g_s = torch.sigmoid(self.conv_IBS_1_3(g_s))
        g_m = torch.sigmoid(self.conv_IBM_1_3(g_m))
        x_s = self.relu_IB_1(self.conv_IBS_1_4(g_s))
        x_m = self.relu_IB_1(self.conv_IBM_1_4(g_m))
        x_s = self.relu_IB_1(self.conv_IBS_1_5(x_s))
        x_m = self.relu_IB_1(self.conv_IBM_1_5(x_m))
        x_s, x_m = self.cross_a_1(x_s, x_m)

        return x_s, x_m, g_s, g_m


class Attention(nn.Module):
    def __init__(self, input_channels, reduction=4):
        super().__init__()
        self.conv1_1X1 = nn.Conv2d(input_channels * 2, input_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1)
        self.at_vit = SwinT(n_feats=input_channels, depth=2)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.w_1 = nn.Linear(2, input_channels // reduction, bias=True)
        self.w_2 = nn.Linear(input_channels // reduction, input_channels, bias=True)
        self.fc = nn.Sequential(
            self.w_1,
            nn.ReLU(inplace=True),
            self.w_2,
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        conv1_f = self.conv1_1X1(x)
        conv1_f = self.relu(conv1_f)
        conv1_f = self.at_vit(conv1_f)
        conv1_f = self.conv2(conv1_f)
        conv1_f = self.sigmoid(conv1_f)
        out1 = x1 * conv1_f
        out1 = self.conv3(out1)
        out1_GAP = torch.mean(out1, dim=1, keepdim=True)
        out1_GMP, index = torch.max(out1, dim=1, keepdim=True)
        x_P = torch.cat([out1_GAP, out1_GMP], dim=1)
        score = self.fc(x_P.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = out1 * score

        return out


class Fuse(nn.Module):
    def __init__(self):
        super(Fuse, self).__init__()
        self.attention_12 = Attention(input_channels=64)
        self.attention_23 = Attention(input_channels=64)
        self.fuse = nn.Conv2d(192, 64, 3, padding=1)
        self.LRelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, feature1, refer, feature2):
        feature1_1 = self.attention_12(feature1, refer)
        feature1_2 = self.attention_23(feature2, refer)
        out = torch.cat([feature1_1, feature1_2, refer], dim=1)
        out = self.fuse(out)
        out = self.LRelu(out)
        return out


class DTS(nn.Module):
    def __init__(self):
        super(DTS, self).__init__()
        self.E_S = Encoder(in_ch=6)
        self.E_M = Encoder(in_ch=6)
        self.Fuse = Fuse()
        self.single_1 = RB(in_ch=192, depth=1)
        self.single_2 = RB(in_ch=256, depth=1)
        self.single_3 = RB(in_ch=320, depth=2)
        self.multi_1 = RB(in_ch=192, depth=1)
        self.multi_2 = RB(in_ch=256, depth=1)
        self.multi_3 = RB(in_ch=320, depth=2)
        self.total = IB(in_ch=320, depth=2)
        self.HOIB_1 = HOIB()
        self.HOIB_2 = HOIB()
        self.HEIB1 = MR_HEIB()
        self.HEIB2 = DE_HEIB()
        self.conv_s = nn.Conv2d(64, 3, kernel_size=3, padding=1, bias=True)
        self.conv_m = nn.Conv2d(64, 3, kernel_size=3, padding=1, bias=True)
        self.conv_f = nn.Conv2d(64, 3, kernel_size=3, padding=1, bias=True)

    def forward(self, x1, x2, x3):
        x1_image = x1
        x2_image = x2
        x3_image = x3

        x2_Single = self.E_S(x2)
        x1 = self.E_M(x1)
        x2_enconded = self.E_M(x2)
        x3 = self.E_M(x3)

        x2_S_cross = self.HEIB2(x1_image, x2_image, x3_image, x2_Single)
        x2_S = torch.cat([x2_Single, x2_S_cross], dim=1)
        merger_f_source = self.Fuse(x1, x2_enconded, x3)
        merger_f_cross = self.HEIB1(merger_f_source, x2_Single)
        merger_f = torch.cat([merger_f_source, merger_f_cross], dim=1)
        out_single_1 = self.single_1(x2_S)
        out_multi_1 = self.multi_1(merger_f)

        inter_s_1, inter_m_1, gradient_s_1, gradient_m_1 = self.HOIB_1(out_single_1, out_multi_1)
        out_1_m = torch.cat([inter_m_1, out_multi_1, merger_f_source], dim=1)
        out_1_s = torch.cat([inter_s_1, out_single_1, x2_Single], dim=1)
        out_single_2 = self.single_2(out_1_s)
        out_multi_2 = self.multi_2(out_1_m)

        inter_s_2, inter_m_2, gradient_s_2, gradient_m_2 = self.HOIB_2(out_single_2, out_multi_2)
        out_2_m = torch.cat([inter_m_2, out_multi_2, out_multi_1, merger_f_source], dim=1)
        out_2_s = torch.cat([inter_s_2, out_single_1, out_single_2, x2_Single], dim=1)
        out_single_3 = self.single_3(out_2_s)
        out_multi_3 = self.multi_3(out_2_m)
        out_all = self.total(out_2_s, out_2_m)

        out_single = torch.sigmoid(self.conv_s(out_single_3))
        out_multi = torch.sigmoid(self.conv_m(out_multi_3))
        out_final = torch.sigmoid(self.conv_f(out_all))

        return out_single, out_multi, out_final, gradient_s_1, gradient_m_1, gradient_s_2, gradient_m_2
