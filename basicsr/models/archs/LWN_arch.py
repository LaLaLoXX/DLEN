import torch
from functools import partial
import torch.nn.functional as F
import torch.nn as nn
import pywt
from einops import rearrange
from typing import Sequence, Tuple, Union, List, Type, Callable
from timm.layers import SqueezeExcite, SeparableConvNormAct, drop_path, trunc_normal_, Mlp, DropPath
import numbers


def _as_wavelet(wavelet: Union[str, pywt.Wavelet]) -> pywt.Wavelet:
    """将字符串转换为小波对象，或直接返回小波对象"""
    if isinstance(wavelet, str):
        return pywt.Wavelet(wavelet)
    return wavelet


def get_filter_tensors(
        wavelet,
        flip: bool,
        device: Union[torch.device, str] = 'cpu',
        dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    wavelet = _as_wavelet(wavelet)
    # 转换为PyTorch张量并根据需要翻转
    def _create_tensor(filter: Sequence[float]) -> torch.Tensor:
        if flip:
            if isinstance(filter, torch.Tensor):
                return filter.flip(-1).unsqueeze(0).to(device)
            else:
                return torch.tensor(filter[::-1], device=device, dtype=dtype).unsqueeze(0)
        else:
            if isinstance(filter, torch.Tensor):
                return filter.unsqueeze(0).to(device)
            else:
                return torch.tensor(filter, device=device, dtype=dtype).unsqueeze(0)
    # 从小波对象中获取分解和重构滤波器
    dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
    dec_lo_tensor = _create_tensor(dec_lo)
    dec_hi_tensor = _create_tensor(dec_hi)
    rec_lo_tensor = _create_tensor(rec_lo)
    rec_hi_tensor = _create_tensor(rec_hi)
    return dec_lo_tensor, dec_hi_tensor, rec_lo_tensor, rec_hi_tensor



class ShuffleBlock(nn.Module):
    def __init__(self, groups=2):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        x = rearrange(x, 'b (g f) h w -> b g f h w', g=self.groups)
        x = rearrange(x, 'b g f h w -> b f g h w')
        x = rearrange(x, 'b f g h w -> b (f g) h w')
        return x
    

def _get_pad(data_len: int, filt_len: int) -> Tuple[int, int]:

    padr = (2 * filt_len - 3) // 2
    padl = (2 * filt_len - 3) // 2

    # pad to even singal length.
    if data_len % 2 != 0:
        padr += 1

    return padr, padl


def fwt_pad2(
        data: torch.Tensor, wavelet, mode: str = "replicate"
) -> torch.Tensor:
    wavelet = _as_wavelet(wavelet)
    # 计算需要的填充量
    padb, padt = _get_pad(data.shape[-2], len(wavelet.dec_lo))
    padr, padl = _get_pad(data.shape[-1], len(wavelet.dec_lo))
    # 对数据进行填充
    data_pad = F.pad(data, [padl, padr, padt, padb], mode=mode)
    return data_pad


class DWT(nn.Module):
    def __init__(self, dec_lo, dec_hi, wavelet='haar', level=1, mode="replicate"):
        # dec_lo: 低通分解滤波器
        # dec_hi: 高通分解滤波器
        # level: 小波分解层数
        # mode: 边界填充模式
        super(DWT, self).__init__()
        self.wavelet = _as_wavelet(wavelet)
        self.dec_lo = dec_lo
        self.dec_hi = dec_hi
        self.level = level
        self.mode = mode

    def forward(self, x):
        b, c, h, w = x.shape
        # 自适应层数
        if self.level is None:
            self.level = pywt.dwtn_max_level([h, w], self.wavelet)
        wavelet_component: List[
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = []

        l_component = x
        # 1. 构建2D小波核
        dwt_kernel = construct_2d_filt(lo=self.dec_lo, hi=self.dec_hi)
        dwt_kernel = dwt_kernel.repeat(c, 1, 1)
        dwt_kernel = dwt_kernel.unsqueeze(dim=1)
        # 2. 多层小波分解
        for _ in range(self.level):
            # 填充
            l_component = fwt_pad2(l_component, self.wavelet, mode=self.mode)
            # 卷积得到高频分量
            h_component = F.conv2d(l_component, dwt_kernel, stride=2, groups=c)
            res = rearrange(h_component, 'b (c f) h w -> b c f h w', f=4)
            # 分离不同方向的高频分量
            l_component, lh_component, hl_component, hh_component = res.split(1, 2)
            # 保存分解结果
            wavelet_component.append((lh_component.squeeze(2), hl_component.squeeze(2), hh_component.squeeze(2)))
        # 保存低频分量
        wavelet_component.append(l_component.squeeze(2))
        return wavelet_component[::-1]


def _outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """计算两个1D向量的外积"""
    a_flat = torch.reshape(a, [-1])
    b_flat = torch.reshape(b, [-1])
    a_mul = torch.unsqueeze(a_flat, dim=-1)
    b_mul = torch.unsqueeze(b_flat, dim=0)
    # 结果是一个矩阵，其中(i,j)位置的值是a[i]*b[j]
    return a_mul * b_mul


def construct_2d_filt(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """构建2D滤波器组
    
    Args:
        lo: 低通输入滤波器
        hi: 高通输入滤波器
        
    Returns:
        堆叠的2D滤波器,维度为[filt_no, 1, height, width]
        四个滤波器按ll, lh, hl, hh顺序排列
    """
    ll = _outer(lo, lo)
    lh = _outer(hi, lo)
    hl = _outer(lo, hi)
    hh = _outer(hi, hi)
    return torch.stack([ll, lh, hl, hh], 0)


class IDWT(nn.Module):
    def __init__(self, rec_lo, rec_hi, wavelet='haar', level=1, mode="constant"):
        super(IDWT, self).__init__()
        self.rec_lo = rec_lo
        self.rec_hi = rec_hi
        self.wavelet = wavelet
        # self.convT = nn.ConvTranspose2d(c2 * 4, c1, kernel_size=weight.shape[-2:], groups=c1, stride=2)
        # self.convT.weight = torch.nn.Parameter(rec_filt)
        self.level = level
        self.mode = mode

    def forward(self, x, weight=None):
        l_component = x[0]
        _, c, _, _ = l_component.shape
        if weight is None:  # soft orthogonal
            idwt_kernel = construct_2d_filt(lo=self.rec_lo, hi=self.rec_hi)
            idwt_kernel = idwt_kernel.repeat(c, 1, 1)
            idwt_kernel = idwt_kernel.unsqueeze(dim=1)
        else:  # hard orthogonal
            idwt_kernel= torch.flip(weight, dims=[-1, -2])

        self.filt_len = idwt_kernel.shape[-1]
        for c_pos, component_lh_hl_hh in enumerate(x[1:]):
            l_component = torch.cat(
                # ll, lh, hl, hl, hh
                [l_component.unsqueeze(2), component_lh_hl_hh[0].unsqueeze(2),
                 component_lh_hl_hh[1].unsqueeze(2), component_lh_hl_hh[2].unsqueeze(2)], 2
            )
            # cat is not work for the strange transpose
            l_component = rearrange(l_component, 'b c f h w -> b (c f) h w')
            l_component = F.conv_transpose2d(l_component, idwt_kernel, stride=2, groups=c)

            # remove the padding
            padl = (2 * self.filt_len - 3) // 2
            padr = (2 * self.filt_len - 3) // 2
            padt = (2 * self.filt_len - 3) // 2
            padb = (2 * self.filt_len - 3) // 2
            if c_pos < len(x) - 2:
                pred_len = l_component.shape[-1] - (padl + padr)
                next_len = x[c_pos + 2][0].shape[-1]
                pred_len2 = l_component.shape[-2] - (padt + padb)
                next_len2 = x[c_pos + 2][0].shape[-2]
                if next_len != pred_len:
                    padr += 1
                    pred_len = l_component.shape[-1] - (padl + padr)
                    assert (
                            next_len == pred_len
                    ), "padding error, please open an issue on github "
                if next_len2 != pred_len2:
                    padb += 1
                    pred_len2 = l_component.shape[-2] - (padt + padb)
                    assert (
                            next_len2 == pred_len2
                    ), "padding error, please open an issue on github "
            if padt > 0:
                l_component = l_component[..., padt:, :]
            if padb > 0:
                l_component = l_component[..., :-padb, :]
            if padl > 0:
                l_component = l_component[..., padl:]
            if padr > 0:
                l_component = l_component[..., :-padr]
        return l_component


class LWN(nn.Module):
    """可学习的小波网络模块"""
    
    def __init__(self, 
                 dim: int,
                 wavelet: str = 'haar',
                 initialize: bool = True,
                 head: int = 4,
                 drop_rate: float = 0.,
                 use_ca: bool = False,
                 use_sa: bool = False):
        super().__init__()
        self.dim = dim
        self.wavelet = _as_wavelet(wavelet)
        
        # 获取小波滤波器
        dec_lo, dec_hi, rec_lo, rec_hi = get_filter_tensors(
            wavelet, flip=True
        )
        
        # 初始化滤波器参数
        if initialize:
            self.dec_lo = nn.Parameter(dec_lo, requires_grad=True)
            self.dec_hi = nn.Parameter(dec_hi, requires_grad=True)
            self.rec_lo = nn.Parameter(rec_lo.flip(-1), requires_grad=True)
            self.rec_hi = nn.Parameter(rec_hi.flip(-1), requires_grad=True)
        else:
            self.dec_lo = nn.Parameter(torch.rand_like(dec_lo) * 2 - 1, requires_grad=True)
            self.dec_hi = nn.Parameter(torch.rand_like(dec_hi) * 2 - 1, requires_grad=True)
            self.rec_lo = nn.Parameter(torch.rand_like(rec_lo) * 2 - 1, requires_grad=True)
            self.rec_hi = nn.Parameter(torch.rand_like(rec_hi) * 2 - 1, requires_grad=True)

        self.wavedec = DWT(self.dec_lo, self.dec_hi, wavelet=wavelet, level=1)
        self.waverec = IDWT(self.rec_lo, self.rec_hi, wavelet=wavelet, level=1)

        self.conv1 = nn.Conv2d(dim*4, dim*6, 1)
        self.conv2 = nn.Conv2d(dim*6, dim*6, 7, padding=3, groups=dim*6)  # dw
        self.act = nn.GELU()
        self.conv3 = nn.Conv2d(dim*6, dim*4, 1)
        self.use_sa = use_sa
        self.use_ca = use_ca
        if self.use_sa:
            self.sa_h = nn.Sequential(
                nn.PixelShuffle(2),  # 上采样
                nn.Conv2d(dim // 4, 1, kernel_size=1, padding=0, stride=1, bias=True)  # c -> 1
            )
            self.sa_v = nn.Sequential(
                nn.PixelShuffle(2),
                nn.Conv2d(dim // 4, 1, kernel_size=1, padding=0, stride=1, bias=True)
            )
            # self.sa_norm = LayerNorm2d(dim)
        if self.use_ca:
            self.ca_h = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # 全局池化
                nn.Conv2d(dim, dim, 1, padding=0, stride=1, groups=1, bias=True),  # conv2d
            )
            self.ca_v = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, dim, 1, padding=0, stride=1, groups=1, bias=True)
            )
            self.shuffle = ShuffleBlock(2)

    def forward(self, x):
        _, _, H, W = x.shape
        # 1. 小波分解
        ya, (yh, yv, yd) = self.wavedec(x)
        # 2. 特征处理
        dec_x = torch.cat([ya, yh, yv, yd], dim=1)
        x = self.conv1(dec_x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        # 3. 小波重构
        ya, yh, yv, yd = torch.chunk(x, 4, dim=1)
        y = self.waverec([ya, (yh, yv, yd)], None)
        # 4. 空间注意力
        if self.use_sa:
            sa_yh = self.sa_h(yh)   # 水平方向
            sa_yv = self.sa_v(yv)   # 垂直方向
            y = y * (sa_yv + sa_yh)
        # 5. 通道注意力
        if self.use_ca:
            yh = torch.nn.functional.interpolate(yh, scale_factor=2, mode='area')
            yv = torch.nn.functional.interpolate(yv, scale_factor=2, mode='area')
            ca_yh = self.ca_h(yh)   # 水平方向
            ca_yv = self.ca_v(yv)   # 垂直方向
            ca = self.shuffle(torch.cat([ca_yv, ca_yh], 1))  # channel shuffle
            ca_1, ca_2 = ca.chunk(2, dim=1)
            ca = ca_1 * ca_2   # gated channel attention
            y = y * ca
            y = y
        return y

    def get_wavelet_loss(self):
        return self.perfect_reconstruction_loss()[0] + self.alias_cancellation_loss()[0]

    def perfect_reconstruction_loss(self):
        """ Strang 107: Assuming alias cancellation holds:
        P(z) = F(z)H(z)
        Product filter P(z) + P(-z) = 2.
        However since alias cancellation is implemented as soft constraint:
        P_0 + P_1 = 2
        Somehow numpy and torch implement convolution differently.
        For some reason the machine learning people call cross-correlation
        convolution.
        https://discuss.pytorch.org/t/numpy-convolve-and-conv1d-in-pytorch/12172
        Therefore for true convolution one element needs to be flipped.
        """
        # polynomial multiplication is convolution, compute p(z):
        # print(dec_lo.shape, rec_lo.shape)
        pad = self.dec_lo.shape[-1] - 1
        p_lo = F.conv1d(
            self.dec_lo.flip(-1).unsqueeze(0),
            self.rec_lo.flip(-1).unsqueeze(0),
            padding=pad)
        pad = self.dec_hi.shape[-1] - 1
        p_hi = F.conv1d(
            self.dec_hi.flip(-1).unsqueeze(0),
            self.rec_hi.flip(-1).unsqueeze(0),
            padding=pad)

        p_test = p_lo + p_hi

        two_at_power_zero = torch.zeros(p_test.shape, device=p_test.device,
                                        dtype=p_test.dtype)
        two_at_power_zero[..., p_test.shape[-1] // 2] = 2
        # square the error
        errs = (p_test - two_at_power_zero) * (p_test - two_at_power_zero)
        return torch.sum(errs), p_test, two_at_power_zero

    def alias_cancellation_loss(self):
        """ Implementation of the ac-loss as described on page 104 of Strang+Nguyen.
            F0(z)H0(-z) + F1(z)H1(-z) = 0 """
        m1 = torch.tensor([-1], device=self.dec_lo.device, dtype=self.dec_lo.dtype)
        length = self.dec_lo.shape[-1]
        mask = torch.tensor([torch.pow(m1, n) for n in range(length)][::-1],
                            device=self.dec_lo.device, dtype=self.dec_lo.dtype)
        # polynomial multiplication is convolution, compute p(z):
        pad = self.dec_lo.shape[-1] - 1
        p_lo = torch.nn.functional.conv1d(
            self.dec_lo.flip(-1).unsqueeze(0) * mask,
            self.rec_lo.flip(-1).unsqueeze(0),
            padding=pad)

        pad = self.dec_hi.shape[-1] - 1
        p_hi = torch.nn.functional.conv1d(
            self.dec_hi.flip(-1).unsqueeze(0) * mask,
            self.rec_hi.flip(-1).unsqueeze(0),
            padding=pad)

        p_test = p_lo + p_hi
        zeros = torch.zeros(p_test.shape, device=p_test.device,
                            dtype=p_test.dtype)
        errs = (p_test - zeros) * (p_test - zeros)
        return torch.sum(errs), p_test, zeros

# 1. 测试模型的初始化是否正常
def test_initialization():
    try:
        model = LWN(dim=64, wavelet='haar', initialize=True)
        print("Model initialized successfully!")
    except Exception as e:
        print(f"Model initialization failed: {e}")

# 2. 测试前向传播
def test_forward_pass():
    try:
        model = LWN(dim=64, wavelet='haar', initialize=True)
        # 创建一个假的输入数据 (batch_size=1, channels=64, height=32, width=32)
        x = torch.randn(1, 64, 32, 32)
        # 前向传播
        output = model(x)
        print(f"Output shape: {output.shape}")
        assert output.shape == x.shape, "Output shape mismatch"
        print("Forward pass successful!")
    except Exception as e:
        print(f"Forward pass failed: {e}")

# 3. 测试反向传播
def test_backpropagation():
    try:
        model = LWN(dim=64, wavelet='haar', initialize=True)
        # 创建假的输入数据
        x = torch.randn(1, 64, 32, 32, requires_grad=True)
        # 前向传播
        output = model(x)
        # 假设目标是与输入相同
        target = torch.randn_like(output)
        # 使用L2损失
        loss = nn.MSELoss()(output, target)
        # 反向传播
        loss.backward()
        print("Backpropagation successful!")
    except Exception as e:
        print(f"Backpropagation failed: {e}")

# 4. 测试损失函数
def test_loss_function():
    try:
        model = LWN(dim=64, wavelet='haar', initialize=True)
        # 获取小波损失
        wavelet_loss = model.get_wavelet_loss()
        print(f"Wavelet loss: {wavelet_loss}")
        assert isinstance(wavelet_loss, tuple), "Wavelet loss should return a tuple"
        print("Loss function test successful!")
    except Exception as e:
        print(f"Loss function test failed: {e}")

# 5. 测试小波重建与分解的一致性
def test_wavelet_reconstruction():
    try:
        model = LWN(dim=64, wavelet='haar', initialize=True)
        # 创建假的输入数据
        x = torch.randn(1, 64, 32, 32)
        # 小波分解
        ya, (yh, yv, yd) = model.wavedec(x)
        # 小波重建
        y_reconstructed = model.waverec([ya, (yh, yv, yd)], None)
        # 检查重建的图像是否与输入图像相近
        assert torch.allclose(x, y_reconstructed, atol=1e-5), "Wavelet reconstruction mismatch"
        print("Wavelet decomposition and reconstruction successful!")
    except Exception as e:
        print(f"Wavelet reconstruction test failed: {e}")
if __name__ == "__main__":
    test_initialization()
    test_forward_pass()
    test_backpropagation()
    test_loss_function()
    test_wavelet_reconstruction()