"""
Baseline architectures for the MGMTNet comparison (pure PyTorch, no extra packages):
  unet, attunet, swinunet, unetr, transunet
Every baseline is wrapped so that it returns the SAME dict as MGMTNet:
  kidney_logits, stone_logits, cls_logits, m_k, m_s
Design (document this in the Methods):
  * one decoder with a 2-channel output (kidney, stone), as in the original architectures
  * classification head = global average pooling of the bottleneck + the SAME MLP as MGMTNet
    (Linear-BN-ReLU-Dropout(0.3)-Linear). Baselines have no mask-guided attention / SE.
  * trained from scratch (no pre-trained weights), like MGMTNet.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- common blocks
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al., 2018). g and x must have the same spatial size."""
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.wg = nn.Sequential(nn.Conv2d(f_g, f_int, 1), nn.BatchNorm2d(f_int))
        self.wx = nn.Sequential(nn.Conv2d(f_l, f_int, 1), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        return x * self.psi(self.relu(self.wg(g) + self.wx(x)))


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, attention=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.gate = AttentionGate(out_ch, skip_ch, max(skip_ch // 2, 1)) if attention else None
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if self.gate is not None:
            skip = self.gate(x, skip)
        return self.conv(torch.cat([x, skip], 1))


# ----------------------------------------------------------------------------- U-Net / Attention U-Net
class UNet(nn.Module):
    """Same widths as the MGMTNet CNN path (32-64-128-256, bottleneck 512)."""
    def __init__(self, in_ch=3, n_out=2, attention=False):
        super().__init__()
        self.e1, self.e2, self.e3, self.e4 = ConvBlock(in_ch, 32), ConvBlock(32, 64), ConvBlock(64, 128), ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2, 2)
        self.bott = ConvBlock(256, 512)
        self.d4 = DecoderBlock(512, 256, 256, attention)
        self.d3 = DecoderBlock(256, 128, 128, attention)
        self.d2 = DecoderBlock(128, 64, 64, attention)
        self.d1 = DecoderBlock(64, 32, 32, attention)
        self.out = nn.Conv2d(32, n_out, 1)
        self.bottleneck_channels = 512

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        x4 = self.e4(self.pool(x3))
        b = self.bott(self.pool(x4))
        d = self.d1(self.d2(self.d3(self.d4(b, x4), x3), x2), x1)
        return self.out(d), b


# ----------------------------------------------------------------------------- UNETR (2D)
class UNETR(nn.Module):
    """2D UNETR (Hatamizadeh et al., 2022): ViT encoder (patch 16) with skip taps at layers 3/6/9/12 and a conv decoder.
    Default = ViT-B (hidden 768, 12 layers, 12 heads), trained from scratch."""
    def __init__(self, in_ch=3, n_out=2, img_size=512, patch=16, hidden=768, depth=12, heads=12, mlp=3072, f=16):
        super().__init__()
        assert img_size % patch == 0
        self.hidden, self.grid, self.img_size = hidden, img_size // patch, img_size
        self.patch_embed = nn.Conv2d(in_ch, hidden, patch, patch)
        self.pos = nn.Parameter(torch.zeros(1, self.grid ** 2, hidden))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden, heads, mlp, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden)
        self.taps = (depth // 4, depth // 2, 3 * depth // 4, depth)

        def up(i, o):
            return nn.Sequential(nn.ConvTranspose2d(i, o, 2, 2), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.s1 = ConvBlock(in_ch, f)                                           # H
        self.s2 = nn.Sequential(up(hidden, 8 * f), up(8 * f, 4 * f), up(4 * f, 2 * f))   # H/2 (from layer 3)
        self.s3 = nn.Sequential(up(hidden, 8 * f), up(8 * f, 4 * f))                      # H/4 (layer 6)
        self.s4 = up(hidden, 8 * f)                                                         # H/8 (layer 9)
        self.d4 = DecoderBlock(hidden, 8 * f, 8 * f)
        self.d3 = DecoderBlock(8 * f, 4 * f, 4 * f)
        self.d2 = DecoderBlock(4 * f, 2 * f, 2 * f)
        self.d1 = DecoderBlock(2 * f, f, f)
        self.out = nn.Conv2d(f, n_out, 1)
        self.bottleneck_channels = hidden

    def _map(self, t):
        return t.transpose(1, 2).reshape(t.shape[0], self.hidden, self.grid, self.grid)

    def forward(self, x):
        assert x.shape[-1] == self.img_size and x.shape[-2] == self.img_size, "UNETR expects fixed input size"
        t = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos
        taps = {}
        for i, blk in enumerate(self.blocks, 1):
            t = blk(t)
            if i in self.taps:
                taps[i] = t
        z3, z6, z9 = (self._map(taps[k]) for k in self.taps[:3])
        z12 = self._map(self.norm(taps[self.taps[3]]))
        d = self.d1(self.d2(self.d3(self.d4(z12, self.s4(z9)), self.s3(z6)), self.s2(z3)), self.s1(x))
        return self.out(d), z12


# ----------------------------------------------------------------------------- Swin-UNet (2D, from scratch)
def window_partition(x, w):                      # (B,H,W,C) -> (B*nW, w*w, C)
    B, H, W, C = x.shape
    x = x.view(B, H // w, w, W // w, w, C).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, w * w, C)


def window_reverse(xw, w, H, W):                 # (B*nW, w*w, C) -> (B,H,W,C)
    B = xw.shape[0] // ((H // w) * (W // w))
    x = xw.view(B, H // w, W // w, w, w, -1).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(B, H, W, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim, window, heads):
        super().__init__()
        self.heads, self.window = heads, window
        self.scale = (dim // heads) ** -0.5
        self.rpb_table = nn.Parameter(torch.zeros((2 * window - 1) ** 2, heads))
        nn.init.trunc_normal_(self.rpb_table, std=0.02)
        coords = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window), indexing="ij")).flatten(1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window - 1
        rel[:, :, 1] += window - 1
        rel[:, :, 0] *= 2 * window - 1
        self.register_buffer("rel_index", rel.sum(-1), persistent=False)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        Bn, N, C = x.shape
        qkv = self.qkv(x).reshape(Bn, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.rpb_table[self.rel_index.view(-1)].view(N, N, -1).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(Bn // nW, nW, self.heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.heads, N, N)
        attn = attn.softmax(-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(Bn, N, C))


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, window, shift, mlp_ratio=4.0):
        super().__init__()
        self.window, self.shift = window, shift
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))
        self._mask_cache = {}

    def _mask(self, H, W, device):
        if self.shift == 0:
            return None
        key = (H, W, str(device))
        if key not in self._mask_cache:
            w, s = self.window, self.shift
            img = torch.zeros(1, H, W, 1)
            cnt = 0
            for hs in (slice(0, -w), slice(-w, -s), slice(-s, None)):
                for ws in (slice(0, -w), slice(-w, -s), slice(-s, None)):
                    img[:, hs, ws, :] = cnt
                    cnt += 1
            mw = window_partition(img, w).squeeze(-1)
            m = mw.unsqueeze(1) - mw.unsqueeze(2)
            m = m.masked_fill(m != 0, -100.0).masked_fill(m == 0, 0.0)
            self._mask_cache[key] = m.to(device)
        return self._mask_cache[key]

    def forward(self, x):                        # (B,H,W,C)
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)
        if self.shift > 0:
            x = torch.roll(x, (-self.shift, -self.shift), (1, 2))
        xw = self.attn(window_partition(x, self.window), self._mask(H, W, x.device))
        x = window_reverse(xw, self.window, H, W)
        if self.shift > 0:
            x = torch.roll(x, (self.shift, self.shift), (1, 2))
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


def swin_stage(dim, depth, heads, window):
    return nn.Sequential(*[SwinBlock(dim, heads, window, 0 if i % 2 == 0 else window // 2) for i in range(depth)])


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.red = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], -1)
        return self.red(self.norm(x))


class PatchExpand(nn.Module):                    # dim -> dim/2, resolution x2
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.lin(x).view(B, H, W, 2, 2, C // 2).permute(0, 1, 3, 2, 4, 5).reshape(B, 2 * H, 2 * W, C // 2)
        return self.norm(x)


class FinalPatchExpand4(nn.Module):              # dim -> dim, resolution x4
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, 16 * dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.dim = dim

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.lin(x).view(B, H, W, 4, 4, C).permute(0, 1, 3, 2, 4, 5).reshape(B, 4 * H, 4 * W, C)
        return self.norm(x)


class SwinUNet(nn.Module):
    """Swin-UNet (Cao et al., 2022), Swin-T configuration (embed 96, depths 2-2-2-2, heads 3-6-12-24), patch 4.
    Window size 8 (instead of 7) so that 512x512 inputs are divisible at every stage. Stochastic depth omitted."""
    def __init__(self, in_ch=3, n_out=2, embed=96, depths=(2, 2, 2, 2), heads=(3, 6, 12, 24), window=8):
        super().__init__()
        d = [embed * 2 ** i for i in range(4)]
        self.patch_embed = nn.Conv2d(in_ch, embed, 4, 4)
        self.pe_norm = nn.LayerNorm(embed)
        self.enc = nn.ModuleList([swin_stage(d[i], depths[i], heads[i], window) for i in range(4)])
        self.merge = nn.ModuleList([PatchMerging(d[i]) for i in range(3)])
        self.expand = nn.ModuleList([PatchExpand(d[3]), PatchExpand(d[2]), PatchExpand(d[1])])
        self.cat_lin = nn.ModuleList([nn.Linear(2 * d[i], d[i]) for i in (2, 1, 0)])
        self.dec = nn.ModuleList([swin_stage(d[i], 2, heads[i], window) for i in (2, 1, 0)])
        self.final_expand = FinalPatchExpand4(embed)
        self.head = nn.Conv2d(embed, n_out, 1)
        self.bottleneck_channels = d[3]

    def forward(self, x):
        x = self.pe_norm(self.patch_embed(x).permute(0, 2, 3, 1))
        skips = []
        for i in range(3):
            x = self.enc[i](x)
            skips.append(x)
            x = self.merge[i](x)
        x = self.enc[3](x)
        bott = x
        for j, i in enumerate((2, 1, 0)):
            x = self.expand[j](x)
            x = self.dec[j](self.cat_lin[j](torch.cat([x, skips[i]], -1)))
        x = self.final_expand(x)
        return self.head(x.permute(0, 3, 1, 2)), bott.permute(0, 3, 1, 2)


# ----------------------------------------------------------------------------- TransUNet (2D, from scratch)
class Bottleneck(nn.Module):
    """ResNet bottleneck (1x1 - 3x3 - 1x1) with BatchNorm."""
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        mid = cout // 4
        self.c1, self.b1 = nn.Conv2d(cin, mid, 1, bias=False), nn.BatchNorm2d(mid)
        self.c2, self.b2 = nn.Conv2d(mid, mid, 3, stride, 1, bias=False), nn.BatchNorm2d(mid)
        self.c3, self.b3 = nn.Conv2d(mid, cout, 1, bias=False), nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        r = x if self.down is None else self.down(x)
        y = self.relu(self.b1(self.c1(x)))
        y = self.relu(self.b2(self.c2(y)))
        y = self.b3(self.c3(y))
        return self.relu(y + r)


def res_stage(cin, cout, n, stride):
    return nn.Sequential(Bottleneck(cin, cout, stride), *[Bottleneck(cout, cout) for _ in range(n - 1)])


class TUDecoderBlock(nn.Module):
    """Bilinear x2 upsampling, optional skip concatenation, two 3x3 Conv-BN-ReLU (cascaded upsampler of TransUNet)."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], 1)
        return self.conv(x)


class TransUNet(nn.Module):
    """TransUNet (Chen et al., 2021), R50-ViT-B/16 layout: ResNet-50-style CNN encoder (stages with 3-4-9 bottlenecks, 1/16 output),
    a 12-layer ViT-B (hidden 768) on the 1/16 feature map, and a cascaded upsampler with three skip connections
    (1/8, 1/4, 1/2). BatchNorm replaces the GroupNorm/weight-standardisation of the original; trained from scratch."""
    def __init__(self, in_ch=3, n_out=2, img_size=512, hidden=768, depth=12, heads=12, mlp=3072):
        super().__init__()
        assert img_size % 16 == 0
        self.img_size, self.grid, self.hidden = img_size, img_size // 16, hidden
        self.stem = nn.Sequential(nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))  # 1/2
        self.pool = nn.MaxPool2d(3, 2, 1)                                                                                 # 1/4
        self.layer1 = res_stage(64, 256, 3, 1)       # 1/4,  256 ch
        self.layer2 = res_stage(256, 512, 4, 2)      # 1/8,  512 ch
        self.layer3 = res_stage(512, 1024, 9, 2)     # 1/16, 1024 ch
        self.embed = nn.Conv2d(1024, hidden, 1)
        self.pos = nn.Parameter(torch.zeros(1, self.grid ** 2, hidden))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden, heads, mlp, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden)
        self.conv_more = nn.Sequential(nn.Conv2d(hidden, 512, 3, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.d0 = TUDecoderBlock(512, 512, 256)      # -> 1/8
        self.d1 = TUDecoderBlock(256, 256, 128)      # -> 1/4
        self.d2 = TUDecoderBlock(128, 64, 64)        # -> 1/2
        self.d3 = TUDecoderBlock(64, 0, 16)          # -> full resolution
        self.out = nn.Conv2d(16, n_out, 3, padding=1)
        self.bottleneck_channels = hidden

    def forward(self, x):
        assert x.shape[-1] == self.img_size and x.shape[-2] == self.img_size, "TransUNet expects fixed input size"
        B = x.shape[0]
        s2 = self.stem(x)
        s4 = self.layer1(self.pool(s2))
        s8 = self.layer2(s4)
        f16 = self.layer3(s8)
        t = self.embed(f16).flatten(2).transpose(1, 2) + self.pos
        for blk in self.blocks:
            t = blk(t)
        z = self.norm(t).transpose(1, 2).reshape(B, self.hidden, self.grid, self.grid)
        d = self.conv_more(z)
        d = self.d3(self.d2(self.d1(self.d0(d, s8), s4), s2))
        return self.out(d), z


# ----------------------------------------------------------------------------- multi-task wrapper + factory
class MultiTaskWrapper(nn.Module):
    def __init__(self, backbone, num_classes=2):
        super().__init__()
        self.backbone = backbone
        c = backbone.bottleneck_channels
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Linear(c, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
                                        nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x):
        seg, feat = self.backbone(x)
        k, s = seg[:, 0:1], seg[:, 1:2]
        cls = self.classifier(self.gap(feat).flatten(1))
        return {"kidney_logits": k, "stone_logits": s, "cls_logits": cls,
                "m_k": torch.sigmoid(k), "m_s": torch.sigmoid(s)}


MODEL_NAMES = ["unet", "attunet", "swinunet", "unetr", "transunet"]


def build_model(name, img_size=512):
    if name == "unet":
        return MultiTaskWrapper(UNet(attention=False))
    if name == "attunet":
        return MultiTaskWrapper(UNet(attention=True))
    if name == "swinunet":
        return MultiTaskWrapper(SwinUNet())
    if name == "unetr":
        return MultiTaskWrapper(UNETR(img_size=img_size))
    if name == "transunet":
        return MultiTaskWrapper(TransUNet(img_size=img_size))
    raise ValueError(f"unknown model {name}; choose from {MODEL_NAMES}")
