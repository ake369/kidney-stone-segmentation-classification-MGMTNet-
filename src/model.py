import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class TransformerBottleneck(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_layers=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            batch_first=True,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        transformed = self.transformer(tokens)  # (B, H*W, C)
        out = transformed.transpose(1, 2).view(B, C, H, W)
        return out


class MGMTNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        
        # --- ENCODER ---
        self.e1 = ConvBlock(in_channels, 32)
        self.e2 = ConvBlock(32, 64)
        self.e3 = ConvBlock(64, 128)
        self.e4 = ConvBlock(128, 256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.cnn_bot = ConvBlock(256, 512)
        
        # --- TRANSFORMER BOTTLENECK ---
        self.trans_bot = TransformerBottleneck(embed_dim=512, num_heads=8, num_layers=4)
        
        # --- PARALLEL DECODERS ---
        # 1. Kidney Decoder Stream D^(k)
        self.dk4 = DecoderBlock(512, 256, 256)
        self.dk3 = DecoderBlock(256, 128, 128)
        self.dk2 = DecoderBlock(128, 64, 64)
        self.dk1 = DecoderBlock(64, 32, 32)
        self.out_k = nn.Conv2d(32, 1, kernel_size=1)
        
        # 2. Stone Decoder Stream D^(s)
        self.ds4 = DecoderBlock(512, 256, 256)
        self.ds3 = DecoderBlock(256, 128, 128)
        self.ds2 = DecoderBlock(128, 64, 64)
        self.ds1 = DecoderBlock(64, 32, 32)
        self.out_s = nn.Conv2d(32, 1, kernel_size=1)
        
        # --- MASK-GUIDED ATTENTION & CLASSIFIER ---
        self.g_mask = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Squeeze-and-Excitation Channel Attention
        self.se_fc1 = nn.Linear(512, 32)
        self.se_fc2 = nn.Linear(32, 512)
        
        # Diagnostic Classification Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 1. Shared Encoder Path
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        x4 = self.e4(self.pool(x3))
        
        b_cnn = self.cnn_bot(self.pool(x4))
        b_trans = self.trans_bot(b_cnn)
        b_fused = b_cnn + b_trans  # Dual Bottleneck Residual Fusion
        
        # 2. Parallel Kidney Decoder Path D^(k)
        dk4 = self.dk4(b_fused, x4)
        dk3 = self.dk3(dk4, x3)
        dk2 = self.dk2(dk3, x2)
        dk1 = self.dk1(dk2, x1)
        k_logits = self.out_k(dk1)
        m_k = torch.sigmoid(k_logits)
        
        # 3. Parallel Stone Decoder Path D^(s)
        ds4 = self.ds4(b_fused, x4)
        ds3 = self.ds3(ds4, x3)
        ds2 = self.ds2(ds3, x2)
        ds1 = self.ds1(ds2, x1)
        s_logits = self.out_s(ds1)
        m_s = torch.sigmoid(s_logits)
        
        # 4. Mask-Guided Attention & Classification Head
        # Concatenate soft segmentation masks
        masks = torch.cat([m_k, m_s], dim=1)  # (B, 2, H, W)
        
        # Downsample masks to match bottleneck resolution
        masks_down = F.interpolate(masks, size=b_fused.shape[2:], mode='bilinear', align_corners=False)
        a_sp = self.g_mask(masks_down)  # (B, 1, H_bot, W_bot)
        
        # Spatial Modulation
        f_prime = b_fused * (1.0 + a_sp)
        
        # Channel Attention (Squeeze-and-Excitation)
        se_pool = self.gap(f_prime).view(f_prime.size(0), -1)
        se_weights = torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(se_pool)))).unsqueeze(-1).unsqueeze(-1)
        f_double_prime = f_prime * se_weights
        
        # Classification Output
        cls_feat = self.gap(f_double_prime).view(f_double_prime.size(0), -1)
        cls_logits = self.classifier(cls_feat)
        
        return {
            'kidney_logits': k_logits,
            'stone_logits': s_logits,
            'cls_logits': cls_logits,
            'm_k': m_k,
            'm_s': m_s
        }