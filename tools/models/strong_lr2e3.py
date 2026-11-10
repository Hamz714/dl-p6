# %% Model and training settings

CONFIG = dict(task="ddpm", steps=20000, batch_size=64,
              learning_rate=0.002, weight_decay=0.0)
WARMUP, EMA_DECAYS = 500, (0.9995, 0.999, 0.9998, 0.9999)  # first EMA decay is copied into the model
CHANNELS, BLOCKS, ATTENTION = (32, 48, 48), 2, (16, 8)  # widths at 32/16/8 px, blocks per level, attention resolutions


def zero(module):
    nn.init.zeros_(module.weight)
    nn.init.zeros_(module.bias)
    return module


def norm(width):
    return nn.GroupNorm(math.gcd(32, width), width)


class Attention(nn.Module):
    def __init__(self, width, head=32):
        super().__init__()
        self.heads = width // head
        self.norm = norm(width)
        self.qkv = nn.Conv2d(width, 3 * width, 1)
        self.proj = zero(nn.Conv2d(width, width, 1))

    def forward(self, x):
        b, c, hgt, wid = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3 * self.heads, c // self.heads, hgt * wid).transpose(2, 3)
        h = F.scaled_dot_product_attention(*qkv.chunk(3, 1))
        return x + self.proj(h.transpose(2, 3).reshape(b, c, hgt, wid))


class Block(nn.Module):
    """Pre-activation residual block; time enters as adaptive GroupNorm scale and shift (FiLM)."""
    def __init__(self, cin, cout, emb, attention=False):
        super().__init__()
        self.norm1, self.norm2 = norm(cin), norm(cout)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.conv2 = zero(nn.Conv2d(cout, cout, 3, padding=1))
        self.film = nn.Linear(emb, 2 * cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.attention = Attention(cout) if attention else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(emb)[:, :, None, None].chunk(2, 1)
        h = self.conv2(F.silu(self.norm2(h) * (1 + scale) + shift))
        return self.attention(self.skip(x) + h)


class Model(nn.Module):
    """ADM/DDPM++-style U-Net: residual blocks, skip concatenation, attention at low resolution."""
    def __init__(self, channels=CHANNELS, blocks=BLOCKS, attention=ATTENTION):
        super().__init__()
        emb = 2 * channels[0]
        self.register_buffer("freqs", torch.exp(-math.log(1e4) * torch.arange(channels[0] // 2) / (channels[0] // 2)))
        self.time = nn.Sequential(nn.Linear(channels[0], emb), nn.SiLU(), nn.Linear(emb, emb), nn.SiLU())
        self.input = nn.Conv2d(3, channels[0], 3, padding=1)
        self.down, self.up, skips, width = nn.ModuleList(), nn.ModuleList(), [channels[0]], channels[0]
        for level, c in enumerate(channels):
            stage = nn.ModuleList()
            skips += [width] if level else []  # downsampled map is also a skip (ADM)
            for _ in range(blocks):
                stage.append(Block(width, c, emb, (32 >> level) in attention))
                width = c; skips.append(c)
            self.down.append(stage)
        self.middle = nn.ModuleList([Block(width, width, emb, True), Block(width, width, emb)])
        for level, c in reversed(list(enumerate(channels))):
            stage = nn.ModuleList()
            for _ in range(blocks + 1):
                stage.append(Block(width + skips.pop(), c, emb, (32 >> level) in attention))
                width = c
            self.up.append(stage)
        self.out = nn.Sequential(norm(width), nn.SiLU(), zero(nn.Conv2d(width, 3, 3, padding=1)))

    def forward(self, x, t, y=None):
        angles = 1000 * t[:, None] * self.freqs
        emb = self.time(torch.cat([angles.sin(), angles.cos()], 1))
        h = self.input(x)
        skips = [h]
        for level, stage in enumerate(self.down):
            if level:
                h = F.avg_pool2d(h, 2)
                skips.append(h)
            for block in stage:
                h = block(h, emb)
                skips.append(h)
        for block in self.middle:
            h = block(h, emb)
        for level, stage in enumerate(self.up):
            if level:
                h = F.interpolate(h, scale_factor=2, mode="nearest")
            for block in stage:
                h = block(torch.cat([h, skips.pop()], 1), emb)
        return self.out(h)


def build_model():
    return Model()

