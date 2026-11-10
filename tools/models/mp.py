# %% Model and training settings

CONFIG = dict(task="flow", steps=20000, batch_size=64,
              learning_rate=0.01, weight_decay=0.0)
WARMUP, EMA_DECAYS = 500, (0.9995, 0.999, 0.9998, 0.9999)  # first EMA decay is copied into the model
CHANNELS, BLOCKS, ATTENTION = (32, 48, 48), 2, (16, 8)  # widths at 32/16/8 px, blocks per level, attention resolutions


# Magnitude-preserving layers after EDM2 (Karras et al. 2024): weights are kept at unit norm per output
# channel, activations keep unit magnitude through SiLU, residual sums and skip concatenation.
def normalize(x, dim=None, eps=1e-4):
    dim = list(range(1, x.ndim)) if dim is None else dim
    n = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    return x / torch.add(eps, n, alpha=math.sqrt(n.numel() / x.numel())).to(x.dtype)


def mp_silu(x):
    return F.silu(x) / 0.596


def mp_sum(a, b, t=0.3):
    return a.lerp(b, t) / math.sqrt((1 - t) ** 2 + t ** 2)


def mp_cat(a, b, t=0.5):
    c = math.sqrt((a.shape[1] + b.shape[1]) / ((1 - t) ** 2 + t ** 2))
    return torch.cat([a * (c * (1 - t) / math.sqrt(a.shape[1])), b * (c * t / math.sqrt(b.shape[1]))], 1)


class MPConv(nn.Module):
    """Convolution (or linear layer when kernel=()) with unit-norm weights; train() re-projects them each update."""
    def __init__(self, cin, cout, kernel=(3, 3)):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(cout, cin, *kernel))

    def forward(self, x, gain=1):
        w = normalize(self.weight) * (gain / math.sqrt(self.weight[0].numel()))
        w = w.to(x.dtype)
        return x @ w.t() if w.ndim == 2 else F.conv2d(x, w, padding=w.shape[-1] // 2)


class Block(nn.Module):
    """EDM2-style residual block: pixel-normalised encoder input, multiplicative time conditioning, mp_sum residual."""
    def __init__(self, cin, cout, emb, attention=False, encoder=True, head=32):
        super().__init__()
        self.encoder, self.heads = encoder, cout // head if attention else 0
        self.skip = MPConv(cin, cout, (1, 1)) if cin != cout else None
        self.conv1 = MPConv(cout if encoder else cin, cout)
        self.conv2 = MPConv(cout, cout)
        self.emb, self.emb_gain = MPConv(emb, cout, ()), nn.Parameter(torch.zeros([]))
        if attention:
            self.qkv, self.proj = MPConv(cout, 3 * cout, (1, 1)), MPConv(cout, cout, (1, 1))

    def forward(self, x, emb):
        if self.encoder:
            x = normalize(self.skip(x) if self.skip else x, dim=1)
        y = self.conv1(mp_silu(x))
        y = self.conv2(mp_silu(y * (self.emb(emb, self.emb_gain) + 1)[:, :, None, None]))
        if not self.encoder and self.skip:
            x = self.skip(x)
        x = mp_sum(x, y)
        if self.heads:
            b, c, hgt, wid = x.shape
            q, k, v = normalize(self.qkv(x).reshape(b, self.heads, c // self.heads, 3, hgt * wid), dim=2).unbind(3)
            y = F.scaled_dot_product_attention(q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3))
            x = mp_sum(x, self.proj(y.transpose(2, 3).reshape(b, c, hgt, wid)))
        return x


class Model(nn.Module):
    """U-Net with EDM2 magnitude-preserving layers (no normalisation layers, no biases), ADM skip layout."""
    def __init__(self, channels=CHANNELS, blocks=BLOCKS, attention=ATTENTION):
        super().__init__()
        emb = 2 * channels[0]
        self.register_buffer("freqs", torch.exp(-math.log(1e4) * torch.arange(channels[0] // 2) / (channels[0] // 2)))
        self.time = MPConv(channels[0], emb, ())
        self.input = MPConv(4, channels[0])  # image plus a constant channel, as in EDM2
        self.down, self.up, skips, width = nn.ModuleList(), nn.ModuleList(), [channels[0]], channels[0]
        for level, c in enumerate(channels):
            stage = nn.ModuleList()
            skips += [width] if level else []  # downsampled map is also a skip (ADM)
            for _ in range(blocks):
                stage.append(Block(width, c, emb, (32 >> level) in attention))
                width = c; skips.append(c)
            self.down.append(stage)
        self.middle = nn.ModuleList([Block(width, width, emb, True, False), Block(width, width, emb, False, False)])
        for level, c in reversed(list(enumerate(channels))):
            stage = nn.ModuleList()
            for _ in range(blocks + 1):
                stage.append(Block(width + skips.pop(), c, emb, (32 >> level) in attention, False))
                width = c
            self.up.append(stage)
        self.out, self.out_gain = MPConv(width, 3), nn.Parameter(torch.zeros([]))

    def forward(self, x, t, y=None):
        angles = 1000 * t[:, None] * self.freqs
        emb = mp_silu(self.time(torch.cat([angles.sin(), angles.cos()], 1) * math.sqrt(2)))
        h = self.input(torch.cat([x, torch.ones_like(x[:, :1])], 1))
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
                h = block(mp_cat(h, skips.pop()), emb)
        return self.out(h, self.out_gain)


def build_model():
    return Model()

