# Small RGB U-Net for CIFAR-10 horses, using time-conditioned residual blocks for DDPM or image flow.
# Set STUDENT_ID, then run this file or paste it into one notebook cell.
import ast
import hashlib
import json
import math
import random
import re
import zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch_fidelity import calculate_metrics  # pip install torch-fidelity
from torch.nn import functional as F

STUDENT_ID = "xmzj54"
DATA_DIR = Path("data")

# %% Model and training settings

CONFIG = dict(task="flow", steps=20000, batch_size=64,
              learning_rate=0.01, weight_decay=0.0)
WARMUP, T_REF = 500, 1500  # warm-up updates; LR ~ 1/sqrt(step/T_REF) after T_REF (EDM2 schedule)
EMA_DECAYS = (0.9995, 0.999, 0.9998, 0.9999)  # first EMA decay is copied into the model
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

MAX_PARAMETERS, MAX_STEPS, MAX_BATCH = (1000000, 20000, 64)

# %% Data, evaluation and submission
def program_source():
    if "__file__" in globals():
        source = Path(__file__).read_text()
    else:
        source = get_ipython().history_manager.input_hist_raw[-1].rstrip() + "\n"
    tree = ast.parse(source)
    names = {getattr(node, "name", "") for node in tree.body}
    if not {"build_model", "run", "save_submission"} <= names:
        raise ValueError("Paste the complete program into one notebook cell and run that cell.")
    return source


def model_size(model):
    parameters = sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters())
    buffers = sum(b.numel()*(2 if b.is_complex() else 1) for b in model.buffers())
    if parameters + buffers > MAX_PARAMETERS:
        raise ValueError(f"Model has {parameters + buffers} values; limit is {MAX_PARAMETERS}.")
    return parameters, buffers


def save_submission(folder, source, history, metrics, size, seed, split):
    stem = "lab-06"
    code = folder / (stem + "-code.py")
    code.write_text(source)
    names = [f"batch-{i:02d}.png" for i in range(1, 11)]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    updates = {"model": CONFIG["steps"]}
    result = dict(format=6, lab=6, seed=seed, config=dict(CONFIG),
                  parameters=size[0], buffer_elements=size[1], updates=updates,
                  protocol="comp3801-cifar10-horses-kid1000-v1-" + split, code_sha256=digest(code),
                  metrics=metrics, loss=history,
                  samples_sha256={name: digest(folder/"samples"/name) for name in names})
    record = folder / (stem + "-results.json")
    record.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    archive = folder.with_name(folder.name + ".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as pack:
        for path in [code, record] + [folder/"samples"/name for name in names]:
            pack.write(path, path.relative_to(folder))
    print(json.dumps(metrics, indent=2))
    print("Submission:", archive.resolve())
    return result


def show_results(history, folder):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 2.5))
    curves = np.array(history)
    plt.plot(curves[:, 0], curves[:, 1])
    plt.xlabel("Update"); plt.ylabel("Loss"); plt.tight_layout(); plt.show()
    from PIL import Image
    plt.figure(figsize=(6, 6))
    plt.imshow(Image.open(folder/"samples/batch-01.png"))
    plt.axis("off"); plt.tight_layout(); plt.show()


def image_data(split="train"):
    from torchvision.datasets import CIFAR10
    dataset = CIFAR10(root=DATA_DIR, train=split != "test", download=True)
    index = torch.where(torch.tensor(dataset.targets) == 7)[0]
    if split != "test":
        order = torch.randperm(len(index), generator=torch.Generator().manual_seed(3802))
        index = index[order[:4000] if split == "train" else order[4000:]]
    images = torch.from_numpy(dataset.data).permute(0, 3, 1, 2)[index].float()/127.5 - 1
    return images, torch.zeros(len(images), dtype=torch.long)


def image_bytes(images):
    return ((images.clamp(-1, 1) + 1)*127.5 + .5).floor().to(torch.uint8)


class Images(torch.utils.data.Dataset):
    """Give the KID library one uint8 RGB image at a time."""
    def __init__(self, pixels): self.pixels = pixels
    def __len__(self): return len(self.pixels)
    def __getitem__(self, index): return self.pixels[index]


@torch.no_grad()
def image_score(model, folder, device, split):
    from PIL import Image
    model.eval()
    rng = torch.Generator(device=device).manual_seed(921)
    fake = torch.cat([generate(model, 50, device, rng).clamp(-1, 1).cpu() for _ in range(20)])
    real, _ = image_data(split)
    pixels = image_bytes(fake)
    # One full set of 1,000 images, using the same pixels as the saved grids.
    metrics = calculate_metrics(
        input1=Images(image_bytes(real)), input2=Images(pixels),
        cuda=torch.device(device).type == "cuda", batch_size=32,
        kid=True, kid_subset_size=1000, kid_subsets=1,
        feature_extractor="inception-v3-compat", feature_layer_kid="2048",
        rng_seed=921, save_cpu_ram=True, verbose=False)
    score = metrics["kernel_inception_distance_mean"]
    for i, batch in enumerate(pixels.split(100), 1):
        grid = batch.reshape(10, 10, 3, 32, 32).permute(0, 3, 1, 4, 2).reshape(320, 320, 3)
        Image.fromarray(grid.numpy()).save(folder/f"batch-{i:02d}.png")
    print(f"KID x 1000: {1000*score:.3f}")
    return {"kid": score, "samples": 1000}


def diffusion_schedule(steps=100, device='cpu'):
    # The cosine schedule approaches pure noise within 100 diffusion steps.
    grid = torch.linspace(0, 1, steps+1, device=device, dtype=torch.float64)
    cumulative = torch.cos((grid+.008)/1.008 * torch.pi/2).square()
    cumulative = cumulative/cumulative[0]
    betas = (1-cumulative[1:]/cumulative[:-1]).clamp(max=.999).float()
    alphas = 1-betas
    bars = torch.cumprod(alphas, dim=0)
    previous = torch.cat([torch.ones(1, device=device), bars[:-1]])
    variance = betas*(1-previous)/(1-bars)
    return betas, alphas, bars, variance

def add_noise(clean, epsilon, t, bars):
    a = bars[t].reshape(-1, 1, 1, 1)
    return a.sqrt()*clean + (1-a).sqrt()*epsilon

def diffusion_loss(model, clean, bars, labels=None):
    t = torch.randint(len(bars), (len(clean),), device=clean.device)
    epsilon = torch.randn_like(clean)
    noisy = add_noise(clean, epsilon, t, bars)
    # Integer index 0 denotes mathematical diffusion step 1.
    prediction = model(noisy, (t.float()+1)/len(bars), labels) if labels is not None else model(noisy, (t.float()+1)/len(bars))
    return F.mse_loss(prediction, epsilon)

def flow_path(noise, data, t):
    t = t.reshape(-1, 1, 1, 1)
    return (1-t)*noise+t*data, data-noise

def flow_loss(model, data, labels):
    noise = torch.randn_like(data)
    t = torch.rand(len(data), device=data.device)
    x, target = flow_path(noise, data, t)
    return F.mse_loss(model(x, t, labels), target)

@torch.no_grad()
def sample_ddpm(model, count, schedule, device='cpu', generator=None, labels=None, shape=(3, 32, 32)):
    betas, alphas, bars, variance = schedule
    x = torch.randn(count, *shape, device=device, generator=generator)
    for i in reversed(range(len(betas))):
        t = torch.full((count,), (i+1)/len(betas), device=device)
        epsilon = model(x, t, labels) if labels is not None else model(x, t)
        # Predict clean x0, clip to the known data support, then use q(x_{t-1}|x_t,x0).
        clean = ((x-(1-bars[i]).sqrt()*epsilon)/bars[i].sqrt()).clamp(-1, 1)
        prev = bars[i-1] if i else torch.ones((), device=device)
        mean = (prev.sqrt()*betas[i]/(1-bars[i]))*clean
        mean = mean + (alphas[i].sqrt()*(1-prev)/(1-bars[i]))*x
        if i:
            noise = torch.randn(x.shape, device=device, generator=generator)
            x = mean + variance[i].sqrt()*noise
        else:
            x = mean
    return x.clamp(-1, 1)

@torch.no_grad()
def sample_flow(model, labels, steps=25, device='cpu', generator=None, shape=(3, 32, 32)):
    if steps < 1:
        raise ValueError('sampling steps must be positive')
    x = torch.randn(len(labels), *shape, device=device, generator=generator)
    dt = 1/steps
    for i in range(steps):
        t = torch.full((len(labels),), i/steps, device=device)
        x = x + dt*model(x, t, labels)
    return x.clamp(-1, 1)

# %% Training
EXTRA_EMA = {}  # decay -> state_dict of the other tracked EMAs, evaluated separately by the experiment runner


def train(model, device):
    """AdamW with linear warm-up then inverse-square-root LR decay, bf16 autocast, torch.compile, horizontal flips, forced weight normalisation and warm-started EMA of the weights."""
    images, labels = image_data()
    optimiser = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], betas=(0.9, 0.99),
                                  weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda step: min((step + 1) / WARMUP, (T_REF / max(step, T_REF)) ** 0.5))
    schedule = diffusion_schedule(device=device)
    params = list(model.parameters())
    emas = [[p.detach().clone() for p in params] for _ in EMA_DECAYS]
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    net = torch.compile(model)  # shares parameters with model; generate() uses the plain model
    history = []
    model.train()
    for step in range(1, CONFIG["steps"] + 1):
        index = torch.randint(len(images), (CONFIG["batch_size"],))
        x, y = images[index].to(device), labels[index].to(device)
        x = torch.where(torch.rand(len(x), 1, 1, 1, device=device) < 0.5, x.flip(3), x)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            loss = diffusion_loss(net, x, schedule[2], y) if CONFIG["task"] == "ddpm" else flow_loss(net, x, y)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        optimiser.step()
        scheduler.step()
        with torch.no_grad():  # forced weight normalisation (EDM2): keep every MPConv weight at unit scale
            for m in model.modules():
                if isinstance(m, MPConv):
                    m.weight.copy_(normalize(m.weight))
        with torch.no_grad():
            for ema, decay in zip(emas, EMA_DECAYS):  # running mean until 1/(1-decay) updates: no init bias
                torch._foreach_lerp_(ema, params, max(1 - decay, 1 / (step + 1)))
        if step == 1 or step % max(1, CONFIG["steps"]//40) == 0 or step == CONFIG["steps"]:
            history.append([step, round(loss.item(), 6)])
        if step == 1 or step % max(1, CONFIG["steps"]//10) == 0 or step == CONFIG["steps"]:
            print(f"{step} / {CONFIG['steps']}: loss {loss.item():.4f}", flush=True)
    names = [name for name, _ in model.named_parameters()]
    for ema, decay in list(zip(emas, EMA_DECAYS))[1:]:
        EXTRA_EMA[decay] = {**model.state_dict(), **dict(zip(names, ema))}
    with torch.no_grad():
        for p, e in zip(params, emas[0]):
            p.copy_(e)
    return history


@torch.no_grad()
def generate(model, count, device, generator):
    labels = torch.zeros(count, dtype=torch.long, device=device)
    if CONFIG["task"] == "ddpm":
        return sample_ddpm(model, count, diffusion_schedule(device=device), device,
                           generator, labels, shape=(3, 32, 32))
    return sample_flow(model, labels, 25, device, generator, shape=(3, 32, 32))

# %% Run the experiment
def run(out=None, device=None, seed=3801, split="validation", show=True):
    source = program_source()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", STUDENT_ID):
        raise ValueError("Use letters, digits, underscore or hyphen for STUDENT_ID.")
    if CONFIG["task"] not in ("ddpm", "flow"):
        raise ValueError("Set task to 'ddpm' or 'flow'.")
    if not 1 <= CONFIG["steps"] <= MAX_STEPS or not 1 <= CONFIG["batch_size"] <= MAX_BATCH:
        raise ValueError("Training steps or batch size exceeds the practical limit.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    if torch.device(device).type == "cuda":
        torch.backends.cudnn.benchmark = True
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    folder = Path(out or f"lab-06-{STUDENT_ID}")
    (folder/"samples").mkdir(parents=True, exist_ok=True)
    model = build_model().to(device)
    size = model_size(model)
    history = train(model, device)
    metrics = image_score(model, folder/"samples", device, split)
    result = save_submission(folder, source, history, metrics, size, seed, split)
    if show:
        show_results(history, folder)
    return result


if __name__ == "__main__":
    run()
