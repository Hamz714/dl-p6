# %% Training
EXTRA_EMA = {}  # decay -> state_dict of the other tracked EMAs, evaluated separately by the experiment runner


def train(model, device):
    """AdamW with linear warm-up, bf16 autocast, torch.compile, horizontal flips and EMA of the weights."""
    images, labels = image_data()
    optimiser = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], betas=(0.9, 0.99),
                                  weight_decay=CONFIG["weight_decay"])
    warmup = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda step: min(1.0, (step + 1) / WARMUP))
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
        warmup.step()
        with torch.no_grad():
            for ema, decay in zip(emas, EMA_DECAYS):
                torch._foreach_lerp_(ema, params, 1 - decay)
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


