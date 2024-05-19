"""
To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

# # Put these 4 lines at the top of the main script, before any imports
# TODO: remove once https://github.com/pytorch/pytorch/issues/109489 is resolved
# from torch._inductor import utils
# utils._use_template_for_cuda = lambda x, y: True

import math
import os
import pickle
import time
from contextlib import nullcontext
from datetime import datetime

import dill
import numpy as np
import torch

from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = "out"
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
# data
dataset = "openwebtext"
gradient_accumulation_steps = 5 * 8  # used to simulate larger batch sizes
batch_size = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
seq_len = 1024
# model
num_layers = 12
num_heads = 12
dim_model = 768
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# system
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = (
    "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True
# random seed
seed_offset = 0
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, int | float | bool | str)]
exec(open("configurator.py").read())  # overrides from command line or config file  # noqa: SIM115, PTH123, S102
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
tokens_per_iter = gradient_accumulation_steps * batch_size * seq_len
print(f"tokens per iteration will be: {tokens_per_iter:,}")

os.makedirs(out_dir, exist_ok=True)  # noqa: PTH103
torch.manual_seed(seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join("data", dataset)  # noqa: PTH118

print(torch.backends.cuda.matmul.__dict__)


def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == "train":
        data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")  # noqa: PTH118
    else:
        data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")  # noqa: PTH118
    ix = torch.randint(len(data) - seq_len, (batch_size,))  # type: ignore  # noqa: PGH003
    x = torch.stack([torch.from_numpy((data[i : i + seq_len]).astype(np.int64)) for i in ix])  # type: ignore  # noqa: PGH003
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + seq_len]).astype(np.int64)) for i in ix])
    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, "meta.pkl")  # noqa: PTH118
meta_vocab_size = None
if os.path.exists(meta_path):  # noqa: PTH110
    with open(meta_path, "rb") as f:  # noqa: PTH123
        meta = pickle.load(f)  # noqa: S301
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = {
    "num_layers": num_layers,
    "num_heads": num_heads,
    "dim_model": dim_model,
    "n_mlp": 3 * dim_model,
    "seq_len": seq_len,
    "bias": bias,
    "vocab_size": None,
    "dropout": dropout,
    "use_flash": False,
}

checkpoint = None

if init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model: GPT = GPT(gptconf)
else:
    msg = f"Unsupported init_from value: {init_from}"
    raise Exception(msg)

if seq_len < model.config.seq_len:
    msg = "seq_len cannot be smaller than the model's sequence length"
    raise Exception(msg)

model.to(device)

# report number of parameters
print(f"number of parameters: {model.get_num_params() / 1e6:.2f}M")

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

checkpoint = None  # free up memory

# compile the model
# https://pytorch.org/docs/stable/generated/torch.compile.html
# https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model: GPT = torch.compile(
        model,
        fullgraph=True,
        dynamic=False,
        mode="reduce-overhead",
        # mode="max-autotune",
    )  # type: ignore  # noqa: PGH003


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)  # noqa: N806
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()

    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# training loop
X, Y = get_batch("train")  # fetch the very first batch


def filter_dt(dt: list[float]) -> float:
    return min(dt)


dts: list[float] = []

t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model
running_mfu = -1.0
while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                print(f"{datetime.now()} torch.save checkpoint to {out_dir}")  # noqa: DTZ005
                torch.save(
                    checkpoint,
                    os.path.join(out_dir, "ckpt.pt"),  # noqa: PTH118
                    pickle_module=dill,
                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                )

    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):  # noqa: B007
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps  # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch("train")
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()

    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    torch.cuda.synchronize()
    t1 = time.time()
    dts.append(t1 - t0)
    t0 = t1
    max_mem_alloc = torch.cuda.max_memory_allocated() / (1024**3)
    torch.cuda.reset_peak_memory_stats()
    if iter_num % log_interval == 0:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps  # type: ignore  # noqa: PGH003
        dt = filter_dt(dts)
        dts = []
        if local_iter_num >= 5:  # let the training loop settle a bit  # noqa: PLR2004
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            f"iter {iter_num}: loss {lossf:.6f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, max alloc {max_mem_alloc:.4f}GB",
        )
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break
