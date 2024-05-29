# # Put these 4 lines at the top of the main script, before any imports
# TODO: remove once https://github.com/pytorch/pytorch/issues/109489 is resolved
# from torch._inductor import utils
# utils._use_template_for_cuda = lambda x, y: True

import math
import os
import pickle
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Annotated

import numpy as np
import safetensors.torch
import torch
from pydantic import PlainSerializer
from torch import Tensor

from config import Model, OptimizationConfig, RootConfig
from model import GPT


class Result(Model):
    iter: int
    loss: Annotated[float, PlainSerializer(lambda x: round(x, 6), return_type=float)]
    time: Annotated[float, PlainSerializer(lambda x: round(x, 2), return_type=float)]
    mfu: Annotated[float, PlainSerializer(lambda x: round(x, 2), return_type=float)] = float("-31337")
    max_alloc: Annotated[float, PlainSerializer(lambda x: round(x, 4), return_type=float)] = 0.0


def train(config: RootConfig, *, do_save: bool = False) -> list[Result]:  # noqa: C901, PLR0915, PLR0912
    if do_save:
        print("conventional dataloader, saving reproducible train data, ", end="")
    else:
        print("vram dataloader, loading reproducible train data, ", end="")
    print(f"tokens per iteration will be: {config.tokens_per_iter:,}")

    os.makedirs(config.train.out_dir, exist_ok=True)  # noqa: PTH103
    torch.manual_seed(config.train.seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    device_type = config.train.device_type  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[config.train.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.autocast(device_type=device_type, dtype=ptdtype)

    # poor man's data loader
    data_dir = os.path.join("data", config.train.dataset)  # noqa: PTH118

    if do_save:
        train_data_dict: dict[str, Tensor] = {}
    else:
        # load training data using safetensors
        train_data_dict = safetensors.torch.load_file(Path(f"tests/config/{config.name}_train.safetensors"), device=device_type)

    def get_batch(split: str, iter_num: int) -> tuple[Tensor, Tensor]:
        # print(f"get_batch({split}, {iter_num})")
        if do_save:
            # Recreate np.memmap every batch to avoid a memory leak, as per
            # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
            if split == "train":
                data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")  # noqa: PTH118
            else:
                data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")  # noqa: PTH118
            ix = torch.randint(len(data) - config.model.seq_len, (config.train.batch_size,))  # type: ignore  # noqa: PGH003
            x = torch.stack([torch.from_numpy((data[i : i + config.model.seq_len]).astype(np.int64)) for i in ix])  # type: ignore  # noqa: PGH003
            y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + config.model.seq_len]).astype(np.int64)) for i in ix])

            if split == "train" and f"batch_{iter_num}_x" not in train_data_dict:
                train_data_dict[f"batch_{iter_num}_x"] = x.detach().clone()
                train_data_dict[f"batch_{iter_num}_y"] = y.detach().clone()
                # train_data.append((x, y))

            if device_type == "cuda":
                # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
                x, y = x.pin_memory().to(device_type, non_blocking=True), y.pin_memory().to(device_type, non_blocking=True)
            else:
                x, y = x.to(device_type), y.to(device_type)
            return x, y

        return train_data_dict[f"batch_{iter_num}_x"], train_data_dict[f"batch_{iter_num}_y"]

    eval_iters = 20
    eval_interval = 250
    log_interval = 20

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

    if config.train.init_from == "scratch":
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        config.model.vocab_size = meta_vocab_size if meta_vocab_size is not None else 50304
        model: GPT = GPT(config=config.model)
    else:
        msg = f"Unsupported init_from value: {config.train.init_from}"
        raise Exception(msg)

    model.to(device_type)

    # report number of parameters
    print(f"number of parameters: {model.get_num_params() / 1e6:.2f}M")

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(config.train.dtype == "float16"))

    # optimizer
    optimizer = model.configure_optimizers(
        config.train.optimization.weight_decay,
        config.train.optimization.learning_rate.max,
        (config.train.optimization.optimizer.beta1, config.train.optimization.optimizer.beta2),
        device_type,
    )

    # compile the model
    # https://pytorch.org/docs/stable/generated/torch.compile.html
    # https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py
    if config.train.compile:
        print("compiling the model...")
        # unoptimized_model = model
        model: GPT = torch.compile(
            model,
            fullgraph=True,
            dynamic=False,
            mode="reduce-overhead",
            # mode="max-autotune",
        )  # type: ignore  # noqa: PGH003

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(iter_num: int):  # noqa: ANN202
        model.eval()
        out = {}
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split=split, iter_num=iter_num)  # noqa: N806
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()

            out[split] = losses.mean()

        model.train()

        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(iter_num: int, config: OptimizationConfig) -> float:
        if config.learning_rate.decay_schedule == "none":
            return config.learning_rate.max

        # 1) linear warmup for warmup_iters steps
        if iter_num < config.learning_rate.warmup_iters:
            return config.learning_rate.max * iter_num / config.learning_rate.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        lr_decay_iters = config.max_iters
        if iter_num > lr_decay_iters:
            return config.learning_rate.min
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (iter_num - config.learning_rate.warmup_iters) / (lr_decay_iters - config.learning_rate.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return config.learning_rate.min + coeff * (config.learning_rate.max - config.learning_rate.min)

    # training loop
    X, Y = get_batch(split="train", iter_num=0)  # fetch the very first batch  # noqa: N806

    def filter_dt(dt: list[float]) -> float:
        return min(dt)

    dts: list[float] = []
    results: list[Result] = []

    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    raw_model = model
    running_mfu = -1.0
    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num=iter_num, config=config.train.optimization)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and do_save:
            losses = estimate_loss(iter_num=iter_num)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    # TODO: save checkpoint
                    ...

        # forward backward update, using the GradScaler if data type is float16
        with ctx:
            logits, loss = model(X, Y)
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch("train", iter_num=iter_num + 1)  # noqa: N806
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()

        # clip the gradient
        if config.train.optimization.use_grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.optimization.grad_clip)
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
            lossf = loss.item()
            dt = filter_dt(dts)
            dts = []
            if local_iter_num >= 5:  # let the training loop settle a bit  # noqa: PLR2004
                mfu = raw_model.estimate_mfu(config.train.batch_size, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

            results.append(Result(iter=iter_num, loss=lossf, time=dt * 1000, mfu=running_mfu * 100, max_alloc=max_mem_alloc))
            # print(f"iter {iter_num}: loss {lossf:.6f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, max alloc {max_mem_alloc:.4f}GB")
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > config.train.optimization.max_iters:
            break

    # save training data using safetensors
    if do_save:
        safetensors.torch.save_file(train_data_dict, Path(f"tests/config/{config.name}_train.safetensors"))

    return results
