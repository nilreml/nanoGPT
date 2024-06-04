"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import inspect
import math
from types import EllipsisType
from typing import (
    TYPE_CHECKING,
    Generic,
    Literal,
    Self,
    SupportsIndex,
    TypeVar,
    TypeVarTuple,
    overload,
)

import torch
from torch import nn  # Tensor

from config import (
    Batch,
    DimHead,
    DimMLP,
    DimModel,
    DimModelX3,
    GPTConfig,
    Head,
    Token,
)

ellipsis = EllipsisType


In = TypeVar("In")
Out = TypeVar("Out")

Shape = TypeVarTuple("Shape")
ShapeRest = TypeVarTuple("ShapeRest")

First = TypeVar("First")
Second = TypeVar("Second")
Third = TypeVar("Third")

Last = TypeVar("Last")
LastButOne = TypeVar("LastButOne")
LastButTwo = TypeVar("LastButTwo")

AxisA = TypeVar("AxisA")
AxisB = TypeVar("AxisB")
AxisC = TypeVar("AxisC")

if TYPE_CHECKING:
    from torch._C import _NestedSequence
    from torch.nn.modules.normalization import _shape_t
    from torch.types import (
        _bool,
        _int,
    )

    class Tensor(Generic[*Shape]):
        def contiguous(self, memory_format=torch.contiguous_format) -> Self: ...

        def split(
            self: "Tensor[*ShapeRest, DimModelX3]",
            split_size: DimModel,
            dim: Literal[-1],
        ) -> tuple[
            "Tensor[*ShapeRest, DimModel]",
            "Tensor[*ShapeRest, DimModel]",
            "Tensor[*ShapeRest, DimModel]",
        ]: ...

        @overload
        def view(
            self: "Tensor[Batch, Token, Head, DimHead]",
            first: Batch,
            second: Token,
            third: DimModel,
        ) -> "Tensor[Batch, Token, DimModel]": ...
        @overload
        def view(
            self: "Tensor[Batch, Token, DimModel]",
            first: Batch,
            second: Token,
            third: Head,
            fourth: DimHead,
        ) -> "Tensor[Batch, Token, Head, DimHead]": ...
        def view(self: "Tensor", *args, **kwargs) -> "Tensor": ...

        def __getitem__(
            self,
            indices: SupportsIndex
            | None
            | _bool
            | _int
            | slice
            | ellipsis
            | "Tensor"
            | _NestedSequence[None | _bool | _int | slice | ellipsis | "Tensor"]
            | tuple[
                SupportsIndex
                | None
                | _bool
                | _int
                | slice
                | ellipsis
                | "Tensor"
                | _NestedSequence[None | _bool | _int | slice | ellipsis | "Tensor"],
                ...,
            ],
        ) -> "Tensor": ...

        @overload
        def size(self: "Tensor[First, *ShapeRest]", dim: Literal[0]) -> First: ...
        @overload
        def size(self: "Tensor[First,  Second, *ShapeRest]", dim: Literal[1]) -> Second: ...
        @overload
        def size(self: "Tensor[*ShapeRest, LastButOne]", dim: Literal[-2]) -> LastButOne: ...
        @overload
        def size(self: "Tensor[*ShapeRest, Last]", dim: Literal[-1]) -> Last: ...
        @overload
        def size(self) -> tuple[*Shape]: ...
        def size(self: "Tensor", dim: int) -> int: ...  # type:ignore  # noqa: PGH003

        # These shouldn't be used, always specify axes starting from the last one,
        # allowing an arbitrary number of leading dimensions
        # TODO: make use of torch's named axes
        # @overload
        # def transpose(
        #     self: "Tensor[First, Second, *ShapeRest]",
        #     dim0: Literal[0],
        #     dim1: Literal[1],
        # ) -> "Tensor[Second, First, *ShapeRest]": ...
        # @overload
        # def transpose(
        #     self: "Tensor[First, Second, Third, *ShapeRest]",
        #     dim0: Literal[1],
        #     dim1: Literal[2],
        # ) -> "Tensor[First, Third, Second, *ShapeRest]": ...
        @overload
        def transpose(
            self: "Tensor[*ShapeRest, LastButOne, Last]",
            dim0: Literal[-2],
            dim1: Literal[-1],
        ) -> "Tensor[*ShapeRest, Last, LastButOne]": ...
        @overload
        def transpose(
            self: "Tensor[*ShapeRest, LastButTwo, LastButOne, Last]",
            dim0: Literal[-3],
            dim1: Literal[-2],
        ) -> "Tensor[*ShapeRest, LastButOne, LastButTwo, Last]": ...
        def transpose(self, dim0: int, dim1: int) -> "Tensor": ...

        def matmul(
            self: "Tensor[*ShapeRest, AxisA, AxisB]",
            other: "Tensor[*ShapeRest, AxisB, AxisC]",
        ) -> "Tensor[*ShapeRest, AxisA, AxisC]": ...

        def __mul__(
            self,
            other: float,
        ) -> Self: ...

        @overload
        def __add__(
            self,
            other: float,
        ) -> Self: ...
        @overload
        def __add__(
            self,
            other: Self,
        ) -> Self: ...
        @overload
        def __add__(
            self: "Tensor[First, *ShapeRest]",
            other: "Tensor[*ShapeRest]",
        ) -> "Tensor[First, *ShapeRest]": ...
        @overload
        def __add__(  # TODO: report this pyright bug # type: ignore  # noqa: PGH003
            self: "Tensor[First, Second, *ShapeRest]",
            other: "Tensor[*ShapeRest]",
        ) -> "Tensor[First, Second, *ShapeRest]": ...
        def __add__(
            self,
            other,
        ) -> "Tensor": ...

    def softmax(
        x: Tensor[*Shape],  # noqa: ARG001
        dim: int,  # noqa: ARG001
    ) -> Tensor[*Shape]: ...

    class Linear(Generic[In, Out]):  # nn.Module
        in_features: In
        out_features: Out
        weight: Tensor

        def __new__(cls, in_features: In, out_features: Out, bias: bool = True, device=None, dtype=None) -> "Linear[In, Out]": ...  # noqa: PLR0913, FBT001, FBT002
        def __init__(self, in_features: In, out_features: Out, bias: bool = True, device=None, dtype=None) -> None: ...  # noqa: PLR0913, FBT001, FBT002

        def forward(self, input: Tensor[*Shape, In]) -> Tensor[*Shape, Out]: ...
        def __call__(self, input: Tensor[*Shape, In]) -> Tensor[*Shape, Out]: ...

    class Dropout(nn.Module):
        p: float
        inplace: bool

        def __init__(self, p: float, inplace: bool = False) -> None: ...  # noqa: FBT002, FBT001
        def forward(self, input: Tensor[*Shape]) -> Tensor[*Shape]: ...
        def __call__(self, input: Tensor[*Shape]) -> Tensor[*Shape]: ...

    class GELU(nn.Module):
        approximate: str

        def __init__(self, approximate: str = "none") -> None: ...
        def forward(self, input: Tensor[*Shape]) -> Tensor[*Shape]: ...
        def __call__(self, input: Tensor[*Shape]) -> Tensor[*Shape]: ...

    class LayerNorm(nn.Module):
        normalized_shape: tuple[int, ...]
        eps: float
        elementwise_affine: bool

        def __init__(  # noqa: PLR0913
            self,
            normalized_shape: _shape_t,
            eps: float = 1e-5,
            elementwise_affine: bool = True,  # noqa: FBT001, FBT002
            bias: bool = True,  # noqa: FBT002, FBT001
            device=None,
            dtype=None,
        ) -> None: ...

        def forward(self, input: Tensor[*Shape]) -> Tensor[*Shape]: ...
        def __call__(self, input: Tensor[*Shape]) -> Tensor[*Shape]: ...
else:
    # from torch import Tensor
    from torch.nn import GELU, Dropout, LayerNorm, Linear
    from torch.nn.functional import softmax

    class Tensor(Generic[*Shape], torch.Tensor): ...


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.head: Head = config.num_heads
        self.dim_model: DimModel = config.dim_model
        self.dim_head: DimHead = config.dim_head
        self.dim_model_x3: DimModelX3 = config.dim_model_x3
        self.dropout = config.dropout

        # query, key, value projections for all heads in a batch
        self.qkv_proj: Linear[DimModel, DimModelX3] = Linear(self.dim_model, self.dim_model_x3, bias=config.bias)

        # output projection
        self.out_proj: Linear[DimModel, DimModel] = Linear(config.dim_model, config.dim_model, bias=config.bias)

        # regularization
        self.attn_dropout = Dropout(config.dropout)
        self.resid_dropout = Dropout(config.dropout)

        # attention bias
        self.attn_bias: Tensor[Token, Token]
        self.register_buffer(
            "attn_bias",
            # causal: bias attention to future tokens with negative infinity
            torch.tril(torch.ones(config.seq_len, config.seq_len) * float("-inf"), diagonal=-1).transpose(-2, -1),
            persistent=False,
        )

    def forward(self, x: Tensor[Batch, Token, DimModel]) -> Tensor[Batch, Token, DimModel]:
        # shorthand names for tensor axes
        batch, token, dim_model = x.size()  # batch size, sequence length, model dimensions
        head = self.head  # number of heads
        dim_head = self.dim_head  # head dimensions = model dimensions / number of heads

        # project query, key, value for all heads in a batch
        qkv = self.qkv_proj(x)

        # split last axis into query, key, value
        q, k, v = qkv.split(dim_model, dim=-1)

        # split last axis into heads with head dimensions, then swap token and head axes
        q = q.view(batch, token, head, dim_head).transpose(-3, -2)
        k = k.view(batch, token, head, dim_head).transpose(-3, -2)
        v = v.view(batch, token, head, dim_head).transpose(-3, -2)

        # causal multi-head self-attention: Softmax( QKᵀ / ⎷dₕ ) V
        logits = q.matmul(k.transpose(-2, -1)) * (1.0 / math.sqrt(dim_head))  # QKᵀ / ⎷dₕ

        logits += self.attn_bias  # apply attention bias (causal or otherwise)  [:, :, :T, :T]

        scores = softmax(logits, dim=-1)  # apply softmax on last axis
        scores = self.attn_dropout(scores)
        y = scores.matmul(v)  # [*, *, Token, Token] x [*, *, Token, DimHead] -> [*, *, Token, DimHead]

        # re-assemble all head outputs
        # TODO: benchmark contiguous() in different places
        y = y.transpose(-3, -2).contiguous()  # swap token and head axes again
        y = y.view(batch, token, dim_model)  # collapse last two axes to put head outputs side-by-side

        # project output
        z = self.out_proj(y)

        return self.resid_dropout(z)

    if TYPE_CHECKING:

        def __call__(self, x: Tensor[Batch, Token, DimModel]) -> Tensor[Batch, Token, DimModel]: ...


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.in_proj: Linear[DimModel, DimMLP] = Linear(config.dim_model, config.dim_mlp, bias=config.bias)  # input projection
        self.activation = GELU()  # activation function
        self.out_proj: Linear[DimMLP, DimModel] = Linear(config.dim_mlp, config.dim_model, bias=config.bias)  # output projection
        self.dropout = Dropout(config.dropout)

    def forward(self, x: Tensor[Batch, Token, DimModel]) -> Tensor[Batch, Token, DimModel]:
        y = self.in_proj(x)
        y = self.activation(y)
        z = self.out_proj(y)
        return self.dropout(z)


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.dim_model, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.dim_model, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: Tensor[Batch, Token, DimModel]) -> Tensor[Batch, Token, DimModel]:
        # Pre-LN, sequential att and mlp
        # x_norm = self.ln_1(x)
        # att = self.attn(x_norm)
        # att_resid = x + att
        # att_norm = self.ln_2(att_resid)
        # return att_norm + self.mlp(att_norm)

        y = x + self.attn(self.ln_1(x))
        return y + self.mlp(self.ln_2(y))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.dim_model),  # token embedding
                "wpe": nn.Embedding(config.seq_len, config.dim_model),  # position embedding
                "drop": Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.num_layers)]),  # layers
                "ln_f": LayerNorm(config.dim_model, bias=config.bias),  # final layernorm
            },
        )
        self.lm_head = Linear(config.dim_model, config.vocab_size, bias=False)  # token un-embedding

        # https://paperswithcode.com/method/weight-tying
        self.transformer.wte.weight = self.lm_head.weight  # tie token embedding and un-embedding

        # init all weights
        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        # TODO: be explicit about which modules are affected
        # TODO: delegate initialization to each module
        for param_name, param in self.named_parameters():
            if param_name.endswith("out_proj.weight"):
                nn.init.normal_(
                    param,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * config.num_layers),
                )

    def _init_weights(self, module) -> None:
        if isinstance(module, Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)  # TODO: remove # type:ignore  # noqa: PGH003
            if module.bias is not None:  # TODO: remove # type:ignore  # noqa: PGH003
                nn.init.zeros_(module.bias)  # TODO: remove # type:ignore  # noqa: PGH003
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.seq_len, f"Cannot forward sequence of length {t}, max. sequence length is {self.config.seq_len}"

        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, dim_model)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, dim_model)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),  # TODO: remove # type:ignore  # noqa: PGH003
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            # note: using list [-1] to preserve the time dim
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    # def crop_seq_len(self, seq_len):
    #     # model surgery to decrease the block size if necessary
    #     # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
    #     # but want to use a smaller block size for some smaller, simpler model
    #     assert seq_len <= self.config.seq_len
    #     self.config.seq_len = seq_len
    #     self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:seq_len])
    #     for block in self.transformer.h:
    #         if hasattr(block.attn, "bias"):
    #             block.attn.bias = block.attn.bias[:, :, :seq_len, :seq_len]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = dict(self.named_parameters())

        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]  # noqa: PLR2004
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]  # noqa: PLR2004
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = {"fused": True} if use_fused else {}
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            **extra_args,
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def get_num_params(self, non_embedding=True):  # noqa: FBT002
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of 3080 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()  # noqa: N806
        cfg = self.config

        L, H, Q, T = cfg.num_layers, cfg.num_heads, cfg.dim_model // cfg.num_heads, cfg.seq_len  # noqa: N806
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 60e12  # 3080 GPU bfloat16 peak flops
        # flops_promised = 312e12  # A100 GPU bfloat16 peak flops

        return flops_achieved / flops_promised

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at seq_len
            idx_cond = idx if idx.size(1) <= self.config.seq_len else idx[:, -self.config.seq_len :]

            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # apply softmax to convert logits to (normalized) probabilities
            probs = softmax(logits, dim=-1)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # TODO: remove # type:ignore  # noqa: PGH003

            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
