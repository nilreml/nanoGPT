"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
from types import EllipsisType
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    Self,
    SupportsIndex,
    TypeVar,
    TypeVarTuple,
    overload,
)

import torch
from pydantic import BaseModel, computed_field, model_validator
from torch import nn  # Tensor
from torch.nn import functional as F  # noqa: N812

ellipsis = EllipsisType

Head = Literal[6]
Batch = Literal[20]
Token = Literal[64]
Embed = Literal[192]
EmbedPerHead = Literal[32]
EmbedX3 = Literal[576]

InFeat = Embed
OutFeat = EmbedX3
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

# class Tensor(Generic[*Shape]): ...

# @overload
# def transpose(
#             self: "Tensor[First, Second, *ShapeRest]",
#             dim0: Literal[1],
#             dim1: Literal[2],
#         ) -> "Tensor[Second, First, *ShapeRest]":
#             ...


if TYPE_CHECKING:
    # class Tensor[](torch.Tensor):

    from torch._C import _NestedSequence
    from torch.types import (
        _bool,
        _int,
    )

    class Tensor(Generic[*Shape]):
        # def __init__(self, shape: tuple[*Shape]) -> None:
        #     self._shape: tuple[*Shape] = shape

        # def get_shape(self) -> tuple[*Shape]:
        #     return self._shape

        def contiguous(self, memory_format=torch.contiguous_format) -> Self: ...

        def view(
            self: "Tensor[Batch, Token, Head, EmbedPerHead]",
            first: Batch,
            second: Token,
            third: Embed,
        ) -> "Tensor[Batch, Token, Embed]": ...

        def split(
            self: "Tensor[*ShapeRest, EmbedX3]",
            split_size: Embed,
            dim: Literal[-1],
        ) -> tuple[
            "Tensor[*ShapeRest, Embed]",
            "Tensor[*ShapeRest, Embed]",
            "Tensor[*ShapeRest, Embed]",
        ]: ...

        # def split_qkv(
        #     self,
        #     x: Tensor[Batch, Token, EmbedX3],
        # ) -> tuple[
        #     Tensor[Batch, Token, Embed],
        #     Tensor[Batch, Token, Embed],
        #     Tensor[Batch, Token, Embed],
        # ]:
        #     return x.split(self.n_embd, dim=-1)

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
        def size(self: "Tensor[Any]", dim: int) -> int: ...  # type:ignore  # noqa: PGH003

        @overload
        # def size(self: "Tensor[Any]") -> tuple[int]: ...
        def transpose(
            self: "Tensor[First, Second, *ShapeRest]",
            dim0: Literal[0],
            dim1: Literal[1],
        ) -> "Tensor[Second, First, *ShapeRest]": ...

        @overload
        def transpose(
            self: "Tensor[First, Second, Third, *ShapeRest]",
            dim0: Literal[1],
            dim1: Literal[2],
        ) -> "Tensor[First, Third, Second, *ShapeRest]": ...

        @overload
        def transpose(
            self: "Tensor[*ShapeRest, LastButOne, Last]",
            dim0: Literal[-2],
            dim1: Literal[-1],
        ) -> "Tensor[*ShapeRest, Last, LastButOne]": ...

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

        # @overload
        # def __add__(
        #     self: "Tensor[First, Second, *ShapeRest]",
        #     other: "Tensor[Literal[1], Literal[1], *ShapeRest]",
        # ) -> "Tensor[First, Second, *ShapeRest]": ...

        # @overload
        # def __add__(
        #     self: "Tensor[Literal[1], Literal[1], *ShapeRest]",
        #     other: "Tensor[First, Second, *ShapeRest]",
        # ) -> "Tensor[First, Second, *ShapeRest]": ...

        def __add__(
            self,
            other,
        ) -> "Tensor": ...

    #     # @overload
    #     # @override
    #     # def transpose(self: "Tensor[First, Second]", dim0: Literal[2], dim1: Literal[2]) -> "Tensor[Second, First]":
    #     #     return super().transpose(1, 2)  # type: ignore  # noqa: PGH003

    def softmax(
        x: Tensor[*Shape],
        dim: int,
    ) -> Tensor[*Shape]: ...

    # class Linear(Generic[In, Out]):  # nn.Module
    # class Linear(In, Out):
    class Linear(Generic[In, Out]):  # nn.Module
        in_features: In
        out_features: Out
        weight: Tensor

        def __new__(cls, in_features: In, out_features: Out, bias: bool = True, device=None, dtype=None) -> "Linear[In, Out]": ...  # noqa: PLR0913

        def __init__(self, in_features: In, out_features: Out, bias: bool = True, device=None, dtype=None) -> None:  # noqa: PLR0913
            ...
            # factory_kwargs = {"device": device, "dtype": dtype}
            # super().__init__()
            # self.in_features = in_features
            # self.out_features = out_features
            # self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
            # if bias:
            #     self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
            # else:
            #     self.register_parameter("bias", None)
            # self.reset_parameters()

        # def reset_parameters(self) -> None:
        #     # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        #     # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        #     # https://github.com/pytorch/pytorch/issues/57109
        #     init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        #     if self.bias is not None:
        #         fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        #         bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        #         init.uniform_(self.bias, -bound, bound)

        def forward(self, input: Tensor[*Shape, In]) -> Tensor[*Shape, Out]:
            ...
            # return F.linear(input, self.weight, self.bias)

        def __call__(self, input: Tensor[*Shape, In]) -> Tensor[*Shape, Out]: ...

    # class Linear(nn.Module):
    #     def __call__(self, input: Tensor[*Shape, Embed]) -> Tensor[*Shape, EmbedX3]: ...

    class Dropout(nn.Module):
        p: float
        inplace: bool

        def __init__(self, p: float, inplace: bool = False) -> None: ...
        def __call__(self, input: Tensor[*Shape]) -> Tensor[*Shape]: ...

    class GPTConfig(BaseModel):
        block_size: Token
        # TODO: handle padding vocab_size to neaerest multiple of 64 somewhere
        vocab_size: int
        n_layer: int
        n_head: Head
        # head dimension = multiple of 8
        n_embd: Embed
        n_mlp: int
        dropout: float
        use_flash: bool
        # TODO: separate bias for Linear and LayerNorm
        bias: bool  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

        @computed_field
        @property
        def n_embd_per_head(self) -> EmbedPerHead: ...

        @computed_field
        @property
        def n_embd_x3(self) -> EmbedX3: ...

        # n_embd_per_head: EmbedPerHead
# from torch.nn import Linear
else:
    from torch import Tensor
    from torch.nn import Dropout, Linear
    from torch.nn.functional import softmax

    class GPTConfig(BaseModel):
        block_size: int
        # TODO: handle padding vocab_size to neaerest multiple of 64 somewhere
        vocab_size: int
        n_layer: int
        n_head: int
        # head dimension = multiple of 8
        n_embd: int
        n_mlp: int
        dropout: float
        use_flash: bool
        # TODO: separate bias for Linear and LayerNorm
        bias: bool  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

        @computed_field
        @property
        def n_embd_per_head(self) -> int:
            return self.n_embd // self.n_head

        @model_validator(mode="after")
        def check_embedding_size(self) -> Self:
            if self.n_embd % self.n_head != 0:
                msg = "n_embd must be divisible by n_head"
                raise ValueError(msg)
            return self


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: Tensor[*Shape]) -> Tensor[*Shape]:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

    def __call__(self, input: Tensor[*Shape]) -> Tensor[*Shape]:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    attn_bias: Tensor[Token, Token]

    n_head: Head
    n_embd: Embed
    n_embd_per_head: EmbedPerHead
    n_embd_x3: EmbedX3

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.n_embd_per_head = config.n_embd_per_head
        self.n_embd_x3 = config.n_embd_x3
        self.dropout = config.dropout
        self.use_flash = config.use_flash

        # key, query, value projections for all heads in a batch
        self.c_attn: Linear[Embed, EmbedX3] = Linear(self.n_embd, self.n_embd_x3, bias=config.bias)

        # output projection
        self.c_proj: Linear[Embed, Embed] = Linear(config.n_embd, config.n_embd, bias=config.bias)

        # regularization
        self.attn_dropout = Dropout(config.dropout)
        self.resid_dropout = Dropout(config.dropout)

        if not self.use_flash:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "attn_bias",
                # torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)).view(
                # torch.tril(torch.ones(config.block_size, config.block_size)).view(
                # torch.tril(torch.ones(config.block_size, config.block_size) * float("-inf")).view(
                #     1,
                #     1,
                #     config.block_size,
                #     config.block_size,
                # ),
                torch.tril(torch.ones(config.block_size, config.block_size) * float("-inf")),
                persistent=False,
            )

    def project_qkv(
        self,
        x: Tensor[Batch, Token, Embed],
    ) -> Tensor[Batch, Token, EmbedX3]:
        return self.c_attn(x)

    def split_qkv(
        self,
        x: Tensor[Batch, Token, EmbedX3],
    ) -> tuple[
        Tensor[Batch, Token, Embed],
        Tensor[Batch, Token, Embed],
        Tensor[Batch, Token, Embed],
    ]:
        return x.split(self.n_embd, dim=-1)

    def split_heads(
        self,
        q: Tensor[Batch, Token, Embed],
        k: Tensor[Batch, Token, Embed],
        v: Tensor[Batch, Token, Embed],
    ) -> tuple[
        Tensor[Batch, Head, Token, EmbedPerHead],
        Tensor[Batch, Head, Token, EmbedPerHead],
        Tensor[Batch, Head, Token, EmbedPerHead],
    ]:
        batch_size, seq_len, num_embed = q.size()
        q1: Tensor[Batch, Token, Head, EmbedPerHead] = q.view(batch_size, seq_len, self.n_head, self.n_embd_per_head)  # type: ignore  # noqa: PGH003
        k1: Tensor[Batch, Token, Head, EmbedPerHead] = k.view(batch_size, seq_len, self.n_head, self.n_embd_per_head)  # type: ignore  # noqa: PGH003
        v1: Tensor[Batch, Token, Head, EmbedPerHead] = v.view(batch_size, seq_len, self.n_head, self.n_embd_per_head)  # type: ignore  # noqa: PGH003
        q2 = q1.transpose(1, 2)
        k2 = k1.transpose(1, 2)
        v2 = v1.transpose(1, 2)
        return q2, k2, v2

    def forward(self, x: Tensor[Batch, Token, Embed]) -> Tensor[Batch, Token, Embed]:
        # batch size, sequence length, embedding dimensionality (n_embd)
        B, T, C = x.size()  # noqa: N806

        def combine_heads(
            x: Tensor[Batch, Head, Token, EmbedPerHead],
        ) -> Tensor[Batch, Token, Embed]:
            y = x.transpose(1, 2)
            y = y.contiguous()
            return y.view(B, T, C)
            # return x.transpose(1, 2).contiguous().view(B, T, C)

        # # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        # q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        # v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        qkv = self.project_qkv(x)
        q, k, v = self.split_qkv(qkv)
        q, k, v = self.split_heads(q, k, v)

        # causal self-attention: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = q.matmul(k.transpose(-2, -1))  # q @ k.transpose(-2, -1)
        att *= 1.0 / math.sqrt(C)  # math.sqrt(k.size(-1))

        att += self.attn_bias
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        att = softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att.matmul(v)  # att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # re-assemble all head outputs side by side
        y = combine_heads(y)  # y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        z = self.c_proj(y)
        return self.resid_dropout(z)
