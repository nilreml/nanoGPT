"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

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
from pydantic import BaseModel, computed_field
from torch import nn  # Tensor

ellipsis = EllipsisType

# Place-holder values for static type-checking
# NOTE: values should be unique
# TODO: use type arithmetic once implemented by PyRight
Batch = Literal[20]
Head = Literal[6]
Token = Literal[64]
DimModel = Literal[192]
DimHead = Literal[32]  # Dim/Head
DimX3 = Literal[576]  # Dim*3
DimMLP = Literal[480]  # Dim*2.5

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
            self: "Tensor[*ShapeRest, DimX3]",
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

        @overload
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
        def __add__(
            self,
            other,
        ) -> "Tensor": ...

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

    #     # @overload
    #     # @override
    #     # def transpose(self: "Tensor[First, Second]", dim0: Literal[2], dim1: Literal[2]) -> "Tensor[Second, First]":
    #     #     return super().transpose(1, 2)  # type: ignore  # noqa: PGH003

    def softmax(
        x: Tensor[*Shape],  # noqa: ARG001
        dim: int,  # noqa: ARG001
    ) -> Tensor[*Shape]: ...

    class Linear(Generic[In, Out]):  # nn.Module
        in_features: In
        out_features: Out
        weight: Tensor

        def __new__(cls, in_features: In, out_features: Out, bias: bool = True, device=None, dtype=None) -> "Linear[In, Out]": ...  # noqa: PLR0913, FBT001, FBT002
        def __init__(self, in_features: In, out_features: Out, bias: bool = True, device=None, dtype=None) -> None:  # noqa: PLR0913, FBT001, FBT002
            ...

        def forward(self, input: Tensor[*Shape, In]) -> Tensor[*Shape, Out]: ...
        def __call__(self, input: Tensor[*Shape, In]) -> Tensor[*Shape, Out]: ...

    # class Linear(nn.Module):
    #     def __call__(self, input: Tensor[*Shape, Embed]) -> Tensor[*Shape, EmbedX3]: ...

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

        # def __init__(self, ndim, bias) -> None: ...
        def forward(self, input: Tensor[*Shape]) -> Tensor[*Shape]: ...
        def __call__(self, input: Tensor[*Shape]) -> Tensor[*Shape]: ...
else:
    # from torch import Tensor
    from torch.nn import GELU, Dropout, Linear

    class Tensor(Generic[*Shape], torch.Tensor): ...


class GPTConfig(BaseModel):
    # TODO: handle padding vocab_size to neaerest multiple of 64 somewhere
    vocab_size: int
    num_layers: int

    # Literal types for static type-checking
    if TYPE_CHECKING:
        seq_len: Token
        num_heads: Head
        dim_model: DimModel
        # dim_mlp: DimMLP
    else:
        seq_len: int
        num_heads: int
        dim_model: int
        # dim_mlp: int

    dropout: float
    # TODO: separate bias for LayerNorm and Bias, input and output projections, attention and feedforward
    bias: bool  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

    @computed_field
    @property
    def dim_x3(self) -> DimX3:
        return self.dim_model * 3

    @computed_field
    @property
    def dim_mlp(self) -> DimMLP:
        return self.dim_model * 3  # type: ignore  # noqa: PGH003

    @computed_field
    @property
    def dim_head(self) -> DimHead:
        if self.dim_model % self.num_heads != 0:
            msg = "dim_model must be divisible by num_heads"
            raise ValueError(msg)
        return self.dim_model // self.num_heads

    # @model_validator(mode="after")
    # def check_embedding_size(self) -> Self:
    #     if self.dim_model % self.num_heads != 0:
    #         msg = "dim_model must be divisible by num_heads"
    #         raise ValueError(msg)
    #     return self


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
