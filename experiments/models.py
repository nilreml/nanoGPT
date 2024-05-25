"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

from typing import Generic, Literal, TypeAlias, TypeVar, TypeVarTuple, overload

from torch import nn  # Tensor

Head: TypeAlias = Literal[6]
Batch: TypeAlias = Literal[16]
Token: TypeAlias = Literal[64]
Embed: TypeAlias = Literal[256]
EmbedPerHead: TypeAlias = Literal[32]
EmbedX3: TypeAlias = Literal[768]

Shape = TypeVarTuple("Shape")
ShapeRest = TypeVarTuple("ShapeRest")

First = TypeVar("First")
Second = TypeVar("Second")

class Tensor(Generic[*Shape]):
    # def __init__(self, shape: tuple[*Shape]) -> None:
    #     self._shape: tuple[*Shape] = shape

    # def get_shape(self) -> tuple[*Shape]:
    #     return self._shape

    @overload
    def transpose(
        self: Tensor[First, Second, *ShapeRest],
        dim0: Literal[1],
        dim1: Literal[2],
    ) -> Tensor[Second, First, *ShapeRest]: ...
    @overload
    def transpose(self, dim0: int, dim1: int) -> Tensor: ...

    # @overload
    # @override
    # def transpose(self: "Tensor[First, Second]", dim0: Literal[2], dim1: Literal[2]) -> "Tensor[Second, First]":
    #     return super().transpose(1, 2)  # type: ignore  # noqa: PGH003

class Linear(nn.Module):
    def __call__(self, input: Tensor[*Shape, Embed]) -> Tensor[*Shape, EmbedX3]: ...

@overload
def transpose(
    a: Tensor[First, Second, *ShapeRest],
    dim0: Literal[1],
    dim1: Literal[2],
) -> Tensor[Second, First, *ShapeRest]: ...
@overload
def transpose(a: Tensor, dim0: int, dim1: int) -> Tensor: ...
