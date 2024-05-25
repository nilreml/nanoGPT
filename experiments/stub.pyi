from typing import Generic, Literal, TypeVar, TypeVarTuple, overload

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
    ) -> Tensor[Second, First, *ShapeRest]:
        ...
        # return super().transpose(1, 2)  # type: ignore  # noqa: PGH003

    @overload
    def transpose(self, dim0: int, dim1: int) -> Tensor:
        ...
        # return super().transpose(dim0, dim1)  # type: ignore  # noqa: PGH003

    # @overload
    # @override
    # def transpose(self: "Tensor[First, Second]", dim0: Literal[2], dim1: Literal[2]) -> "Tensor[Second, First]":
    #     return super().transpose(1, 2)  # type: ignore  # noqa: PGH003


@overload
def transpose(
    a: Tensor[First, Second, *ShapeRest],
    dim0: Literal[1],
    dim1: Literal[2],
) -> Tensor[Second, First, *ShapeRest]: ...
@overload
def transpose(a: Tensor, dim0: int, dim1: int) -> Tensor: ...
