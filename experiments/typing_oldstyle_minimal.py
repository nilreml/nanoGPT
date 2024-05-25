from typing import Generic, Self, TypeVar
from typing import Literal as L

AxisA = TypeVar("AxisA")
AxisB = TypeVar("AxisB")
AxisC = TypeVar("AxisC")


class Matrix(Generic[AxisA, AxisB]):
    # def __init__(self, a: AxisA, b: AxisB) -> None: ...

    def __new__(cls, a: AxisA, b: AxisB) -> Self:
        return cls(a, b)


def matmul(
    a: Matrix[AxisA, AxisB],
    b: Matrix[AxisB, AxisC],
) -> Matrix[AxisA, AxisC]:
    # Second axis of a must be the same as first axis of b
    ...


def elmul(
    a: Matrix[AxisA, AxisB],
    b: Matrix[AxisA, AxisB],
) -> Matrix[AxisA, AxisB]:
    # Both axes of a and b must be the same
    ...


RowsA = L["32"]
ColsA = L["65"]
RowsB = L["64"]
ColsB = L["128"]

aa = Matrix(RowsA, ColsA)

a: Matrix[RowsA, ColsA] = Matrix()
b: Matrix[RowsB, ColsB] = Matrix()

c = matmul(a, b)

x: Matrix[ColsA, ColsB] = Matrix()
y: Matrix[RowsB, ColsB] = Matrix()
z = elmul(x, y)
