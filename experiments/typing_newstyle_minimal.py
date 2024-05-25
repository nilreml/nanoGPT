from typing import Generic, Literal, TypeVar

AxisA = TypeVar("AxisA")
AxisB = TypeVar("AxisB")
AxisC = TypeVar("AxisC")


class Matrix(Generic[AxisA, AxisB]): ...


def matmul(
    a: Matrix[AxisA, AxisB],
    b: Matrix[AxisB, AxisC],
) -> Matrix[AxisA, AxisC]:
    """Second axis of a must be the same as first axis of b"""


def elmul(
    a: Matrix[AxisA, AxisB],
    b: Matrix[AxisA, AxisB],
) -> Matrix[AxisA, AxisB]:
    """Both axes of a and b must be the same"""


RowsA = Literal["32"]
ColsA = Literal["65"]
RowsB = Literal["64"]
ColsB = Literal["128"]

a: Matrix[RowsA, ColsA] = Matrix()
b: Matrix[RowsB, ColsB] = Matrix()

c = matmul(a, b)

x: Matrix[ColsA, ColsB] = Matrix()
y: Matrix[RowsB, ColsB] = Matrix()
z = elmul(x, y)
