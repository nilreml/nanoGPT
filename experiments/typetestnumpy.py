from typing import Generic, Literal, TypeVar

# Width = NewType("Width", int)
# Height = NewType("Height", int)

DType = TypeVar("DType")
AxisA = TypeVar("AxisA")
AxisB = TypeVar("AxisB")
AxisC = TypeVar("AxisC")


class Array(Generic[DType, AxisA, AxisB]): ...


# def matmul(
#     a: Array[DType, AxisA1, AxisA2],
#     b: Array[DType, AxisB1, AxisC],
# ) -> Array[DType, AxisA1, AxisC]: ...


def matmul(
    a: Array[DType, AxisA, AxisB],
    b: Array[DType, AxisB, AxisC],
) -> Array[DType, AxisA, AxisC]: ...


def elmul(
    a: Array[DType, AxisA, AxisB],
    b: Array[DType, AxisA, AxisB],
) -> Array[DType, AxisA, AxisB]: ...


# class Array(Generic[DType, AxisA1, AxisA2]):

#     def matmul(self, other: Array[Generic[DType, AxisB1, AxisC]]) -> Array[Generic[DType, AxisA1, AxisC]]:
#         ...

Float16 = Literal["float16"]

WidthA = Literal["32"]
HeightA = Literal["65"]
WidthB = Literal["64"]
HeightB = Literal["128"]

a: Array[Float16, WidthA, HeightA] = Array()
b: Array[Float16, WidthB, HeightB] = Array()

c = matmul(a, b)

x: Array[Float16, HeightA, HeightB] = Array()
y: Array[Float16, WidthB, HeightB] = Array()
z = elmul(x, y)
