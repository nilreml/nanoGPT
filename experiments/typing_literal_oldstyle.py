from typing import Annotated, Generic, Literal, TypeVar, get_type_hints, reveal_type

AxisA = TypeVar("AxisA")

VeryImportant = "very_important"

SomeStr = Annotated[str, VeryImportant]


class Array(Generic[AxisA]):
    # def __new__(cls, a) -> Self: ...

    def __init__(self, a: AxisA) -> None:
        # return Array(a)
        self.a = a
        self.ta = type(self.a)


def array(a: AxisA) -> AxisA:
    return a


# reveal_type(array)
# print(get_type_hints(array, include_extras=True))
print(SomeStr.__metadata__)

a = Array(Literal[2])

reveal_type(a)
print(get_type_hints(Array, include_extras=True))

print(a.a)
print(a.ta)
