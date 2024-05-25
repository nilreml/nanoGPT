from typing import Literal, reveal_type


class Array[AxisA]:
    # def __new__(cls, a) -> Self:
    #     ...
    # return Array(a)

    def __init__(self, a: AxisA) -> None:
        self.a = a
        self.ta = type(self.a)


a = Array(Literal[2])

reveal_type(a)

print(a.a)
print(a.ta)
