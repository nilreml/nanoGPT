from typing import Literal

from torch import Tensor

# type Batch[int] = int
# type Token[int] = int


type Matrix[Rows: int, Cols: int] = Tensor


def ones[Rows: int, Cols: int](
    rows: Rows,
    cols: Cols,
) -> Matrix[Rows, Cols]:
    ...
    # return torch.ones(rows, cols)


def matmul[R1: int, C1: int, R2: int, C2: int](
    a: Matrix[R1, C1],
    b: Matrix[R2, C2],
) -> Matrix[R1, C2]:
    ...
    # return torch.matmul(a, b)


rowsa = Literal[2]
colsa = Literal[3]

rowsb = Literal[3]
colsb = Literal[4]

a = ones(rowsa, rowsb)
b = ones(3, 4)

# a = ones(2, 3)
# b = ones(3, 4)

c = matmul(a, b)
