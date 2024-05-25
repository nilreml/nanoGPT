from typing import TypeAlias, TypeVar
from typing import NewType

import torch
from torch import Tensor

Rows = TypeVar("Rows", bound=int, covariant=False, contravariant=False, infer_variance=False)
Cols = TypeVar("Cols", bound=int, covariant=False, contravariant=False, infer_variance=False)

RowsA = TypeVar("RowsA", bound=int, covariant=False, contravariant=False, infer_variance=False)
RowsB = TypeVar("RowsB", bound=int, covariant=False, contravariant=False, infer_variance=False)
ColsA = TypeVar("ColsA", bound=int, covariant=False, contravariant=False, infer_variance=False)
ColsB = TypeVar("ColsB", bound=int, covariant=False, contravariant=False, infer_variance=False)

#UserId = NewType('UserId', int)

Matrix = TypeVar("Matrix", bound=list[list[float]], covariant=False, contravariant=False, infer_variance=False)

MatrixAlias: TypeAlias = Tensor




def ones(rows: int, cols: int) -> MatrixAlias:
    return torch.ones(rows, cols)

def matmul(a: Matrix[RowsA, RowsB]) ->
