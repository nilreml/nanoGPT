import math
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import BaseModel


class Result(BaseModel):
    iter: int
    loss: float
    time: float
    mfu: float = float("-31337")
    max_alloc: float = 0.0


@pytest.fixture()
def results(request: pytest.FixtureRequest) -> list[Result]:
    # Run python train.py tests/config/config_1.py and capture its output

    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
    # os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    print(os.environ)
    process = subprocess.run(
        args=f"nice -n -20 python3 train.py {request.param}",
        check=False,
        # Use active virtualenv in current shell
        shell=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    if process.returncode != 0:
        print(process.stdout)
        print(process.stderr)
        raise subprocess.CalledProcessError(
            returncode=process.returncode,
            cmd=process.args,
            output=process.stdout,
            stderr=process.stderr,
        )
    output = process.stdout

    results: list[Result] = []
    for line in output.split("\n"):
        match = re.search(
            r"iter (\d+): loss (\d+\.\d+), time (\d+\.\d+)ms, mfu (-?\d+\.\d+)%, max alloc (\d+\.\d+)GB",
            line,
        )
        if match:
            r = Result(
                iter=int(match.group(1)),
                loss=float(match.group(2)),
                time=float(match.group(3)),
                mfu=float(match.group(4)),
                max_alloc=float(match.group(5)),
            )
            if r.iter >= 0:
                results.append(r)
    return results


@pytest.fixture(scope="session")
def _log() -> None:
    with Path("tests/results.txt").open("a") as f:
        f.write("\n")


@pytest.mark.usefixtures("_log")
@pytest.mark.parametrize(
    ("results", "expected"),
    [
        (
            "tests/config/config_1.py",
            [
                Result(iter=0, loss=4.236855, time=310),
                Result(iter=20, loss=3.037239, time=8.65),
            ],
        ),
        (
            "tests/config/config_2.py",
            [
                Result(iter=0, loss=4.210068, time=314),
                Result(iter=20, loss=3.100468, time=9),
            ],
        ),
        (
            "tests/config/config_3.py",
            [
                Result(iter=0, loss=4.176258, time=319),
                Result(iter=20, loss=3.006874, time=9.7),
            ],
        ),
        (
            "tests/config/config_4.py",
            [
                Result(iter=0, loss=4.215, time=5900),
                Result(iter=20, loss=2.93995, time=3.6),
                # Result(iter=0, loss=4.214890, time=5900),  # pytorch 2.2
                # Result(iter=20, loss=2.93995, time=4.6),   # pytorch 2.2
            ],
        ),
    ],
    indirect=["results"],
)
def test_train(results: list[Result], expected: list[Result]) -> None:
    assert len(results) == len(expected)

    msg = f"{datetime.now()} {results}\n"  # noqa: DTZ005
    with Path("tests/results.txt").open("a") as f:
        f.write(msg)

    for r, e in zip(results, expected, strict=True):
        assert r.iter == e.iter
        assert math.isclose(r.loss, e.loss, rel_tol=5e-5)
        assert r.time <= e.time

    # pytest.fail(str(results))
