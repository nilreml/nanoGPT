import subprocess
import re
from pydantic import BaseModel
import pytest
import math
import os


class Result(BaseModel):
    iter: int
    loss: float
    time: float


@pytest.fixture
def results(request: pytest.FixtureRequest) -> list[Result]:
    # Run python train.py tests/config/config_1.py and capture its output
    cmd = f"python3 train.py {request.param}"

    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    process = subprocess.run(
        cmd,
        check=True,
        shell=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    output = process.stdout
    print(output)

    results: list[Result] = []
    for line in output.split("\n"):
        match = re.search(
            r"iter (\d+): loss (\d+\.\d+), time (\d+\.\d+)ms, mfu (-?\d+\.\d+)%", line
        )
        if match:
            r = Result(
                iter=int(match.group(1)),
                loss=float(match.group(2)),
                time=float(match.group(3)),
            )
            if r.iter >= 0:
                results.append(r)
    return results


# @pytest.mark.flaky
@pytest.mark.parametrize(
    "results, expected",
    [
        (
            "tests/config/config_1.py",
            [
                Result(iter=0, loss=4.236855, time=250),
                Result(iter=20, loss=3.037239, time=8.5),
            ],
        ),
        (
            "tests/config/config_2.py",
            [
                Result(iter=0, loss=4.210068, time=215),
                Result(iter=20, loss=3.100468, time=8.5),
            ],
        ),
        (
            "tests/config/config_3.py",
            [
                Result(iter=0, loss=4.214890, time=5030),
                Result(iter=20, loss=2.93995, time=4.5),
            ],
        ),
    ],
    indirect=["results"],
)
def test_train(results, expected):
    print(results)
    print(expected)
    assert len(results) == len(expected)

    for r, e in zip(results, expected):
        assert r.iter == e.iter
        assert math.isclose(r.loss, e.loss, rel_tol=2e-5)
        assert math.isclose(r.time, e.time, rel_tol=0.2)
