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


@pytest.fixture()
def results(request: pytest.FixtureRequest) -> list[Result]:
    # Run python train.py tests/config/config_1.py and capture its output

    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
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
            r"iter (\d+): loss (\d+\.\d+), time (\d+\.\d+)ms, mfu (-?\d+\.\d+)%",
            line,
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


@pytest.fixture(scope="session")
def _log() -> None:
    with Path("tests/results.txt").open("a") as f:
        f.write("\n")


# @pytest.mark.flaky()
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
                Result(iter=0, loss=4.214890, time=5900),
                Result(iter=20, loss=2.93995, time=4.6),
            ],
        ),
    ],
    indirect=["results"],
)
# WTF does it fail:
# 2024-04-26 22:35:55.254473 [Result(iter=0, loss=4.236855, time=290.48), Result(iter=80, loss=2.549376, time=8.55)]
# 2024-04-26 22:35:57.801171 [Result(iter=0, loss=4.210068, time=291.75), Result(iter=80, loss=2.570952, time=8.9)]
# 2024-04-26 22:36:00.411026 [Result(iter=0, loss=4.176258, time=297.74), Result(iter=80, loss=2.575707, time=9.62)]
# 2024-04-26 22:36:08.269912 [Result(iter=0, loss=4.214901, time=5696.1), Result(iter=80, loss=2.526675, time=4.55)]


# post optimization, pre statistic filtering:
# 2024-04-26 20:48:41.773429 [Result(iter=0, loss=4.236855, time=290.32), Result(iter=80, loss=2.549376, time=8.61)]
# 2024-04-26 20:48:44.344302 [Result(iter=0, loss=4.210068, time=294.82), Result(iter=80, loss=2.570952, time=8.96)]
# 2024-04-26 20:48:46.960198 [Result(iter=0, loss=4.176258, time=299.05), Result(iter=80, loss=2.575707, time=9.64)]
# 2024-04-26 20:48:54.876659 [Result(iter=0, loss=4.2149 , time=5739.05), Result(iter=80, loss=2.526495, time=4.52)]

# @pytest.mark.usefixtures("_log")
# @pytest.mark.parametrize(
#     ("results", "expected"),
#     [
#         (
#             "tests/config/config_1.py",
#             [
#                 Result(iter=0, loss=4.236855, time=265),
#                 Result(iter=80, loss=3.037239, time=22),
#             ],
#         ),
#         (
#             "tests/config/config_2.py",
#             [
#                 Result(iter=0, loss=4.210068, time=265),
#                 Result(iter=80, loss=3.100468, time=22),
#             ],
#         ),
#         (
#             "tests/config/config_3.py",
#             [
#                 Result(iter=0, loss=4.176258, time=265),
#                 Result(iter=80, loss=3.006874, time=22),
#             ],
#         ),
#         (
#             "tests/config/config_4.py",
#             [
#                 Result(iter=0, loss=4.214890, time=5200),
#                 Result(iter=80, loss=2.93995, time=11),
#             ],
#         ),
#     ],
#     indirect=["results"],
# )

def test_train(results: list[Result], expected: list[Result]) -> None:
    assert len(results) == len(expected)

    msg = f"{datetime.now()} {results}\n"  # noqa: DTZ005
    with Path("tests/results.txt").open("a") as f:
        f.write(msg)

    for r, e in zip(results, expected, strict=True):
        assert r.iter == e.iter
        assert math.isclose(r.loss, e.loss, rel_tol=3e-5)
        assert r.time <= e.time

    # pytest.fail(str(results))
