import math
from datetime import datetime
from pathlib import Path

import pytest

from config import RootConfig
from train import Result, train


@pytest.fixture()
def results(request: pytest.FixtureRequest) -> list[Result]:
    # os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
    # os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

    config_path = Path(request.param)
    config = RootConfig.model_validate_yaml(config_path.read_text())

    return train(config=config, do_save=True)


@pytest.fixture(scope="session")
def _log() -> None:
    with Path("tests/results.txt").open("a") as f:
        f.write("\n")


@pytest.mark.usefixtures("_log")
@pytest.mark.parametrize(
    ("results", "expected"),
    [
        (
            "tests/config/config_1.yaml",
            [
                Result(iter=0, loss=4.236855, time=310),
                Result(iter=20, loss=3.037239, time=8.65),
            ],
        ),
        (
            "tests/config/config_2.yaml",
            [
                Result(iter=0, loss=4.210068, time=314),
                Result(iter=20, loss=3.100468, time=9),
            ],
        ),
        (
            "tests/config/config_3.yaml",
            [
                Result(iter=0, loss=4.176258, time=319),
                Result(iter=20, loss=3.006874, time=9.7),
            ],
        ),
        (
            "tests/config/config_4.yaml",
            [
                Result(iter=0, loss=4.215, time=5900),
                Result(iter=20, loss=2.93995, time=3.8),
                # Result(iter=0, loss=4.214890, time=5900),  # pytorch 2.2
                # Result(iter=20, loss=2.93995, time=4.6),   # pytorch 2.2
            ],
        ),
    ],
    indirect=["results"],
)
def test_train(results: list[Result], expected: list[Result]) -> None:
    assert len(results) == len(expected)

    msg = f"{datetime.now()} {[r.round_trip() for r in results]}\n"  # noqa: DTZ005
    with Path("tests/results.txt").open("a") as f:
        f.write(msg)

    for r, e in zip(results, expected, strict=True):
        assert r.iter == e.iter
        assert math.isclose(r.loss, e.loss, rel_tol=5e-5)
        assert r.time <= e.time

    # pytest.fail(str(results))
    # pytest.fail()
