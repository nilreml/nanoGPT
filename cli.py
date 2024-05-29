#!/usr/bin/env python3

from pathlib import Path

import typer

from config import RootConfig
from train import train


def cli(config_path: Path) -> None:
    if not config_path.exists() and not config_path.is_absolute():
        config_path = Path.cwd() / "tests" / "config" / config_path

    config = RootConfig.model_validate_yaml(config_path.read_text())
    train(config)


if __name__ == "__main__":
    typer.run(cli)
