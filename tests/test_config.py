from decimal import Decimal
from typing import Annotated

import pydantic
import pytest
from pydantic import AllowInfNan

from config import (
    GPTConfig,
    Model,
)


def test_model_validate_yaml_invalid() -> None:
    yaml_data = "sup"
    with pytest.raises(ValueError, match="Invalid YAML string 'sup'"):
        GPTConfig.model_validate_yaml(yaml_data)


def test_model_validate_yaml_valid() -> None:
    class Person(Model):
        name: str
        age: int

    yaml_data = """
name: Bors
age: 42
    """
    expected = Person(name="Bors", age=42)
    assert expected == Person.model_validate_yaml(yaml_data)


def test_model_infinity_float() -> None:
    class Foo(pydantic.BaseModel):
        number: Annotated[float, AllowInfNan(True)]

    json_data = """{ "number": "Infinity" }"""
    expected = Foo(number=float("inf"))
    assert expected == Foo.model_validate_json(json_data)


def test_model_infinity_decimal() -> None:
    class Foo(Model):
        decimal: Annotated[Decimal, AllowInfNan(True)]

    json_data = """{ "decimal": "Infinity" }"""
    expected = Foo(decimal=Decimal("Infinity"))
    assert expected == Foo.model_validate_json(json_data)
