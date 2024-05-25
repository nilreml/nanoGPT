import pytest

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
