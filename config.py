import json
from enum import StrEnum, auto
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    Self,
)

import pydantic
import yaml
from annotated_types import Ge, Gt, Le, Lt
from pydantic import (
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    computed_field,
    model_validator,
)

ZeroExclToOneExclFloat = Annotated[float, Gt(0), Lt(1)]
"""A float that must be greater than zero and less than one."""

ZeroExclToOneInclFloat = Annotated[float, Gt(0), Le(1)]
"""A float that must be greater than zero and less than or equal to one."""

ZeroInclToOneExclFloat = Annotated[float, Ge(0), Lt(1)]
"""A float that must be greater than or equal to zero and less than one."""

ZeroInclToOneInclFloat = Annotated[float, Ge(0), Le(1)]
"""A float that must be greater than or equal to zero and less than or equal to one."""


class Model(pydantic.BaseModel):
    """A base class for creating Pydantic models."""

    model_config = ConfigDict(
        strict=True,
        validate_default=True,
        revalidate_instances="always",
        ser_json_inf_nan="constants",
        # use_enum_values=True,
    )

    @classmethod
    def model_validate_yaml(
        cls,
        yaml_data: str,
        *,
        strict: bool | None = None,
        context: dict[str, Any] | None = None,
    ) -> Self:
        """Usage docs: https://docs.pydantic.dev/2.7/concepts/json/#json-parsing

        Validate the given YAML data against the Pydantic model.

        Args:
            yaml_data: The YAML data to validate.
            strict: Whether to enforce types strictly.
            context: Extra variables to pass to the validator.

        Returns:
            The validated Pydantic model.

        Raises:
            ValueError: If `yaml_data` is not a YAML string.
        """
        yaml_dict = yaml.safe_load(yaml_data)
        if not isinstance(yaml_dict, dict):
            msg = f"Invalid YAML string '{yaml_data}'"
            raise ValueError(msg)  # noqa: TRY004
        return cls.model_validate_json(
            json.dumps(yaml_dict),
            strict=strict,
            context=context,
        )


class DType(StrEnum):
    float16 = auto()
    float32 = auto()
    bfloat16 = auto()


# Config domains:
# - model
# - training
#   - dataset
#   - optimizer
#   - initialization

# Place-holder values for static type-checking
# NOTE: values should be unique
# TODO: use type arithmetic once implemented by PyRight
Batch = Literal[20]
Head = Literal[6]
Token = Literal[64]
DimModel = Literal[192]
DimHead = Literal[32]  # DimModel/Head
DimModelX3 = Literal[576]  # DimModel*3
DimMLP = Literal[480]  # DimModel*2.5


class GPTConfig(Model):
    # TODO: handle padding vocab_size to neaerest multiple of 64 somewhere

    # Literal types for static type-checking
    if TYPE_CHECKING:
        seq_len: Token
        dim_model: DimModel
        num_heads: Head
        # dim_mlp: DimMLP
    else:
        seq_len: int
        dim_model: int
        num_heads: int
        # dim_mlp: int

    num_layers: int
    vocab_size: int

    # TODO: separate bias for LayerNorm and Bias, input and output projections, attention and feedforward
    bias: bool  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

    dropout: ZeroInclToOneExclFloat = Field(description="Dropout probability, 0.0 to disable")

    @computed_field
    @property
    def use_dropout(self) -> bool:
        return self.dropout > 0.0

    @computed_field
    @property
    def dim_model_x3(self) -> DimModelX3:
        return self.dim_model * 3

    # TODO: allow specifying dim_mlp
    @computed_field
    @property
    def dim_mlp(self) -> DimMLP:
        return self.dim_model * 3  # type: ignore  # noqa: PGH003

    @computed_field
    @property
    def dim_head(self) -> DimHead:
        if self.dim_model % self.num_heads != 0:
            msg = "dim_model must be divisible by num_heads"
            raise ValueError(msg)
        return self.dim_model // self.num_heads

    @model_validator(mode="after")
    def check_embedding_size(self) -> Self:
        if self.dim_model % self.num_heads != 0:
            msg = "dim_model must be divisible by num_heads"
            raise ValueError(msg)
        return self


class AdamWConfig(Model):
    name: Literal["adam_w"]
    beta1: ZeroExclToOneExclFloat
    beta2: ZeroExclToOneExclFloat


class LearningRateConfig(Model):
    max: PositiveFloat
    min: PositiveFloat
    warmup_schedule: Literal["linear"]
    warmup_iters: NonNegativeInt
    decay_schedule: Literal["none", "cosine_decay"]


class OptimizationConfig(Model):
    max_iters: PositiveInt = Field(description="Maximum number of training iterations")
    grad_clip: PositiveFloat = Field(description="Gradient clipping value, Inf to disable")
    weight_decay: NonNegativeFloat = Field(description="Weight decay value, 0.0 to disable")
    learning_rate: LearningRateConfig
    optimizer: AdamWConfig

    @computed_field
    @property
    def use_grad_clip(self) -> bool:
        return self.grad_clip < float("inf")

    @computed_field
    @property
    def use_weight_decay(self) -> bool:
        return self.weight_decay > 0.0


class TrainConfig(Model):
    dataset: str
    batch_size: PositiveInt
    dtype: DType
    optimization: OptimizationConfig

    compile: bool
    device_type: Literal["cpu", "cuda"]  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    seed_offset: NonNegativeInt

    out_dir: str = "out"
    init_from: str = "scratch"  # 'scratch' or 'resume' or 'gpt2*'

    # TODO: benchmark these:
    # torch.utils.deterministic.fill_uninitialized_memory = False
    # torch.use_deterministic_algorithms(True)


class RootConfig(Model):
    model: GPTConfig
    train: TrainConfig

    @computed_field
    @property
    def tokens_per_iter(self) -> NonNegativeInt:
        return self.train.batch_size * self.model.seq_len


def main() -> None:
    schema = RootConfig.model_json_schema(mode="validation")  # mode="serialization"

    schema_path = Path("schema").resolve()

    json_path = schema_path / "config.schema.json"
    json_path.write_text(json.dumps(schema, indent=2))

    yaml_path = schema_path / "config.schema.yaml"
    yaml_path.write_text(yaml.dump(schema))


if __name__ == "__main__":
    main()
