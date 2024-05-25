import json
from functools import partial
from itertools import chain, product
from pathlib import Path
from typing import Annotated

from annotated_types import Ge, Gt, Le, Lt
from jsf import JSF
from pydantic import TypeAdapter

path = Path("test_schemata").resolve()
path.mkdir(parents=True, exist_ok=True)

printf = partial(print, end="")

cmat_both = [
    ([Gt(0), Ge(0)], [None, Le(1), Lt(1), Le(2), Lt(2)]),  # Positive, NonNegative
    ([Lt(0), Le(0)], [None, Ge(-1), Gt(-1), Ge(-2), Gt(-2)]),  # Negative, NonPositive
]
cmat_float = [
    ([Gt(0.1), Ge(0.1)], [None, Le(0.9), Lt(0.9), Le(2.1), Lt(2.1)]),  # Positive
    ([Lt(-0.1), Le(-0.1)], [None, Ge(-0.9), Gt(-0.9), Ge(-2.1), Gt(-2.1)]),  # Negative
]
invalid_int = [
    (Gt(0), Lt(1)),  # 0 < x < 1
    (Lt(0), Gt(-1)),  # -1 < x < 0
]
cons_both = [*chain(*[product(a, b) for a, b in cmat_both])]
cons_float = [*chain(*[product(a, b) for a, b in cmat_float])]

for origin in [int, float]:
    for i, c in enumerate(cons_both + cons_float if origin == float else cons_both):
        # remove None constraints
        constraints = tuple([x for x in c if x is not None])

        typ = Annotated[origin, *constraints]
        title = str(typ).replace("typing.", "")
        printf(f"\n{title:<42}")

        if origin == int and constraints in invalid_int:
            printf("  <- skip: invalid constraints for int")
            continue

        schema = TypeAdapter(typ).json_schema()
        prefix = "pass"

        try:
            JSF(schema).generate()
        except Exception as e:  # noqa: BLE001
            printf(f"  <- fail: {e}")
            prefix = "fail"
            title += f":  {e}"

        # write json schema file
        schema["title"] = title
        (path / f"{prefix}_{origin.__name__}_{i:02}.json").write_text(json.dumps(schema, indent=2))

print()
