import argparse
import json
from pathlib import Path
from typing import Optional, Set, Type

import pydantic

from ocean_emulators.config import TrainConfig


def get_pydantic_models(
    model: Type[pydantic.BaseModel],
    seen: Optional[Set[Type[pydantic.BaseModel]]] = None,
) -> Set[Type[pydantic.BaseModel]]:
    """Recursively find all Pydantic models in a model's fields."""
    if seen is None:
        seen = set()

    if model in seen:
        return seen

    seen.add(model)

    for field in model.model_fields.values():
        field_type = field.annotation
        if isinstance(field_type, type) and issubclass(field_type, pydantic.BaseModel):
            get_pydantic_models(field_type, seen)

    return seen


def generate_schemas(output_dir: Path) -> None:
    """Generate JSON schemas for all Pydantic models and save them to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    models = get_pydantic_models(TrainConfig)
    for model in models:
        schema = model.model_json_schema()
        output_path = output_dir / f"{model.__name__}.json"

        with open(output_path, "w") as f:
            json.dump(schema, f, indent=2)

        print(f"Generated schema for {model.__name__} at {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate JSON schemas from Pydantic models"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("configs/schemas"),
        help="Directory to save the generated schemas",
    )

    args = parser.parse_args()
    generate_schemas(args.output_dir)


if __name__ == "__main__":
    main()
