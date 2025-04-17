import argparse
import json
import re
from pathlib import Path
from typing import Optional, Set, Type

import pydantic
import yaml
from jsonschema import validate

from ocean_emulators.config import EvalConfig, TrainConfig


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
    models.update(get_pydantic_models(EvalConfig))

    for model in models:
        schema = model.model_json_schema()
        output_path = output_dir / f"{model.__name__}.json"

        with open(output_path, "w") as f:
            json.dump(schema, f, indent=2)

        print(f"Generated schema for {model.__name__} at {output_path}")


def validate_schemas(config_dir: Path) -> None:
    """Validate YAML configuration files against their JSON schemas.

    Args:
        config_dir: Directory containing YAML configuration files to validate
    """
    # Find all YAML files in the config directory
    yaml_files = list(config_dir.rglob("*.yaml"))

    for yaml_file in yaml_files:
        # Read the YAML file
        with open(yaml_file, "r") as f:
            yaml_content = f.read()

        # Extract the schema path from the YAML file's first line
        schema_match = re.search(
            r"# yaml-language-server: \$schema=(.*\.json)", yaml_content
        )
        if not schema_match:
            print(f"Warning: No schema specified in {yaml_file}")
            continue

        schema_path = yaml_file.parent / schema_match.group(1)
        if not schema_path.exists():
            print(f"Warning: Schema file {schema_path} not found for {yaml_file}")
            continue

        # Load the schema
        with open(schema_path, "r") as f:
            schema = json.load(f)

        # Load and validate the YAML content
        try:
            with open(yaml_file, "r") as f:
                # we re-load so the path is known
                config = yaml.safe_load(f)
                validate(instance=config, schema=schema)
            print(f"✓ {yaml_file} is valid against {schema_path}")
        except Exception as e:
            print(f"✗ {yaml_file} is invalid against {schema_path}: {str(e)}")
            raise e


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
    parser.add_argument(
        "--validate-dir",
        type=Path,
        default=Path("configs"),
        help="Directory validate the schemas against",
    )

    args = parser.parse_args()
    generate_schemas(args.output_dir)

    if args.validate_dir:
        validate_schemas(args.validate_dir)


if __name__ == "__main__":
    main()
