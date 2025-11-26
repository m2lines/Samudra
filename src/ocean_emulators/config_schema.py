import argparse
import json
import re
import sys
from pathlib import Path
from types import UnionType

import pydantic
import yaml

from ocean_emulators.config import EvalConfig, TrainConfig
from ocean_emulators.viz import VizConfig


def get_pydantic_models(
    model: type[pydantic.BaseModel],
    seen: dict[str, type[pydantic.BaseModel]] | None = None,
) -> dict[str, type[pydantic.BaseModel]]:
    """Recursively find all Pydantic models in a model's fields.

    Args:
        model: The Pydantic model to start from
        seen: Dictionary of already seen models (keyed by model name)

    Returns:
        Dictionary of model names to model classes
    """
    if seen is None:
        seen = {}

    model_name = model.__name__
    if model_name in seen:
        return seen

    seen[model_name] = model

    for field in model.model_fields.values():
        field_type = field.annotation
        if isinstance(field_type, type) and issubclass(field_type, pydantic.BaseModel):
            get_pydantic_models(field_type, seen)
        elif isinstance(field_type, UnionType):
            for type_ in field_type.__args__:
                if isinstance(type_, type) and issubclass(type_, pydantic.BaseModel):
                    get_pydantic_models(type_, seen)

    return seen


def generate_schemas(
    output_dir: Path,
    models: dict[str, type[pydantic.BaseModel]],
) -> None:
    """Generate JSON schemas for all Pydantic models and save them to output_dir.

    Args:
        output_dir: Directory to save the generated schemas
        models: Dictionary of model names to model classes
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        schema = model.model_json_schema()
        output_path = output_dir / f"{model_name}.json"

        # Check if file exists and content is different
        if output_path.exists():
            with open(output_path) as f:
                existing_schema = json.load(f)
                if existing_schema == schema:
                    continue

        # Write only if content differs or file doesn't exist
        with open(output_path, "w") as f:
            json.dump(schema, f, indent=2)
            print(f"🆕 Updated schema for {model_name} at {output_path}")


def validate_config_files(
    config_dir: Path,
    models: dict[str, type[pydantic.BaseModel]],
) -> bool:
    """Validate YAML configuration files against their Pydantic models.

    Args:
        config_dir: Directory containing YAML configuration files to validate
        models: Dictionary of model names to model classes
    """
    valid = True

    # Find all YAML files in the config directory
    yaml_files = list(config_dir.rglob("*.yaml"))

    for yaml_file in yaml_files:
        # Read the YAML file
        with open(yaml_file) as f:
            yaml_content = f.read()

        # Extract the schema path from the YAML file's first line
        schema_match = re.search(
            r"\s*#\s+yaml-language-server:\s+\$schema=(.*\.json)", yaml_content
        )
        if not schema_match:
            print(f"⚠️ No schema specified in {yaml_file}")
            continue

        schema_name = Path(schema_match.group(1)).stem
        if schema_name not in models:
            print(f"✗ Model {schema_name} not found for {yaml_file}")
            valid = False
            continue

        # Load and validate the YAML content
        try:
            with open(yaml_file) as f:
                config = yaml.safe_load(f)
                # Validate using Pydantic model
                models[schema_name].model_validate(config)
                print(f"✅ {yaml_file} is a valid {schema_name}")
        except Exception as e:
            print(f"❌ {yaml_file} is an invalid {schema_name}: {str(e)}")
            valid = False

    return valid


def main():
    parser = argparse.ArgumentParser(
        description="Generate JSON schemas and validate config files"
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
        help="Directory of config files to validate",
    )

    args = parser.parse_args()

    # Get all available models
    models = get_pydantic_models(TrainConfig)
    models.update(get_pydantic_models(EvalConfig))
    models.update(get_pydantic_models(VizConfig))

    generate_schemas(args.output_dir, models)

    if not validate_config_files(args.validate_dir, models):
        sys.exit(1)


if __name__ == "__main__":
    main()
