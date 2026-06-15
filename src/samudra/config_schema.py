# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import re
import sys
from pathlib import Path
from typing import get_args, get_origin

import pydantic
import yaml

from samudra.config import EvalConfig, TrainConfig
from samudra.viz import VizConfig


def get_pydantic_models(
    model: type[pydantic.BaseModel],
    models: dict[str, type[pydantic.BaseModel]] | None = None,
    seen: set[int] | None = None,
) -> dict[str, type[pydantic.BaseModel]]:
    """Recursively find all Pydantic models in a model's fields.

    Args:
        model: The Pydantic model to start from
        models: Collected models keyed by model name
        seen: Visited annotation objects keyed by identity

    Returns:
        Dictionary of model names to model classes
    """
    if models is None:
        models = {}
    if seen is None:
        seen = set()

    _collect_pydantic_models(model, models, seen)
    return models


def _collect_pydantic_models(
    annotation: object,
    models: dict[str, type[pydantic.BaseModel]],
    seen: set[int],
) -> None:
    annotation_id = id(annotation)
    if annotation_id in seen:
        return
    seen.add(annotation_id)

    if isinstance(annotation, type) and issubclass(annotation, pydantic.BaseModel):
        models[annotation.__name__] = annotation
        for field in annotation.model_fields.values():
            _collect_pydantic_models(field.annotation, models, seen)
        return

    # We check if this is a parameterized type and recurse if so (eg Union[int, str])
    # (Union is the origin in that case.)
    origin = get_origin(annotation)
    if origin is None:
        return

    for arg in get_args(annotation):
        _collect_pydantic_models(arg, models, seen)


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
