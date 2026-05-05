import argparse
import os
import textwrap
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict
from pydantic.fields import FieldInfo
try:
    from pydantic_settings import (
        BaseSettings,
        CliSettingsSource,
        PydanticBaseSettingsSource,
        SettingsConfigDict,
        YamlConfigSettingsSource,
    )

    HAS_PYDANTIC_SETTINGS = True
except ModuleNotFoundError:
    BaseSettings = BaseModel  # type: ignore[misc,assignment]
    CliSettingsSource = object  # type: ignore[assignment]
    PydanticBaseSettingsSource = Any
    SettingsConfigDict = ConfigDict  # type: ignore[assignment]
    YamlConfigSettingsSource = None
    HAS_PYDANTIC_SETTINGS = False


def register_include_constructor():
    """Set up yaml.safe_load to include other yaml files via !include."""

    def include_constructor(loader: yaml.Loader, node: yaml.Node) -> Any:
        if hasattr(loader.stream, "name"):
            name = loader.stream.name  # type: ignore
        else:
            raise ValueError(
                "To support includes, you must load a file object, not a string"
            )
        filename = os.path.normpath(os.path.join(os.path.dirname(name), node.value))
        with open(filename) as f:
            return yaml.safe_load(f)

    # This is arguably unsafe, but we don't parse untrusted YAML
    yaml.loader.SafeLoader.add_constructor("!include", include_constructor)


register_include_constructor()


class BaseConfig(BaseModel):
    """
    Base class for all configs.
    """

    model_config = ConfigDict(extra="forbid")


class TopLevelConfig(BaseSettings):
    """
    Base class for top-level configs (ie tasks like train or eval).
    """

    model_config = SettingsConfigDict(nested_model_default_partial_update=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # We don't need env/dotenv/secrets, yaml & CLI are injected in from_yaml_and_cli
        return (init_settings,)

    @classmethod
    def from_yaml_and_cli(
        cls,
        args_to_parse: list[str] | None = None,
    ) -> Self:
        """Load config from YAML & CLI with validation."""
        parser = argparse.ArgumentParser(
            description=cls.__doc__,
            epilog=textwrap.dedent(
                """
                YAML files can include other YAML files using the !include tag,
                as in `data: !include configs/data/something.yaml`
                You can also replace any JSON argument listed above with a YAML file by
                specifying it with an @ symbol,
                eg `--some_param=@configs/data/something.yaml`.
                """
            ),
        )
        parser.add_argument("config", type=str, help="Path to config YAML file")
        if HAS_PYDANTIC_SETTINGS:
            cli_source = IncludeYamlCliSettingsSource(
                cls,
                root_parser=parser,
                # If args_to_parse is None, we parse argv, which is what `True` does
                cli_parse_args=args_to_parse if args_to_parse is not None else True,
            )
            # We do this after creating CliSettingsSource (which populates the parser)
            # so the help is complete on error.
            args = parser.parse_args(args_to_parse)
        else:
            args, unknown_args = parser.parse_known_args(args_to_parse)

        # Then we read the YAML file specified in the CLI
        # Note that by default, YamlConfigSettingsSource will ignore missing files
        config_path = Path(args.config).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file `{args.config}` (full path: {config_path}) not found"
            )
        if HAS_PYDANTIC_SETTINGS:
            yaml_values = YamlConfigSettingsSource(cls, yaml_file=config_path)()
            return cls(
                _cli_settings_source=cli_source,
                **yaml_values,
            )

        with open(config_path) as f:
            yaml_values = yaml.safe_load(f) or {}
        cli_values = parse_cli_overrides(unknown_args)
        merged_values = deep_update(yaml_values, cli_values)
        return cls(**merged_values)

    def save_yaml(self, save_path: Path) -> None:
        """Save config to YAML file."""
        with open(save_path, "w") as f:
            yaml.dump(self.model_dump(), f)


class IncludeYamlCliSettingsSource(CliSettingsSource):
    """CliSettingsSource which permits @filename.yaml for JSON arguments."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def decode_complex_value(
        self, field_name: str, field: FieldInfo, value: Any
    ) -> Any:
        if isinstance(value, str) and value.startswith("@"):
            with open(value[1:]) as f:
                return yaml.safe_load(f)
        return super().decode_complex_value(field_name, field, value)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge CLI override values into the YAML-loaded config."""
    merged = dict(base)
    for key, value in updates.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def parse_cli_overrides(args: list[str]) -> dict[str, Any]:
    """Parse `--foo.bar value` style overrides when pydantic-settings is absent."""
    overrides: dict[str, Any] = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if not arg.startswith("--"):
            raise ValueError(f"Unexpected CLI token without flag: {arg}")
        key = arg[2:]
        if i + 1 >= len(args):
            raise ValueError(f"Missing value for CLI override `{arg}`")
        raw_value = args[i + 1]
        if raw_value.startswith("@"):
            with open(raw_value[1:]) as f:
                value = yaml.safe_load(f)
        else:
            try:
                value = yaml.safe_load(raw_value)
            except yaml.YAMLError:
                value = raw_value

        cursor = overrides
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
        i += 2
    return overrides
