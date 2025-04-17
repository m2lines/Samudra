import argparse
import os
from pathlib import Path
from typing import Any, Optional, Self

import yaml
from pydantic import BaseModel, ConfigDict
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    CliSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# Set up yaml.safe_load to include other yaml files via !include


def include_constructor(loader, node):
    filename = os.path.normpath(
        os.path.join(os.path.dirname(loader.stream.name), node.value)
    )
    with open(filename, "r") as f:
        return yaml.safe_load(f)


# This is arguably unsafe, but we don't parse untrusted YAML
yaml.loader.SafeLoader.add_constructor("!include", include_constructor)


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
        args_to_parse: Optional[list[str]] = None,
    ) -> Self:
        """Load config from YAML & CLI with validation."""
        parser = argparse.ArgumentParser(
            description=cls.__doc__,
            epilog="""
YAML files can include other YAML files using the !include tag,
as in `data: !include configs/data/something.yaml`
You can also replace any JSON argument listed above with a YAML file by
specifying it with an @ symbol, eg `--some_param=@configs/data/something.yaml`.
""",
        )
        parser.add_argument("config", type=str, help="Path to config YAML file")

        cli_source = IncludeYamlCliSettingsSource(
            cls,
            root_parser=parser,
            # If args_to_parse is None, we parse argv, which is what `True` does
            cli_parse_args=args_to_parse if args_to_parse is not None else True,
        )

        # We do this after creating CliSettingsSource (which populates the parser)
        # so the help is complete on error.
        args = parser.parse_args(args_to_parse)

        # Then we read the YAML file specified in the CLI
        # Note that by default, YamlConfigSettingsSource will ignore missing files
        config_path = Path(args.config).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file `{args.config}` (full path: {config_path}) not found"
            )
        yaml_values = YamlConfigSettingsSource(cls, yaml_file=config_path)()

        return cls(
            _cli_settings_source=cli_source,
            **yaml_values,
        )

    def save_yaml(self, save_path: Path):
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
            with open(value[1:], "r") as f:
                return yaml.safe_load(f)
        return super().decode_complex_value(field_name, field, value)
