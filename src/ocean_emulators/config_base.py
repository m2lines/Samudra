import argparse
import functools
import os
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any, Self

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

    def save_yaml(self, save_path: Path) -> None:
        """Save config to YAML file."""
        with open(save_path, "w") as f:
            yaml.dump(self.model_dump(), f)

    def inject(
        self,
        path: str,
        *,
        as_key: str | None = None,
        response_model: type | None = None,
    ) -> Callable:
        """A decorator that access a nested config value and injects it as an argument.

        Arguments:
            path: The access path to the value in the config, e.g.
              "experiment.data_dir".
            as_key: If specified, the value will be injected as a keyword to the
              decorated function
            response_model: If specified, we will check that the value at the access
              path is of the expected type.

        Returns:
            The decorated function (which may take additional arguments).
        """
        # Path is expected to be an accessor path for a nested object, for example:
        # "experiment.data_dir". This block of code walks down the tree of attributes
        # until the value (for example "data_dir") is found.
        access_tokens = path.split(".")

        walker = self
        while access_tokens:
            # Pop not only gets the first item, it also shortens the access path.
            # Thus, in the success case, we'll exit the loop because the path is empty
            # and the `walker` variable will have the value we're looking for.
            attribute = access_tokens.pop(0)
            if hasattr(walker, attribute):
                walker = getattr(walker, attribute)
                continue
            # If we miss the child attribute in the tree and the path is not empty,
            # we've gone astray.
            raise ValueError(f"Cannot match {path!r} to config value!")

        # Optionally, we allow the user to type check the found value.
        if response_model is not None:
            if not isinstance(walker, response_model):
                raise ValueError(
                    f'Item ("{walker!r}") at access path {path!r} is not the expected '
                    f"type {response_model!r}."
                )

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Optionally, we allow the user to inject the found value as a keyword
                # argument.
                if as_key is not None:
                    kwargs[as_key] = walker
                    out = func(*args, **kwargs)
                # Otherwise, we inject the found value as a positional argument.
                else:
                    out = func(*args, walker, **kwargs)
                return out

            return wrapper

        return decorator


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
