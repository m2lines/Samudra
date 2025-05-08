import argparse
import functools
import inspect
import logging
import os
import textwrap
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any, Self, get_type_hints

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, create_model
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    CliSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

logger = logging.getLogger(__name__)


def _type_check(value: Any, expected_type: type) -> bool:
    """Check if a value matches a parameter type, including parameterized types."""
    try:
        # Create a temporary model with a field of the target type
        TempModel = create_model("TempModel", value=(expected_type, ...))
        # Attempt to validate
        TempModel(value=value)
        return True
    except ValidationError:
        return False


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

    def bind(self, func=None, *, mappings=None) -> Callable:
        """
        Decorator that binds config attributes to function parameters based on types.

        By default, it will:
        - Auto-inject full models/data classes that match parameter types
        - Use mappings for specific access paths when provided

        Args:
            func (callable): Function to bind. Allows decorator to take optional args.
            mappings: Optional mapping of function parameter names to config attribute
               paths. The paths can be dot-notated for nested attributes
               (e.g., "db.connection.host")
        """
        if func is None:
            return functools.partial(self.bind, mappings=mappings)

        mappings = mappings or {}

        # Get function signature and type hints
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a copy of kwargs that we can modify
            updated_kwargs = kwargs.copy()

            # Skip the first parameter if it's 'self' (for methods)
            bound_args = sig.bind_partial(*args)
            params_to_inject = list(sig.parameters.keys())
            if params_to_inject and params_to_inject[0] in bound_args.arguments:
                # This is likely a method call, so skip first param
                params_to_inject = params_to_inject[1:]

            # For each parameter that isn't already provided...
            for param_name in params_to_inject:
                if param_name in updated_kwargs:
                    continue  # Already provided explicitly

                param_type = type_hints.get(param_name)
                if not param_type:
                    continue  # No type hint to match

                # Check if there's a mapping for this parameter
                if param_name in mappings:
                    # Use the mapping to get a specific path
                    attr_path = mappings[param_name]
                    attr_value = self._get_nested_attr(attr_path)

                    # If the attribute is not found, skip this parameter
                    if attr_value is None:
                        break

                    if isinstance(attr_value, param_type):
                        updated_kwargs[param_name] = attr_value
                    elif (
                        isinstance(param_type, type)
                        and issubclass(param_type, BaseModel)
                        and isinstance(attr_value, dict)
                    ):
                        updated_kwargs[param_name] = param_type.model_validate(
                            attr_value
                        )
                else:
                    # No mapping - try to auto-inject based on type
                    attr_value = self._find_attribute_by_type(param_name, param_type)
                    if attr_value is not None:
                        updated_kwargs[param_name] = attr_value

            # Call the function with the updated kwargs
            return func(*args, **updated_kwargs)

        return wrapper

    def _get_nested_attr(self, attr_path: str) -> Any | None:
        """
        Get a nested attribute from the config using dot notation.

        Args:
            attr_path: Attribute path using dot notation (e.g., "db.connection.host")

        Returns:
            The attribute value or None if not found
        """
        walker = self

        # Handle the simple case first
        if "." not in attr_path:
            return getattr(walker, attr_path, None)

        # Handle nested paths
        parts = attr_path.split(".")
        for part in parts:
            if not hasattr(walker, part):
                return None
            walker = getattr(walker, part)

            # If we hit a non-object that can't have attributes, but we still have path
            # components, then the path is invalid
            if (
                not hasattr(walker, "__dict__")
                and not isinstance(walker, dict)
                and part != parts[-1]
            ):
                return None

        return walker

    def _find_attribute_by_type(self, param_name: str, param_type: type) -> Any | None:
        """
        Find an attribute in the config that matches the parameter type using
        iterative breadth-first search through the object hierarchy.

        Args:
            param_name: The parameter name
            param_type: The parameter type (target)

        Returns:
            Matching attribute value or None if not found
        """
        # Initialize queue for BFS and visited set to prevent cycles
        queue: deque = deque([(self, None, [])])  # (obj, attr_name, path)
        visited = set()

        while queue:
            current_obj, attr_name, path = queue.popleft()

            # Skip if already visited
            obj_id = id(current_obj)
            if obj_id in visited:
                continue
            visited.add(obj_id)

            current_path: list[str] = path + [attr_name] if attr_name else []
            logger.debug(".".join(current_path))

            # First check if current object matches the type
            # This only applies if this is not the starting object
            if attr_name is not None:
                # First priority: name and type match -- this breaks ties.
                if attr_name == param_name and _type_check(current_obj, param_type):
                    return current_obj

                # Second priority: type match only
                if _type_check(current_obj, param_type):
                    return current_obj

            # Then process all attributes of the current object
            if isinstance(current_obj, BaseModel):
                for name, value in current_obj.__dict__.items():
                    # Check direct attributes
                    if name == param_name and isinstance(value, param_type):
                        return value  # Exact name and type match has highest priority

                    self._enqueue_if_explorable(queue, value, name, current_path)

            # Special handling for containers
            elif isinstance(current_obj, list):
                for i, item in enumerate(current_obj):
                    self._enqueue_if_explorable(queue, item, f"[{i}]", current_path)

            elif isinstance(current_obj, dict):
                for key, value in current_obj.items():
                    self._enqueue_if_explorable(
                        queue,
                        value,
                        f"[{key}]",
                        current_path,
                    )

        # No match found
        return None

    def _enqueue_if_explorable(
        self, queue: deque, obj: Any, name: str, path: list[str]
    ) -> None:
        """
        Helper method to enqueue objects that can be explored further.

        Args:
            queue: The BFS queue
            obj: The object to potentially enqueue
            name: The attribute name
            path: The path so far
        """
        # Only enqueue objects that might contain nested attributes
        if isinstance(obj, (BaseModel, list, dict)) or hasattr(obj, "__dict__"):
            queue.append((obj, name, path))


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
