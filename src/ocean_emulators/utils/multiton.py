import functools
from abc import ABC


class MultitonScope:
    """
    Context manager for managing the scope of Multiton instances.

    You can create a fresh scope and make it active/inactive, for example:

    with MultitonScope() as scope:
        # Create a new scope
        instance = SomeMultiton.init_instance()
        # Use the instance within this scope
        ...

    or it can be disjoint:

    scope = MultitonScope()
    with scope:
        # Create a new scope
        instance = SomeMultiton.init_instance()

    # later on

    with scope:
        # gets the instance created above
        instance = SomeMultiton.instance()

    """

    def __init__(self):
        self.scope = {}

    def __enter__(self):
        self.previous_scope = Multiton._current_scope
        Multiton._current_scope = self.scope
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        assert Multiton._current_scope is self.scope, (
            "MultitonScope.__exit__ called without matching __enter__"
        )
        Multiton._current_scope = self.previous_scope


def scoped(func):
    """Decorator which binds a multiton scope to an instance method."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if getattr(self, "_multiton_scope", None) is None:
            self._multiton_scope = MultitonScope()
        with self._multiton_scope:
            return func(self, *args, **kwargs)

    return wrapper


class Multiton(ABC):
    """
    Allows creation of a set of singletons within some scope, allowing clearing
    and restoring them.
    """

    _current_scope: dict = {}

    def __new__(cls):
        # Prevent direct instantiation
        raise TypeError(
            f"{cls} cannot be instantiated directly. Use init_instance() instead."
        )

    @classmethod
    def get_instance(cls):
        """
        Get the instance of the Multiton class for the current scope.
        """
        instance = cls._current_scope.get(cls)
        if instance is None:
            raise ValueError(f"{cls} not initialized")
        return instance

    @classmethod
    def init_instance(cls, *args, **kwargs):
        if Multiton._current_scope.get(cls) is not None:
            raise ValueError(f"{cls} already initialized")

        instance = super().__new__(cls)
        instance._initialize(*args, **kwargs)
        Multiton._current_scope[cls] = instance

        return instance

    def _initialize(self, *args, **kwargs):
        """
        Override this method to initialize the instance.
        """
        raise NotImplementedError(
            "Subclasses must override _initialize() to set up the instance."
        )
