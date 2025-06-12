"""
Base classes for defence strategies.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

class DefenceStrategy(ABC):
    """Base class for all defence strategies."""

    def __init__(self, defence_name: str = "UnnamedDefence"):
        """Initializes the defence strategy.

        Args:
            defence_name: A descriptive name for the defence strategy.
        """
        self.name = defence_name

    @abstractmethod
    def apply(self, art_estimator: Any, **kwargs) -> Any:
        """
        Apply the defence to an ART estimator.

        Args:
            art_estimator: The ART estimator to be defended.
            **kwargs: Additional parameters for the defence.

        Returns:
            A new, defended ART estimator.
        """
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get defence parameters.

        Returns:
            Dictionary of parameter names and values.
        """
        pass

    def set_params(self, **params: Any) -> None:
        """
        Set defence parameters.
        Default implementation does nothing, subclasses should override if params are mutable.
        """
        pass
