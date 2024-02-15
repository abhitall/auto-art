"""
Evaluation observers for monitoring and logging evaluation progress.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class EvaluationEvent:
    """Event data for evaluation notifications."""
    event_type: str
    timestamp: datetime
    data: Dict[str, Any]

class EvaluationObserver(ABC):
    """Base class for evaluation observers."""
    
    def __init__(self):
        """Initialize the observer."""
        self._events: List[EvaluationEvent] = []
    
    @abstractmethod
    def on_evaluation_start(self, data: Dict[str, Any]) -> None:
        """Handle evaluation start event."""
        pass
    
    @abstractmethod
    def on_evaluation_progress(self, data: Dict[str, Any]) -> None:
        """Handle evaluation progress event."""
        pass
    
    @abstractmethod
    def on_evaluation_complete(self, data: Dict[str, Any]) -> None:
        """Handle evaluation completion event."""
        pass
    
    @abstractmethod
    def on_evaluation_error(self, data: Dict[str, Any]) -> None:
        """Handle evaluation error event."""
        pass
    
    def get_events(self) -> List[EvaluationEvent]:
        """Get all recorded events."""
        return self._events
    
    def clear_events(self) -> None:
        """Clear all recorded events."""
        self._events.clear()
    
    def _record_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Record an event with timestamp."""
        event = EvaluationEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            data=data
        )
        self._events.append(event) 