import logging
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any

log = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """
    Section 5: Base Strategy Class for Institutional Diversification.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        pass
