"""Synthesis engine interface and registry (§10.6).

Separates the statistical specification from the sampling engine so alternative
engines can be selected behind a stable interface without forking pipeline logic.
"""

from __future__ import annotations

from typing import Protocol, Dict, Any, runtime_checkable
import pandas as pd
from agentdataset.models.schemas import Parameters


@runtime_checkable
class SynthesisEngine(Protocol):
    def synthesize(self, parameters: Parameters, noise_level: float = 0.1, **kwargs) -> pd.DataFrame:
        ...

    @property
    def last_diagnostics(self) -> Any:
        ...


class EngineRegistry:
    _engines: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, engine_cls: type) -> None:
        cls._engines[name] = engine_cls

    @classmethod
    def get(cls, name: str) -> type:
        if name not in cls._engines:
            raise KeyError(f"Unknown synthesis engine: {name}")
        return cls._engines[name]

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._engines.keys())

    @classmethod
    def create(cls, name: str, **kwargs):
        return cls.get(name)(**kwargs)


# Register the default engine lazily to avoid circular imports.
def _register_defaults():
    from agentdataset.core.synthesizer import Synthesizer
    EngineRegistry.register("gaussian_copula", Synthesizer)
    EngineRegistry.register("default", Synthesizer)


_register_defaults()