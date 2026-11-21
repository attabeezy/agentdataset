from agentdataset.core.synthesis_engine import EngineRegistry, SynthesisEngine
from agentdataset.core.synthesizer import Synthesizer


def test_registry_available():
    assert "default" in EngineRegistry.available()
    assert "gaussian_copula" in EngineRegistry.available()


def test_registry_create():
    eng = EngineRegistry.create("default", n_rows=10, seed=1)
    assert isinstance(eng, Synthesizer)
    assert isinstance(eng, SynthesisEngine)