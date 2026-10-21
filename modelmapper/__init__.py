"""Model Mapper: find which neurons of a language model a concept depends on, and what they do.

    from modelmapper import map_concept
    r = map_concept("Bitcoin", model="Qwen/Qwen3-0.6B")     # -> MapResult
    r.top[0].neuron, r.top[0].drop, r.nulls                  # its strongest neuron, the drop, the random controls

Functions: map_concept, map_batch, steer, lookup, load_model, check. Results: MapResult, NeuronEffect, HeadEffect.
README.md says what the numbers mean, METHOD.md how they are measured, FORMAT.md what a record holds.
Importing this package is light; torch and the model load only when a function needs them.
"""
__version__ = "0.2.0"

from modelmapper.api import LabellerError, check, load_model, lookup, map_batch, map_concept, steer  # noqa: E402
from modelmapper.records import HeadEffect, MapResult, NeuronEffect  # noqa: E402

__all__ = ["map_concept", "map_batch", "steer", "lookup", "load_model", "check", "MapResult", "NeuronEffect", "HeadEffect",
           "LabellerError", "__version__"]
