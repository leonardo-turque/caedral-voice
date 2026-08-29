"""CAEDRAL-NAR model package (Phase F).

Public surface:
    CaedralNAR(config)          -> full NAR prototype module
    CaedralNARConfig            -> dataclass config (1:1 with configs/*.json)
    load_model(path, seed=None) -> seeded build from a JSON config
    BackboneConfig / Transformer        (backbone.py)
    ConditioningProvider / Duration|Pitch|EnergyPredictor / expand_tokens
    IstftHead                          (istft_head.py)
    FlowDecoder / Invertible1x1Conv    (flow_decoder.py, CV-G1-E0006)
    maximum_path                       (mas.py, CV-G1-E0006)

Prototype-only: untrained stubs; contracts are param targets, shapes,
determinism, conditioning sensitivity, and CPU measurability.
"""
from .backbone import BackboneConfig, Transformer, TransformerBlock
from .caedral_nar import CaedralNAR, CaedralNARConfig, load_model
from .ctc_aligner import (
    CTCAligner,
    CTCAlignerConfig,
    expand_ds_durations_to_mel,
    viterbi_force_align,
)
from .flow_decoder import AffineCoupling, FlowDecoder, Invertible1x1Conv
from .istft_head import ConvNeXtBlock1d, IstftHead
from .mas import maximum_path, maximum_path_reference
from .variance import (
    ConditioningProvider,
    DurationPredictor,
    EnergyPredictor,
    FiLM,
    PitchPredictor,
    VariancePredictor,
    expand_tokens,
)

__all__ = [
    "CaedralNAR",
    "CaedralNARConfig",
    "load_model",
    "BackboneConfig",
    "Transformer",
    "TransformerBlock",
    "ConditioningProvider",
    "VariancePredictor",
    "DurationPredictor",
    "PitchPredictor",
    "EnergyPredictor",
    "FiLM",
    "expand_tokens",
    "IstftHead",
    "ConvNeXtBlock1d",
    "FlowDecoder",
    "AffineCoupling",
    "Invertible1x1Conv",
    "maximum_path",
    "maximum_path_reference",
    "CTCAligner",
    "CTCAlignerConfig",
    "viterbi_force_align",
    "expand_ds_durations_to_mel",
]
