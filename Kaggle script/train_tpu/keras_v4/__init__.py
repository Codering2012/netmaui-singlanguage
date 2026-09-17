"""
ASL Keras V4 Flagship Foundation Suite — Multi-Backend (JAX / PyTorch / TensorFlow)
Targeting Keras 3.15.1 on Cloud TPU v5e/v4 and local verification.
"""

from .train_all_in_one_keras import (
    ASLKerasFoundationModel,
    KerasHomoscedasticLossWrapper,
    KerasSignDPOLoss,
    KerasSoftDTWLoss,
    KerasTwoStreamMeshVisualFusion,
    KerasBattisonDominanceSymmetry,
    KerasProsodicGrammarScope,
    KerasDynamicPhonologicalCondenser,
    KerasSinkhornTransducer,
    KerasPerceiverResampler,
    ConformerBlock,
)

__version__ = "4.0.0-keras"
