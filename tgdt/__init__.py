"""Trust-Guided Decision Transformer (TGDT): training, calibration and trust-filtered execution on D4RL."""
from . import config  # noqa: F401
from .data import Setup  # noqa: F401
from .evaluate import evaluate, mean_se  # noqa: F401
from .train import train, get_model  # noqa: F401

__version__ = '1.0.0'
