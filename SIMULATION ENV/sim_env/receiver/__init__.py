"""Receiver component for the RF simulation system.

A deterministic, static-policy scanning (sieve) receiver. It consumes the
repository's RF pulse/event representation and produces structured
:class:`~.models.ReceiverObservation` objects -- the future input to an ML
scheduler. No ML logic lives here.
"""

from .models import DetectionObservation, ReceiverObservation
from .adapter import RadioReceiverBridge, attach_receiver
from .sieve_receiver import (
    SieveReceiver,
    ReceiverConfigError,
    to_hz,
    to_ghz,
    MHZ_TO_HZ,
    ACTION_TUNE,
    ACTION_STEP_UP,
    ACTION_STEP_DOWN,
    ACTION_DWELL,
)

__all__ = [
    "DetectionObservation",
    "ReceiverObservation",
    "RadioReceiverBridge",
    "attach_receiver",
    "SieveReceiver",
    "ReceiverConfigError",
    "to_hz",
    "to_ghz",
    "MHZ_TO_HZ",
    "ACTION_TUNE",
    "ACTION_STEP_UP",
    "ACTION_STEP_DOWN",
    "ACTION_DWELL",
]
