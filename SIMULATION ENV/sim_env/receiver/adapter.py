"""Live RadioEnvironment -> SieveReceiver integration.

This bridge connects the existing event-driven RadioEnvironment to the
SieveReceiver without:

    - retuning to each arriving pulse;
    - setting receiver time directly to every pulse;
    - exposing future pulses to the receiver.

The receiver maintains its own deterministic scan state and pulse buffer.
"""

from __future__ import annotations

from typing import List

from .sieve_receiver import SieveReceiver


__all__ = [
    "RadioReceiverBridge",
    "attach_receiver",
]


class RadioReceiverBridge:
    """Connect a SieveReceiver to RadioEnvironment callbacks."""

    def __init__(
        self,
        receiver: SieveReceiver,
    ) -> None:
        self.receiver = receiver

    def on_event(
        self,
        event,
    ) -> None:
        """Consume one live RadioEnvironment event.

        The receiver itself decides how to advance its clock and update its
        pulse buffer.  The bridge does not tune or modify receiver time.
        """
        if event is None:
            return

        self.receiver.handle_environment_event(
            event
        )

    def scan_steps(
        self,
        n: int = 1,
    ) -> List:
        """Run deterministic receiver scan steps."""
        count = int(n)

        if count < 0:
            raise ValueError(
                "n must be >= 0"
            )

        return self.receiver.scan(
            count
        )


def attach_receiver(
    env,
    receiver: SieveReceiver,
) -> RadioReceiverBridge:
    """Attach a receiver to an existing RadioEnvironment.

    Environment events are delivered through the repository's callback
    mechanism.
    """
    bridge = RadioReceiverBridge(
        receiver
    )

    env.add_callback(
        bridge.on_event
    )

    return bridge
