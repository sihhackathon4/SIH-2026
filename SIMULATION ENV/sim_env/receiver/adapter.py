"""First-class connection between :class:`RadioEnvironment` and a
:class:`~.sieve_receiver.SieveReceiver`.

The environment emits a live stream of :class:`~.environment.SimulationEvent`
objects (``entry`` / ``exit`` / ``snapshot``). This bridge subscribes to that
stream (via the environment's ``on_event`` callback list) and feeds the events
into the receiver *without* re-tuning the receiver to every arriving pulse and
*without* setting the receiver clock to every pulse's arrival time.

The receiver keeps its own deterministic clock and scan position; the bridge
merely makes the receiver aware of pulses that have actually arrived in the
simulated timeline and clears them once they exit. Detection then depends on the
receiver's real frequency window and dwell interval overlap.

::

    RadioEnvironment
         |
         | on_event(SimulationEvent)     <- bridge attached here
         v
    RadioReceiverBridge
         |  entry -> receiver.add_pulse(...)
         |  exit  -> receiver.remove_pulse(...)
         v
    SieveReceiver (own clock / scan via scan_once)
         |
         v
    ReceiverObservation
"""

from __future__ import annotations

from typing import List, Optional

from .sieve_receiver import SieveReceiver

__all__ = ["RadioReceiverBridge", "attach_receiver"]


class RadioReceiverBridge:
    """A live event consumer that drives a :class:`SieveReceiver` from an
    environment event stream.

    The bridge performs no ML logic and no frequency selection. It only
    propagates ``entry`` / ``exit`` bookkeeping into the receiver's time-aware
    pulse buffer; the *static* interception policy lives in the receiver.
    """

    def __init__(self, receiver: SieveReceiver):
        self.receiver = receiver

    # ------------------------------------------------------------------ events

    def on_event(self, event) -> None:
        """Environment event callback (safe to pass straight to ``on_event``).

        The environment and the receiver share one simulated clock. On every
        event the bridge keeps the receiver's clock synced to the environment's
        current simulated time, so the receiver is aware of what has *already*
        happened. It never leaps forward to a future event on its own: it only
        sees the current ``env`` time, which never exceeds what the environment
        has already produced.
        """
        if event is None:
            return
        if isinstance(event, dict):
            etype = event.get("event")
            pulse = event.get("pulse")
            pulse_id = event.get("pulse_id")
            time_us = event.get("time_us")
        else:
            etype = getattr(event, "event_type", getattr(event, "event", None))
            pulse = getattr(event, "pulse", None)
            pulse_id = getattr(event, "pulse_id", None)
            time_us = getattr(event, "time_us", None)
        # Sync the receiver clock to the (already-produced) simulated time.
        if time_us is not None and time_us > self.receiver.current_time_us:
            self.receiver.advance(float(time_us))
        if etype == "entry":
            if pulse is not None:
                self.receiver.add_pulse(pulse)
        elif etype == "exit":
            if pulse_id is not None or (isinstance(pulse, dict) and pulse.get("pulse_id") is not None):
                pid = pulse_id if pulse_id is not None else pulse.get("pulse_id")
                self.receiver.remove_pulse(pid)

    # -------------------------------------------------------------- scanning

    def scan_steps(self, n: int = 1) -> List:
        """Run ``n`` deterministic static scan steps on the receiver's own clock.

        The receiver's clock advances by ``dwell_time_us`` per step and its
        frequency by ``frequency_step_mhz``; it is never retuned to an arriving
        pulse. Returns the list of observations (one per dwell).
        """
        return self.receiver.scan(n)


def attach_receiver(env, receiver: SieveReceiver) -> RadioReceiverBridge:
    """Attach a :class:`SieveReceiver` to a running :class:`RadioEnvironment`.

    This registers the bridge callback on the environment so that live events
    propagate into the receiver. Returns the :class:`RadioReceiverBridge` so a
    caller can drive scanning afterwards.
    """
    bridge = RadioReceiverBridge(receiver)
    env.add_callback(bridge.on_event)
    return bridge
