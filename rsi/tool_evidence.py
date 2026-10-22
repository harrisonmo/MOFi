"""Trusted projection of a real Crystal-Builder UFF optimization result."""

from rsi.tool_runtime import digest, validate_input


def from_uff_event(event):
    """Called by the science broker, never by an untrusted candidate."""
    result = event["result"]
    if event["action"]["verb"] != "optimize" or result.get("ok") is not True:
        raise ValueError("A successful UFF optimization event is required")
    data = result["data"]
    payload = {key: data[key] for key in ("initial_energy", "energy", "max_force", "converged")}
    payload.update(energy_unit="kcal/mol", force_unit="kcal/mol/A")
    validate_input(payload)
    return {"payload": payload, "event_sha256": digest(event),
            "unit_source": "Crystal-Builder xtal.agent.session.Session.optimize at df014447"}
