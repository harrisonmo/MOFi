"""Versioned reporting semantics shared by agents and the independent evaluator."""

REPORT_PROTOCOL_VERSION = 3
REPORT_INSTRUCTIONS = (
    "Report task_completed=true only if every authorized obligation was fulfilled, including saving. "
    "An explicitly requested conditional refusal or bounded nonconverged calculation can complete a task. "
    "Separately report scientific_status: blocked for an observed structural blocker; needs_input for "
    "unresolved chemistry requiring a user decision; not_converged or converged only for a calculation "
    "with an explicit convergence result; otherwise not_assessed. These states describe the evidence "
    "for the requested task, not general physical validity. Task completion does not imply scientific readiness. "
    "If several states apply, use blocked, then needs_input, then the latest calculation convergence, "
    "then not_assessed. List observed diagnostic codes and explain limitations in the summary."
)
REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "task_completed": {"type": "boolean"},
        "scientific_status": {
            "type": "string",
            "enum": ["not_assessed", "blocked", "needs_input", "not_converged", "converged"],
        },
        "diagnostic_codes": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string", "minLength": 1},
    },
    "required": ["task_completed", "scientific_status", "diagnostic_codes", "summary"],
    "additionalProperties": False,
}
