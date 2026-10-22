"""Frozen structured transport for the approved Q2 Astra pilot."""

from rsi.tool_runtime import INPUT_FIELDS, OUTPUT_FIELDS, CONTRACT


def obj(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


EXPR = {"anyOf": [
    obj({"field": {"type": "string", "enum": sorted(INPUT_FIELDS)}}),
    obj({"number": {"type": "number"}}),
    obj({"op": {"type": "string", "enum": ["sub", "add", "mul"]},
         "args": {"type": "array", "items": {"$ref": "#/$defs/expr"}, "minItems": 2, "maxItems": 2}}),
    obj({"op": {"type": "string", "enum": ["convert"]},
         "dimension": {"type": "string", "enum": ["energy", "force"]},
         "args": {"type": "array", "items": {"$ref": "#/$defs/expr"}, "minItems": 2, "maxItems": 2}}),
]}
SPEC_SCHEMA = obj({"id": {"type": "string", "enum": ["uff_audit"]},
                   "version": {"type": "integer", "enum": [1]},
                   "contract": {"type": "string", "enum": [CONTRACT]},
                   "origin": {"type": "string", "enum": ["outer_generated"]},
                   "outputs": obj({key: {"$ref": "#/$defs/expr"} for key in sorted(OUTPUT_FIELDS)})})
AUTHOR_SCHEMA = obj({"hypothesis": {"type": "string"}, "spec": SPEC_SCHEMA})
AUTHOR_SCHEMA["$defs"] = {"expr": EXPR}
PLAN_SCHEMA = obj({"requests": {"type": "array", "maxItems": 4, "items": obj({
    "tool": {"type": "string"}, "arguments": {"anyOf": [obj({}), {"$ref": "#/$defs/expr"}]}})}})
PLAN_SCHEMA["$defs"] = {"expr": EXPR}
FINAL_SCHEMA = obj({key: {"type": "boolean" if key == "converged" else "number"}
                    for key in sorted(OUTPUT_FIELDS)})

LANGUAGE = (
    "Expression syntax: {field: INPUT_FIELD}, {number: NUMERIC_LITERAL}, "
    "{op: sub|add|mul, args: [EXPR, EXPR]}, or "
    "{op: convert, dimension: energy|force, args: [VALUE_EXPR, UNIT_EXPR]}. "
    "Convert returns kJ/mol for energy or kJ/mol/nm for force. "
    "Input units: kcal/mol or kJ/mol; kcal/mol/A or kJ/mol/nm. "
    "1 kcal/mol = 4.184 kJ/mol and 1 kcal/mol/A = 41.84 kJ/mol/nm. "
    "Numeric inputs are finite and bounded; max_force is a nonnegative magnitude. "
    "No paths, code execution, imports, branches, loops or science actions. "
    "The controller limits each manifest to 16384 bytes, 96 nodes and depth 10. "
)
RESEARCH_TRANSPORT = (
    LANGUAGE + "For plan return requests only. An expression tool takes one expression as arguments. "
    "Any registered audit tool takes empty arguments {}; its input is bound to the observation. "
    "For report return the four required output fields only. "
    "Do not round results: preserve floating point precision. "
    "You may use only tools listed in the context. Do not access files or other services."
)
AUTHOR_INSTRUCTION = (
    "Act as the outer tool author. Propose one bounded expression tool, id uff_audit, version 1, "
    "origin outer_generated. Only compose the fixed numerical primitives. "
    "Return a hypothesis and manifest, not Python or a workflow. "
    "Inputs: initial_energy, energy, max_force, energy_unit, force_unit, converged. "
    "Outputs: energy_kj_mol, delta_kj_mol (final minus initial), max_force_kj_mol_nm, "
    "and the explicit input converged flag. Energy decrease never proves convergence. "
    "Support both admitted unit systems and negative energies. "
    "You cannot access or modify evaluator, held-out data, manual reference tool or science engine. "
    + LANGUAGE
)
