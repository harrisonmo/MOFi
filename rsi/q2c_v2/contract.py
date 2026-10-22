"""Single-source nominal operation contracts and JSON result schemas."""
from copy import deepcopy
import json
import jsonschema
from rsi.q2c.runtime import check, digest, execute as raw_execute, manual_program, TOOL


def obj(properties):
    return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)


INTEGER = {'type': 'integer'}
NUMBER = {'type': 'number'}
INTS = dict(type='array', items=INTEGER)
VECTOR = dict(type='array', items=INTEGER, minItems=3, maxItems=3)
SCHEMAS = {
    'CycleGroups': dict(type='array', items=obj(dict(atoms=INTS, cycles=dict(type='array', items=VECTOR)))),
    'LatticeRows': dict(type='array', items=obj(dict(atoms=INTS, rank=dict(type='integer', minimum=0, maximum=3), index=dict(type='integer', minimum=1)))),
    'MicResult': obj(dict(squared_distance=dict(type='number', minimum=0), multiplicity=dict(type='integer', minimum=1))),
    'TranslationResult': obj(dict(multiplicity=dict(type='integer', minimum=0), matches=dict(type='array', items=obj(dict(mapping=INTS, origin=dict(type='array', items=NUMBER, minItems=3, maxItems=3)))))),
}
FIELDS = {
    'EvidenceBundle': dict(graph_a='LatticeRows', graph_b='LatticeRows', mic='MicResult', mappings='TranslationResult'),
    'MicResult': dict(squared_distance='Number', multiplicity='Integer'),
    'TranslationResult': dict(multiplicity='Integer', matches='Mappings'),
}
SCHEMAS['EvidenceBundle'] = obj({k: SCHEMAS[v] for k, v in FIELDS['EvidenceBundle'].items()})
SCHEMAS.update(Integer=INTEGER, Number=NUMBER, Mappings=SCHEMAS['TranslationResult']['properties']['matches'])
SIGNATURES = {
    'cycle_groups': (['Record'], 'CycleGroups'),
    'integer_lattice': (['CycleGroups'], 'LatticeRows'),
    'mic_probe': (['Record', 'Record'], 'MicResult'),
    'translation_maps': (['Record', 'Record'], 'TranslationResult'),
    TOOL: (['Record', 'Record'], 'EvidenceBundle'),
}
REPORT_TYPES = {'G': ['LatticeRows', 'LatticeRows'], 'M': ['MicResult'], 'S': ['TranslationResult']}


def reference_type(types, ref):
    parts = ref.split('.')
    kind = types[parts[0]]
    for field in parts[1:]:
        if field not in FIELDS.get(kind, {}):
            raise ValueError(f'{ref}: {kind} has no projectable field {field}')
        kind = FIELDS[kind][field]
    return kind


def typecheck(program, tool=None, author=False):
    check(program, tool, author=author)
    if tool is not None:
        typecheck(tool, author=True)
    types = dict(A='Record', B='Record')
    for node in program['nodes']:
        expected, result = SIGNATURES[node['op']]
        actual = [reference_type(types, ref) for ref in node['inputs']]
        if actual != expected:
            raise ValueError(f"{node['id']}: {node['op']} expects {expected}, got {actual}")
        types[node['id']] = result
    outputs = {ref: reference_type(types, ref) for ref in program['outputs']}
    if author and outputs != FIELDS['EvidenceBundle']:
        raise ValueError('Registered outputs must have the exact EvidenceBundle types')
    return outputs


def validate_results(program, output, tool=None):
    for ref, kind in typecheck(program, tool).items():
        if kind in SCHEMAS:
            jsonschema.validate(output['outputs'][ref], SCHEMAS[kind])


def execute(program, a, b, tool=None):
    typecheck(program, tool)
    output = raw_execute(program, a, b, tool)
    validate_results(program, output, tool)
    return output


def manifest(tool):
    typecheck(tool, author=True)
    return dict(name=TOOL, contract_version='q2c-v2-typed-1', program_version=tool['version'],
                program_sha256=digest(tool), contract_sha256=digest(dict(signatures=SIGNATURES, fields=FIELDS, schemas=SCHEMAS)),
                signatures=deepcopy(SIGNATURES), fields=deepcopy(FIELDS), schemas=deepcopy(SCHEMAS))


def catalog(available):
    signatures = {k: v for k, v in SIGNATURES.items() if available or k != TOOL}
    fields = {k: v for k, v in FIELDS.items() if available or k != 'EvidenceBundle'}
    return ('\nMachine-readable nominal input/output contract: ' + json.dumps(dict(signatures=signatures, fields=fields))
            + '\nOutput JSON schemas: ' + json.dumps({k: v for k, v in SCHEMAS.items() if available or k != 'EvidenceBundle'})
            + '\nLatticeRows is a final list of {atoms,rank,index}, not CycleGroups. '
              'integer_lattice only accepts CycleGroups. Do not recalculate a LatticeRows result. '
              'Field projection selects an existing computed value; it does not call a primitive. '
              'A retained whole MicResult/TranslationResult is different from its scalar fields.\n')
