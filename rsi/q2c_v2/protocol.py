"""Typed model-facing catalog; the scientific questions and workflow stay fixed."""
import json
from rsi.q2c.protocol import CATALOG, QUESTIONS, REPORT_SCHEMA, plan_schema
from .contract import catalog, REPORT_TYPES


def research_prompt(task, available):
    metadata = {k: dict(atoms=len(task[k]['frac']), species=task[k]['elements'], edges=len(task[k]['edges'])) for k in ('A', 'B')}
    required = REPORT_TYPES[task['family']]
    return (CATALOG + catalog(available) + QUESTIONS[task['family']]
            + '\nReport evidence must contain exactly these whole-result types, in order: ' + json.dumps(required)
            + '. Each handle must be an exact retained output key. Do not additionally report scalar subfields.'
            + '\nInput metadata only: ' + json.dumps(metadata))
