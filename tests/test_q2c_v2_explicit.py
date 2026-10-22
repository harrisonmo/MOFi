"""Reject invalid inherited admission before reusing fixed-path evidence."""
from copy import deepcopy
import pytest
from rsi.q2c_v2_explicit.run import validate_parent


def fixture():
    tasks = [dict(id=str(i)) for i in range(6)]
    rows = [dict(id=t['id'], arm='fixed', completed=True, passed=True, route_passed=True, registered_uses=0) for t in tasks]
    return tasks, rows, dict(completed=True, stop=None), dict(blocked=None, entries=[dict(tokens=10)])


def test_valid_fixed_reuse():
    assert validate_parent(*fixture())


@pytest.mark.parametrize('field', ['completed', 'passed', 'route_passed'])
def test_reject_failed_fixed(field):
    args = fixture()
    args[1][0][field] = False
    with pytest.raises(RuntimeError):
        validate_parent(*args)


def test_reject_unknown_cost():
    args = fixture()
    args[3]['entries'][0]['tokens'] = None
    with pytest.raises(RuntimeError):
        validate_parent(*args)


def test_reject_duplicated_inherited_task():
    args = fixture()
    args[1][-1] = deepcopy(args[1][0])
    with pytest.raises(RuntimeError):
        validate_parent(*args)
