"""Offline checks for model identity, tool disabling and transport admission."""
import json
from pathlib import Path
import pytest
from scripts import run_blind_codex_cli as driver
from rsi.backends import codex_cli as backend


def test_exact_requested_model_and_high(tmp_path):
    command = driver.command_for(tmp_path, tmp_path / 'final.txt')
    assert command[command.index('--model') + 1] == 'gpt-6.1-sol'
    assert 'model_reasoning_effort="high"' in command
    assert '--ephemeral' in command and '--ignore-user-config' in command
    assert 'project_doc_max_bytes=0' in command
    assert 'web_search="disabled"' in command
    assert 'features.skip_host_skill_discovery=true' in command
    assert all(command[command.index(feature) - 1] == '--disable' for feature in driver.DISABLED)
    assert command[command.index('--sandbox') + 1] == 'read-only'


@pytest.mark.parametrize('item_type,expected', [('agent_message', True), ('command_execution', False), ('mcp_tool_call', False)])
def test_transport_rejects_tool_events(tmp_path, monkeypatch, item_type, expected):
    class Process:
        returncode = 0
        def __init__(self, command, **kwargs):
            final = Path(command[command.index('--output-last-message') + 1])
            final.write_text('{"ready":true,"tools_listing":[]}')
        def communicate(self, *args, **kwargs):
            return ('\n'.join(json.dumps(v) for v in [
                {'type': 'item.completed', 'item': {'type': item_type}},
                {'type': 'turn.completed', 'usage': {'input_tokens': 3, 'output_tokens': 4}}]), '')
        def wait(self, **kwargs):
            return 0
    monkeypatch.setattr(driver.subprocess, 'Popen', Process)
    monkeypatch.setattr(backend, '_WindowsJob', lambda: None)
    monkeypatch.setattr(backend, '_stop_process_tree', lambda *args: None)
    record, final = driver.invoke('synthetic', tmp_path / 'call', 5)
    assert record['passed_transport'] is expected
    assert record['usage']['output_tokens'] == 4
    assert json.loads(final)['ready']
    assert record['requested_model'] == 'gpt-6.1-sol'

def test_explicit_executable_override(tmp_path, monkeypatch):
    monkeypatch.setenv('MOFI_CODEX_EXECUTABLE', 'C:/trusted/new/codex.exe')
    assert driver.command_for(tmp_path, tmp_path / 'final.txt')[0] == 'C:/trusted/new/codex.exe'


def test_only_known_advisories_are_ignored():
    events = [{'item': {'type': 'error', 'message': 'Code Mode is unavailable because code-mode host is disabled. Code mode will fail closed.'}},
              {'item': {'type': 'error', 'message': 'Authentication failed'}}]
    assert driver.inspect_events(events) == ['error']
    assert driver.inspect_events([{'item': {'type': 'command_execution'}}]) == ['command_execution']
