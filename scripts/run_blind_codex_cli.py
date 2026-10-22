"""Tool-disabled, fresh Codex CLI file exchange for the frozen Blind-G1 study."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'gpt-6.1-sol'
EFFORT = 'high'
CAMPAIGN = 'runs/blind-g1-campaign-01'
DISABLED = ('shell_tool', 'apps', 'plugins', 'remote_plugin', 'multi_agent', 'memories',
            'browser_use', 'browser_use_external', 'image_generation', 'view_image',
            'skill_search', 'skill_mcp_dependency_install', 'hooks', 'goals', 'code_mode_host',
            'sleep_tool', 'tool_suggest')
WRAPPERS = {'functions.wait', 'functions.request_user_input', 'functions.request_user_input_async', 'functions.exec'}


def advisory(item):
    if item.get('type') != 'error':
        return False
    message = item.get('message', '')
    return (message.startswith('Under-development features enabled: skip_host_skill_discovery.') or
            message.startswith('Code Mode is unavailable because code-mode host is disabled.'))


def inspect_events(events):
    allowed = {'agent_message', 'reasoning', 'plan'}
    return [e['item'].get('type') for e in events if isinstance(e.get('item'), dict)
            and e['item'].get('type') not in allowed and not advisory(e['item'])]


def smoke_admitted(directory):
    record = read(directory / 'call.json')
    events = [json.loads(line) for line in (directory / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
    record['unexpected_item_types'] = inspect_events(events)
    record['passed_transport'] = (record.get('exit_code') == 0 and not record['unexpected_item_types']
        and not record.get('timed_out') and not record.get('malformed_event')
        and not record['error_events'] and record['workspace_unchanged'])
    if not record['passed_transport']:
        return False
    reply = read(directory / 'final.txt')
    exposed = set(reply.get('tools_listing', []))
    fail_closed = any(e.get('item', {}).get('message', '').startswith(
        'Code Mode is unavailable because code-mode host is disabled.') for e in events)
    record['transport_gate'] = {'listed_wrappers': sorted(exposed), 'code_host_fail_closed': fail_closed,
        'executable_io_tools_disabled': exposed <= WRAPPERS and (not exposed or fail_closed),
        'tool_events': record['unexpected_item_types']}
    save(directory / 'admission.json', record)
    return reply.get('ready') is True and record['transport_gate']['executable_io_tools_disabled']


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def controller(action, *args):
    command = ['wsl', '-d', 'Ubuntu', '--', 'python3', '-m', 'rsi.blind_campaign', action,
               '--campaign', CAMPAIGN, *args]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=14400)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace')[-2000:])
    return json.loads(result.stdout)


def command_for(workspace, final_path):
    executable = os.environ.get('MOFI_CODEX_EXECUTABLE') or shutil.which('codex') or 'codex'
    command = [executable, 'exec', '--model', MODEL,
               '--ephemeral', '--ignore-user-config', '--ignore-rules', '--skip-git-repo-check',
               '--sandbox', 'read-only', '--json', '--color', 'never',
               '--cd', str(workspace), '--output-last-message', str(final_path),
               '-c', 'model_reasoning_effort="high"', '-c', 'approval_policy="never"',
               '-c', 'project_doc_max_bytes=0', '-c', 'web_search="disabled"',
               '-c', 'agents.enabled=false', '-c', 'features.skip_host_skill_discovery=true']
    for feature in DISABLED:
        command += ['--disable', feature]
    return command + ['-']


def invoke(prompt, output, timeout):
    from rsi.backends.codex_cli import _WindowsJob, _stop_process_tree
    output.mkdir(parents=True, exist_ok=False)
    workspace = output / 'empty-workspace'
    workspace.mkdir()
    final = output / 'final.txt'
    command = command_for(workspace, final)
    environment = {k: v for k, v in os.environ.items()
                   if not k.upper().endswith('API_KEY') and k.upper() not in
                   {'OPENAI_BASE_URL', 'OPENAI_ACCESS_TOKEN', 'CODEX_THREAD_ID', 'CODEX_PARENT_THREAD_ID'}}
    environment['PYTHONUTF8'] = '1'
    (output / 'prompt.txt').write_text(prompt, encoding='utf-8')
    record = {'requested_model': MODEL, 'reasoning_effort': EFFORT, 'argv': command,
              'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
              'timeout_seconds': timeout, 'authentication': 'existing-supported-cli-login'}
    save(output / 'call.json', record)
    job = _WindowsJob() if os.name == 'nt' else None
    process = None
    started = time.monotonic()
    stdout, stderr = '', ''
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=workspace, env=environment, text=True, encoding='utf-8', errors='replace',
            start_new_session=os.name != 'nt', creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
        if job:
            job.assign(process)
        stdout, stderr = process.communicate(prompt, timeout=timeout)
        record['exit_code'] = process.returncode
    except subprocess.TimeoutExpired:
        record['timed_out'] = True
        _stop_process_tree(process, job)
        stdout, stderr = process.communicate(timeout=5)
    finally:
        if process:
            _stop_process_tree(process, job)
            process.wait(timeout=5)
        elif job:
            job.close()
    (output / 'events.jsonl').write_text(stdout, encoding='utf-8')
    (output / 'stderr.txt').write_text(stderr, encoding='utf-8')
    events = []
    for line in stdout.splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            record['malformed_event'] = True
    unexpected = inspect_events(events)
    record.update(seconds=time.monotonic() - started, unexpected_item_types=unexpected,
                  workspace_unchanged=not any(workspace.iterdir()),
                  usage=next((e.get('usage') for e in reversed(events) if e.get('type') == 'turn.completed'), None),
                  actual_model_evidence=[e for e in events if 'model' in e],
                  error_events=[e for e in events if e.get('type') in ('error', 'turn.failed')])
    record['passed_transport'] = (record.get('exit_code') == 0 and not unexpected and not record.get('timed_out')
                                  and not record['error_events'] and not record.get('malformed_event')
                                  and record['workspace_unchanged'] and final.is_file())
    save(output / 'call.json', record)
    return record, final.read_text(encoding='utf-8') if final.exists() else ''


def run_slot(arm):
    state = controller('status')['providers']['chatgpt']['arms'][arm]
    generation = len(state['slots']) + 1
    output = ROOT / 'runs/blind-codex-sol-high' / f'{arm}-{generation}'
    packet = controller('export', '--provider', 'chatgpt', '--arm', arm)
    prompt = ('Act only as a candidate author. All execution and evaluation is performed by an external controller. '
              'Do not use any tools. Return exactly the JSON requested by the packet, without fences. '
              'session.model must exactly match the registered model in response_contract. '
              'hypothesis must be at most 2000 characters. No previous conversation is available.\n\n'
              + (ROOT / 'templates/blind_g1/START_HERE.md').read_text(encoding='utf-8-sig')
              + '\n\npacket.json:\n' + json.dumps(packet))
    print(f'{arm}-{generation}: authoring with {MODEL} / high', flush=True)
    record, text = invoke(prompt, output, 1680)
    response = output / 'response.json'
    # A failed transport consumes its slot; never repair or regenerate a model reply.
    response.write_text(text if record['passed_transport'] else '{}', encoding='utf-8')
    feedback = controller('submit', '--provider', 'chatgpt', '--arm', arm,
                          '--submission', response.relative_to(ROOT).as_posix())
    record['feedback'] = feedback
    save(output / 'call.json', record)
    print(f'{arm}-{generation}: {json.dumps(feedback)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--smoke-name', default='smoke-01')
    args = parser.parse_args()
    if args.smoke:
        record, final = invoke('Synthetic transport check only. Do not use tools or access files. '
                              'Return JSON with ready=true and tools_listing containing the names of tools available to you. '
                              'Do not invent tool names; an empty list is valid.', ROOT / 'runs/blind-codex-sol-high' / args.smoke_name, 120)
        print(json.dumps({'record': record, 'response': final}, indent=2))
    if args.run:
        if (ROOT / 'runs/blind-codex-sol-high/PAUSE_REQUEST.json').exists():
            print('Pause requested. Preserve the ledger; explicit operator resume is required.', flush=True)
            return
        if not smoke_admitted(ROOT / 'runs/blind-codex-sol-high' / args.smoke_name):
            raise RuntimeError('Requested-model transport failed; no registration or formal slot was created')
        providers = controller('status')['providers']
        if 'chatgpt' not in providers:
            controller('register', '--provider', 'chatgpt', '--model', MODEL, '--transport', 'cli-file-exchange')
        elif providers['chatgpt']['model'] != MODEL:
            raise ValueError('Registered model differs')
        for arm in ('independent', 'rsi'):
            while len(controller('status')['providers']['chatgpt']['arms'][arm]['slots']) < 3:
                run_slot(arm)
        if not controller('status')['providers']['chatgpt'].get('confirmation'):
            print('Running nine fresh-seed confirmation fits', flush=True)
            result = controller('confirm', '--provider', 'chatgpt')
            save(ROOT / 'runs/blind-codex-sol-high/confirmation.json', result)
            print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
