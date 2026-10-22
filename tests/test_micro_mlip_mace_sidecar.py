"""Pure checks for the MACE sidecar capability gate helpers."""

from __future__ import annotations

from scripts.prepare_micro_mlip_mace_sidecar import contains_flag


def test_capability_probe_matches_complete_flag_tokens_only():
    help_text = "usage: mace_run_train [--lora LORA] [--freeze FREEZE] [--lora_rank LORA_RANK]"
    assert contains_flag(help_text, "--lora")
    assert contains_flag(help_text, "--freeze")
    assert contains_flag(help_text, "--lora_rank")
    assert not contains_flag(help_text, "--lora_alpha")
    assert not contains_flag("--lora_ranked", "--lora_rank")
