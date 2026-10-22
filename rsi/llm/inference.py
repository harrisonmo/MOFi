"""Fresh-process adapter reload and greedy generation; no reference answers or scorer."""

import json
from pathlib import Path
import sys

from rsi.hpc.core import read_json, write_json
from .core import files_manifest, verify_bundle
from .worker import checkpoint_identity, load_base


def generate(output):
    import torch
    from peft import PeftModel

    output = Path(output)
    request = read_json(output / "request.json")
    identity = checkpoint_identity(request["model_path"], request["config"])
    candidate = request["candidate"]
    if candidate:
        candidate = Path(candidate)
        config, manifest = verify_bundle(candidate)
        if config != request["config"] or manifest["status"] != "trained":
            raise ValueError("Invalid candidate")
        if identity != read_json(candidate / "base_identity.json"):
            raise ValueError("Base checkpoint differs from training")
        if files_manifest(candidate / "adapter") != read_json(candidate / "training_summary.json")["adapter_files"]:
            raise ValueError("Adapter changed before reload")
    model, tokenizer = load_base(request["model_path"])
    if candidate:
        model = PeftModel.from_pretrained(model, str(candidate / "adapter"), is_trainable=False)
    model.eval()
    messages = [{"role": "system", "content": "Return only JSON matching this schema. Do not execute tools.\n"
                 + json.dumps(request["schema"])}, {"role": "user", "content": request["prompt"]}]
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    if len(ids) + request["max_new_tokens"] > request["config"]["max_length"]:
        raise ValueError("Prompt plus generation budget exceeds context; truncation is forbidden")
    inputs = torch.tensor([ids], device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        result = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
            max_new_tokens=request["max_new_tokens"], do_sample=False, temperature=None, top_p=None,
            top_k=None, use_cache=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    completion = result[0, len(ids):].tolist()
    write_json(output / "generation.json", {"text": tokenizer.decode(completion, skip_special_tokens=True),
        "cap_hit": len(completion) >= request["max_new_tokens"] and completion[-1] != tokenizer.eos_token_id,
        "usage": {"input_tokens": len(ids), "output_tokens": len(completion), "total_tokens": len(ids) + len(completion)},
        "base_identity": identity, "enable_thinking": False, "do_sample": False})


if __name__ == "__main__":
    generate(sys.argv[1])
