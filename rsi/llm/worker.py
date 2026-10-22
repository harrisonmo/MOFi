"""Single-GPU, offline QLoRA worker. Launch through rsi.llm, not directly."""

import importlib.metadata
import json
from pathlib import Path
import sys
import time

from .core import admit_rows, digest, files_manifest, read_rows, verify_bundle
from rsi.hpc.core import read_json, write_json


def encode(tokenizer, row, max_length):
    prompt = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=True,
                                          add_generation_prompt=True, enable_thinking=False)
    answer = tokenizer.encode(row["messages"][-1]["content"], add_special_tokens=False)
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if end is None or end == tokenizer.unk_token_id:
        raise ValueError("Qwen chat end token is missing")
    answer += [end]
    if not prompt or len(prompt + answer) > max_length:
        raise ValueError(f"{row['id']}: context exceeds max_length; truncation is forbidden")
    return {"input_ids": prompt + answer, "attention_mask": [1] * (len(prompt) + len(answer)),
            "labels": [-100] * len(prompt) + answer}


def checkpoint_identity(model_path, config):
    model_path = Path(model_path)
    source = model_path / "model_source.json"
    if not source.exists():
        source = model_path.parent / "model_source.json"
    record = read_json(source)
    if record.get("id") != config["model"] or record.get("sha") != config["revision"]:
        raise ValueError("Local model source does not match the pinned model and revision")
    if read_json(model_path / "config.json").get("model_type") != "qwen3":
        raise ValueError("Local checkpoint is not Qwen3")
    return {"model": config["model"], "revision": config["revision"],
            "source_sha256": digest(source), "files": files_manifest(model_path)}


def load_base(model_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with BF16 support is required")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    return model, tokenizer


def train(run, model_path):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import get_cosine_schedule_with_warmup, set_seed

    run = Path(run)
    config, _ = verify_bundle(run)
    identity = checkpoint_identity(model_path, config)
    write_json(run / "base_identity.json", identity)
    set_seed(config["seed"])
    started = time.monotonic()
    model, tokenizer = load_base(model_path)
    rows = admit_rows(read_rows(run / "data.jsonl"))
    data = {s: [encode(tokenizer, r, config["max_length"]) for r in rows if r["split"] == s]
            for s in ("train", "development")}
    def batch(item):
        return {k: torch.tensor([v], device="cuda") for k, v in item.items()}
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=config["rank"], lora_alpha=config["alpha"],
        lora_dropout=config["dropout"], target_modules="all-linear", bias="none", task_type="CAUSAL_LM"))
    model.config.use_cache = False
    environment = {"packages": {p: importlib.metadata.version(p) for p in
                    ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "safetensors")},
                   "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
                   "quantization": "nf4_double", "compute_dtype": "bfloat16", "enable_thinking": False,
                   "micro_batch_size": 1, "completion_only_loss": True,
                   "trainable_parameters": model.get_nb_trainable_parameters()[0],
                   "max_observed_length": max(len(x["input_ids"]) for v in data.values() for x in v)}
    write_json(run / "environment.json", environment)
    def development_loss():
        model.eval()
        total, tokens = 0.0, 0
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for item in data["development"]:
                loss = model(**batch(item), use_cache=False).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite development loss")
                count = sum(x != -100 for x in item["labels"][1:])
                total += loss.item() * count
                tokens += count
        return total / tokens
    initial = development_loss()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config["learning_rate"], weight_decay=0.01)
    schedule = get_cosine_schedule_with_warmup(optimizer, min(5, config["max_steps"] // 10), config["max_steps"])
    generator = torch.Generator().manual_seed(config["seed"])
    order, cursor = [], 0
    input_tokens, supervised_tokens = 0, 0
    torch.cuda.reset_peak_memory_stats()
    with (run / "training_log.jsonl").open("x", encoding="utf-8") as log:
        for step in range(1, config["max_steps"] + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for _ in range(config["gradient_accumulation"]):
                if cursor == len(order):
                    order = torch.randperm(len(data["train"]), generator=generator).tolist()
                    cursor = 0
                item = data["train"][order[cursor]]
                cursor += 1
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(**batch(item), use_cache=False).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                total += loss.item()
                (loss / config["gradient_accumulation"]).backward()
                input_tokens += len(item["input_ids"])
                supervised_tokens += sum(x != -100 for x in item["labels"][1:])
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite gradient norm")
            optimizer.step()
            schedule.step()
            event = {"step": step, "loss": total / config["gradient_accumulation"],
                     "gradient_norm": norm.item(), "seconds": time.monotonic() - started,
                     "input_tokens": input_tokens, "supervised_tokens": supervised_tokens}
            log.write(json.dumps(event) + "\n")
            log.flush()
            print(json.dumps(event), flush=True)
    final = development_loss()
    # Fixed final checkpoint: no adaptive search over validation or held-out tasks.
    model.save_pretrained(run / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(run / "adapter")
    summary = {"steps": config["max_steps"], "initial_development_loss": initial,
               "final_development_loss": final, "input_tokens": input_tokens,
               "supervised_tokens": supervised_tokens, "seconds": time.monotonic() - started,
               "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
               "adapter_files": files_manifest(run / "adapter"), "promotion_status": "not_evaluated"}
    write_json(run / "training_summary.json", summary)


if __name__ == "__main__":
    train(*sys.argv[1:])
