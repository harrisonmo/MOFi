#!/usr/bin/env python3
"""Launch MACE 0.3.16 with a narrow torch 2.2 compatibility shim.

MACE 0.3.16 calls ``torch.compiler.is_compiling`` while the preserved remote
runtime is PyTorch 2.2.0, where that attribute is absent.  The shim only
provides the missing predicate and leaves the active 0.3.13 environment
untouched.
"""

from __future__ import annotations

import torch


if not hasattr(torch, "compiler"):
    class _CompilerNamespace:
        pass

    torch.compiler = _CompilerNamespace()  # type: ignore[attr-defined]
if not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False  # type: ignore[attr-defined]

from mace.cli.run_train import main  # noqa: E402


if __name__ == "__main__":
    main()
