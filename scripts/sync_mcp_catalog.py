"""Synchronize the standalone MCP package's small, portable catalog assets."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "packages/mofi-tools"


def assets():
    source = (ROOT / "rsi/science/mof_databases.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    source = "".join(source.splitlines(keepends=True)[:main.lineno - 1])
    source = source.replace("import argparse\n", "")
    source = source.replace("ROOT = Path(__file__).resolve().parents[2]", "ROOT = Path(__file__).resolve().parent")
    source = source.replace('ROOT / "datasets/catalogs/mofdb.csv"', 'ROOT / "data/mofdb.csv"')
    source = source.replace('ROOT / "datasets/catalogs/mofdb.annotations.json"', 'ROOT / "data/mofdb.annotations.json"')
    source = source.rstrip() + "\n"
    result = {"src/mofi_tools/catalog.py": source.encode("utf-8")}
    for name in ("LICENSE", "NOTICE"):
        result[name] = (ROOT / name).read_bytes()
    for name in ("mofdb.csv", "mofdb.annotations.json"):
        result[f"src/mofi_tools/data/{name}"] = (ROOT / "datasets/catalogs" / name).read_bytes()
    result["src/mofi_tools/data/mof-database-catalog-v1.json"] = (ROOT / "tools/science/mof-database-catalog-v1.json").read_bytes()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, data in assets().items():
        target = PACKAGE / name
        if args.check:
            if not target.is_file() or target.read_bytes() != data:
                parser.error(f"MCP catalog asset differs from its source: {name}; run scripts/sync_mcp_catalog.py")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    print("MCP catalog assets are synchronized.")


if __name__ == "__main__":
    main()
