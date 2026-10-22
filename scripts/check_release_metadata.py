"""Check that Python and MCP release identities agree before publishing."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import tomllib


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    package = root / "packages/mofi-tools"
    project = tomllib.loads((package / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    server = json.loads((root / "server.json").read_text(encoding="utf-8"))
    module = ast.parse((package / "src/mofi_tools/__init__.py").read_text(encoding="utf-8"))
    versions = [ast.literal_eval(node.value) for node in module.body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "__version__"
                        for target in node.targets)]
    if versions != [project["version"]] or server["version"] != project["version"]:
        raise ValueError("Python metadata, module and MCP versions must agree")
    if server["name"] != "io.github.harrisonmo/mofi":
        raise ValueError("Unexpected MCP namespace")
    marker = f"<!-- mcp-name: {server['name']} -->"
    if marker not in (package / "README.md").read_text(encoding="utf-8"):
        raise ValueError("Package README must contain the MCP ownership marker")
    distributions = server["packages"]
    if len(distributions) != 1:
        raise ValueError("Exactly one PyPI distribution is expected")
    distribution = distributions[0]
    if (distribution["registryType"], distribution["identifier"], distribution["version"]) != (
        "pypi", project["name"], project["version"]
    ):
        raise ValueError("MCP distribution must match Python project metadata")
    if distribution["transport"] != {"type": "stdio"} or distribution["packageArguments"] != [
        {"type": "positional", "value": "serve"}
    ]:
        raise ValueError("The registry command must start the stdio server")
    print(f"Release metadata verified: {project['name']} {project['version']} ({server['name']})")


if __name__ == "__main__":
    main()
