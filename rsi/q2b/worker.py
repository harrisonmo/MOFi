"""Read input only after the controller has installed process limits."""
import json
import sys
import traceback


def main():
    request = json.loads(sys.stdin.read())
    try:
        if request["action"] == "source":
            from .fixtures import source
            value = source(request["index"])
        elif request["action"] == "calibrate":
            from .calibrate import check_pair
            value = check_pair(request["pair"])
        elif request["action"] == "execute":
            from .runtime import execute
            value = execute(request["program"], request["A"], request["B"], request.get("tool"))
        elif request["action"] == "score":
            from .scoring import score
            value = score(request["pair"], request["output"], request["report"])
        elif request["action"] == "validate_tool":
            from .runtime import execute, validate_program
            from .reference import verify
            program, pair = request["program"], request["pair"]
            validate_program(program, top=False)
            value = execute(program, pair["A"], pair["B"], _nested=True)
            if set(value["evidence"]) != {"alignment", "site_delta", "edge_delta", "components", "neighborhood"}:
                raise ValueError("Incorrect generic output contract")
            checked = verify(pair["A"], pair["B"], value["evidence"], "A")
            if checked["verdict"] != pair["expected"]:
                raise ValueError("Development equivalence evidence failed")
            value = checked
        elif request["action"] == "small":
            from .checks import check
            value = check(request["name"])
        elif request["action"] == "fixture":
            from .fixtures import pairs
            value = pairs(request["record"], request["index"])
        elif request["action"] == "resource_probe":
            if request["probe"] == "memory":
                try:
                    bytearray(3 * 1024 ** 3)
                except MemoryError:
                    value = {"memory_limit_enforced": True}
                else:
                    raise AssertionError("Memory limit not enforced")
            else:
                import time
                time.sleep(5)
                raise AssertionError("Timeout not enforced")
        else:
            raise ValueError("Unknown controller action")
        print(json.dumps({"ok": True, "value": value}, allow_nan=False))
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))


if __name__ == "__main__":
    main()
