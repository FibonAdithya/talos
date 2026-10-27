"""Check that a built wheel holds every file the job containers and the Modal app need.

Usage: python scripts/check_wheel.py dist/talos_tig-X-py3-none-any.whl

Exits 1 and names each missing entry. It reads JOB_MODULES and DEPLOYED_FILES from the repo's
source with `ast`, so it does not import talos.
"""
import ast
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_FILES = ("talos/data/rust_rules.md", "talos/data/evidence_template.md",
              "talos/data/submission_readme_template.md")


def literal_tuple(source_file: str, name: str) -> tuple[str, ...]:
    tree = ast.parse((ROOT / source_file).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise SystemExit(f"{name} not found in {source_file}")


def required() -> list[str]:
    job_modules = literal_tuple("talos/c3_jobdir.py", "JOB_MODULES")
    deployed = literal_tuple("talos/deploy_digest.py", "DEPLOYED_FILES")
    names = [*DATA_FILES, "modal_app/talos_bench.py",
             *(f"talos/{m}.py" for m in job_modules), *deployed]
    return list(dict.fromkeys(names))


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    with zipfile.ZipFile(argv[0]) as z:
        present = set(z.namelist())
    missing = [n for n in required() if n not in present]
    for n in missing:
        print(f"missing from {argv[0]}: {n}", file=sys.stderr)
    if missing:
        return 1
    print(f"{argv[0]}: all {len(required())} required files present")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
