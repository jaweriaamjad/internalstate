#!/usr/bin/env python3
"""
Run all figure scripts in order (fig3, fig4, fig5, fig6, fig7) and report success/failure.
Figures are saved to figures_dir/ (from config). Run from repo root:

    python src/manuscript_figures/run_all_figures.py

Requires config/paths.json and config/paper_config.json to be set up; each script
may require prior pipeline outputs (VAE latents, GLM-HMM results, pupil data, etc.).
"""
import subprocess
import sys
from pathlib import Path

# Repo root (src/manuscript_figures -> src -> repo)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANUSCRIPT_FIGURES = REPO_ROOT / "src" / "manuscript_figures"


def run_script(name, argv=None):
    """Run a figure script; return True if exit code 0, else False."""
    script = MANUSCRIPT_FIGURES / name
    if not script.exists():
        print(f"  SKIP {name} (file not found)")
        return False
    cmd = [sys.executable, str(script)]
    if argv:
        cmd.extend(argv)
    try:
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), timeout=600)
        ok = result.returncode == 0
        if ok:
            print(f"  OK   {name}")
        else:
            print(f"  FAIL {name} (exit code {result.returncode})")
        return ok
    except subprocess.TimeoutExpired:
        print(f"  FAIL {name} (timeout)")
        return False
    except Exception as e:
        print(f"  FAIL {name} ({e})")
        return False


def main():
    print("Running all figure scripts (outputs go to figures_dir/)...")
    print()

    results = []

    # print("fig3.py (fig3b, fig3c)...")
    # results.append(("fig3", run_script("fig3.py")))
    # print()

    # print("fig4.py (fig4a–fig4d)...")
    # results.append(("fig4", run_script("fig4.py")))
    # print()

    # print("fig5.py (fig5a–fig5f)...")
    # results.append(("fig5", run_script("fig5.py")))
    # print()

    # print("fig6.py (fig6a–fig6c)...")
    # results.append(("fig6", run_script("fig6.py", ["3"])))  # default K=3
    # print()

    # print("fig7.py (fig7a–fig7c)...")
    # results.append(("fig7", run_script("fig7.py", ["3"])))  # default K=3
    # print()

    print("fig8.py (fig8a, fig8b)...")
    results.append(("fig8", run_script("fig8.py", ["3"])))  # K=3, kernel 0.4 to match fig7 cache
    print()

    print("fig9.py (fig9 — Bayesian model performance)...")
    results.append(("fig9", run_script("fig9.py")))
    print()

    # Summary
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print("=" * 50)
    print(f"Done: {passed}/{total} scripts succeeded.")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    print()
    print("Figures (if generated) are in: figures/ (config paths.json figures_dir)")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
