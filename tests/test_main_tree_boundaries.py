from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_active_factory_and_evaluator_import_without_legacy_policy_package():
    script = """
import importlib.abc
import sys

class RejectPolicyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "oad_stress_test.policies" or fullname.startswith("oad_stress_test.policies."):
            raise ImportError(f"legacy policy import rejected: {fullname}")
        return None

sys.meta_path.insert(0, RejectPolicyImports())
from oad_stress_test.evaluators.streaming import CausalStreamingEvaluator
from oad_stress_test.utils.factory import make_classifier, make_datasets
assert CausalStreamingEvaluator is not None
assert make_classifier is not None
assert make_datasets is not None
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
