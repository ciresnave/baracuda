"""Tests for scripts/bump-version.py. Run: python -m unittest discover -s scripts -p 'test_*.py'

The blind-replace test is the point: it fails for a naive `alpha.OLD -> alpha.NEW` replace.
"""
import importlib.util
import pathlib
import shutil
import subprocess  # nosec B404 - test runs only `git ls-files` with fixed arguments, no shell
import unittest

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bump_version", HERE / "bump-version.py")
bv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bv)


def blind(text, old, new):
    return text.replace(f"alpha.{old}", f"alpha.{new}")


HISTORY = (
    "Fuel bisected an all-zero output at alpha.78 (pinned `=0.0.1-alpha.78`)\r\n"
    "[`0.1.0-alpha.78`] was tagged\r\n"
)


class BumpTests(unittest.TestCase):
    def test_cargo_version_lines_bump_exactly(self):
        src = 'version = "0.1.0-alpha.78"\nbaracuda = { version = "0.1.0-alpha.78", path = "x" }\n'
        self.assertEqual(
            bv.bump_cargo(src, "78", "81"),
            'version = "0.1.0-alpha.81"\nbaracuda = { version = "0.1.0-alpha.81", path = "x" }\n',
        )

    def test_whole_token_only(self):
        self.assertEqual(bv.bump_cargo('version = "0.1.0-alpha.780"\n', "78", "81"), 'version = "0.1.0-alpha.780"\n')
        self.assertEqual(bv.bump_cargo('version = "0.0.1-alpha.78"\n', "78", "81"), 'version = "0.0.1-alpha.78"\n')

    def test_current_statements_bump(self):
        self.assertEqual(
            bv.bump_file("ARCHITECTURE.md", "The workspace is at **v0.1.0-alpha.78**. x", "78", "81"),
            "The workspace is at **v0.1.0-alpha.81**. x",
        )
        self.assertEqual(
            bv.bump_file("README.md", "> `main` is at `0.1.0-alpha.78`; y", "78", "81"),
            "> `main` is at `0.1.0-alpha.81`; y",
        )

    def test_history_is_untouched_where_a_blind_replace_rewrites_it(self):
        for path in ("ARCHITECTURE.md", "OP-MATRIX.md", "README.md", "docs/gap-001-emitter-exoneration.md"):
            self.assertNotEqual(blind(HISTORY, "78", "81"), HISTORY)  # the blind replace corrupts it ...
            self.assertEqual(bv.bump_file(path, HISTORY, "78", "81"), HISTORY)  # ... the script does not

    def test_unrelated_file_untouched(self):
        self.assertEqual(bv.bump_file("src/lib.rs", "// alpha.78 all-zero", "78", "81"), "// alpha.78 all-zero")

    def test_repo_history_sentinels(self):
        """Historical text repaired in alpha.81 must stay as written (git ls-files-relative paths)."""
        root = HERE.parent
        sentinels = {
            "docs/gap-001-emitter-exoneration.md": ['`baracuda-kernelgen = "=0.0.1-alpha.77"`'],
            "crates/baracuda-kernels-bench/tests/max_nan_propagation.rs": ["shape as the alpha.78 all-zero"],
            "OP-MATRIX.md": ["The body reflects the alpha.77 line"],
        }
        git = shutil.which("git")
        self.assertIsNotNone(git, "git not found on PATH")
        # Absolute git path, fixed argv, no shell.
        tracked = set(subprocess.run([git, "ls-files"], cwd=root, capture_output=True, text=True).stdout.split())  # nosec B603
        for rel, needles in sentinels.items():
            if rel not in tracked:
                continue
            text = (root / rel).read_bytes().decode("latin-1")
            for needle in needles:
                self.assertIn(needle, text, f"{rel}: historical text was rewritten")


if __name__ == "__main__":
    unittest.main()
