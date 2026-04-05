from pathlib import Path

# Expose the real package under src/cvcap when running from the project root.
__path__ = [str(Path(__file__).resolve().parent.parent / "src" / "cvcap")]
