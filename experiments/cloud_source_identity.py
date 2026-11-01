"""Stdlib-only source fingerprint shared by smoke controllers and workers."""
import hashlib
from pathlib import Path


def source_identity(test_file):
    # Preserve the workers' existing fingerprint definition. Controllers do not
    # install ML packages, so validating a smoke receipt must not import them.
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for folder in ("adaptive_escher", "unbiased_escher", "vr_deep_cfr", "escher_poker", "experiments", "gcp"):
        for path in sorted((root/folder).rglob("*")):
            if path.suffix in (".py", ".sh"):
                digest.update(str(path.relative_to(root)).encode())
                digest.update(path.read_bytes())
    for name in ("requirements.txt", "setup.py", "pyproject.toml", test_file):
        path = root/name
        if path.is_file():
            digest.update(name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()
