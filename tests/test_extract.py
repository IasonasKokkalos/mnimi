"""The extraction package: import discipline, schema, prompt, fake, cache, prefilter, resolver."""

from __future__ import annotations

import subprocess
import sys


def test_importing_mnimi_extract_never_imports_llama_cpp():
    # The [extract] extra is optional: `import mnimi` and `import mnimi.extract`
    # must work with the two core deps alone, and CI installs [dev] only.
    code = (
        "import sys, mnimi, mnimi.extract; "
        "assert 'llama_cpp' not in sys.modules, 'llama_cpp imported eagerly'; "
        "assert 'huggingface_hub' not in sys.modules, 'huggingface_hub imported eagerly'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
