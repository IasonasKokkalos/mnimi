"""mnimi_cli: the ``mnimi`` command line, a sibling package of the library.

Ships in the same wheel as ``mnimi`` and never changes it: the library is frozen at
``paper-v1`` (``evals/freeze.py``), and this package is a thin consumer of its public
``Memory`` API. ``mnimi export`` is the one command (MERGED-PLAN T2, LAUNCH §5).
"""
