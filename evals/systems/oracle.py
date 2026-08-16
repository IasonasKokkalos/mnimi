"""The evidence-availability bound: only the annotated evidence reaches the reader."""

from __future__ import annotations

from .full_history import FullHistorySystem


class OracleSystem(FullHistorySystem):
    """Reads only the question's annotated evidence sessions, whole.

    This is **the evidence-availability bound**, the paper's own reference point
    (§5.5). It bounds what evidence reaches the reader, not how well it is
    presented — a focused retriever handing the reader fewer, cleaner tokens can
    match or exceed it, so it is deliberately never called a ceiling (SPEC,
    CHANGELOG #14). ``full_history`` cannot play the reference role either: at
    the pinned 32K reader context it
    truncated 20/20 smoke-slice questions, feeding ~27,210 tokens and dropping
    ~1.84M — the reader saw ~23% of each history, and a "ceiling" that
    truncates measures the context window, not achievable accuracy.

    Two design points, both deliberate:

    * **Whole sessions, not retrieval within them.** Perfect retrieval returns
      the evidence sessions; retrieving inside them would measure this
      retriever against a pre-cleaned corpus, which is a different (also
      interesting) number and not this bound.
    * **It subclasses ``FullHistorySystem`` instead of copying its renderer.**
      Structured formatting is worth up to 10 points *at oracle retrieval*
      (LongMemEval Fig 6, §5.5), so a divergence between the oracle's format
      and the baseline's would land inside the comparison. Sharing the method
      makes identical formatting a property of the code rather than a claim
      someone has to re-verify after every edit.

    The session filtering happens in the runner, driven by ``evidence_only``;
    this class never sees a distractor session.
    """

    name = "oracle"
    evidence_only = True
