# FUTURE SPECS AND IDEAS

* have pettern recognition memory.recognition
* be able to handle multiple aganets working at the same time and have a queueing strat(sync feature, multiple agemts could also mean mult users)
* would be cool for it to hanlde 1M entities per user, so a DB graph will be be applied in the future.
Although for this don't hardcode sqlite-vec calls into the write-path logic.
* Multi-device sync on a raw SQLite file risks corruption.
*  Replayable write path: persist an ordered op-log so a user's history can be re-run deterministically under different config (new decay half-life, threshold, etc.) and diffed against the original — enables ablations without live re-benchmarking.
* Optional purge/GC: hard-delete records below salience floor for >N days, for users who don't need audit history. Off by default — deletion breaks export()'s point-in-time guarantee.
* MemoryConfig.embedder - swappable at init time only, not mid-corpus.
* custom embedder model in config init,Matryoshka dim truncation, Hot-swap handling / re-embed workflows, API-backed embedder options.
* Consider a MCP wrapper

## Notes

* (when switching to other sotarge than sqlite) Swap to ANN and a near-duplicate can occasionally miss the candidate set entirely, silently degrading my merge quality. Worth a comment/test now, fix later.
* use cross encoder if needed in the future( after cosine KNN narrows candidates, as a reranker — not a replacement for the embedding pass, cause its heavy for an entire DB).