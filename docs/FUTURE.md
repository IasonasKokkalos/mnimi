# FUTURE SPECS AND IDEAS

* have pettern recognition memory.recognition
* be able to handle multiple aganets working at the same time and have a queueing strat(sync feature, multiple agemts could also mean mult users)
* would be cool for it to hanlde 1M entities per user, so a DB graph will be be applied in the future.
Although for this don't hardcode sqlite-vec calls into the write-path logic.
* Multi-device sync on a raw SQLite file risks corruption.

## Notes

* (when switching to other sotarge than sqlite) Swap to ANN and a near-duplicate can occasionally miss the candidate set entirely, silently degrading your merge quality. Worth a comment/test now, fix later.