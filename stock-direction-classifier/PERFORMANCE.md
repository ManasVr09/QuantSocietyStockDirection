# Dashboard performance

The dashboard loads only the saved artifacts required by the active research view. The overview projects its five needed prediction columns; unused fields and the unused ledger are not loaded into its charts. No saved artifacts are rewritten.

Artifact caches are limited to 24 entries and keyed by absolute path, nanosecond modification time, size, and requested columns. Editing a saved CSV/Parquet invalidates it on the next full run. Cached DataFrames are copied per call. Classification imports scikit-learn only when requested. Stock selection runs in a fragment; optional sections and comparison tabs execute only when visible. All existing report information remains available.

Regression tests cover each view, saved-artifact invalidation, projected loads, and deferred comparisons. Model/feature/backtest source and report artifacts are unchanged.
