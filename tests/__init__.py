# Regular-package marker so `python -m tests.smoke_test` works from the repo root:
# descript-audio-codec (a `scorers`-extra transitive dep) installs a top-level
# `tests/` package into site-packages, which would otherwise shadow this
# __init__-less namespace directory under `-m`.
