# Performance Profile Traces

Developers may optionally run [performance profile traces](https://docs.python.org/3/library/profile.html) in unit tests
via the `pytest-profiling` plugin. This will automatically save a timestamped copy of a cProfile trace to this
directory. The intention is that we'll be able to compare project performance improvements over time, to build a culture
of testing performance.

To capture performance for all relevant tests, please run:

```bash
pytest -k profile --profile
# with uv
uv run pytest -k profile --profile
```

This will write out `.prof` traces to this directory. To generate image outputs (if you have [`dot` and
`graphviz` installed on your machine](https://www.graphviz.org/download/)), please run:

```bash
pytest -k profile --profile-svg
# with uv
uv run pytest -k profile --profile-svg
```

To inspect raw `.prof` traces, you can use `snakeviz`:

```bash
uvx snakeviz prof/<your-profile-dump.prof>
# For example:
uvx snakeviz prof/test_profile__train_loader.prof
```