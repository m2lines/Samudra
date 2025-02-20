# Performance Profile Traces

Developers may optionally run [performance profile traces](https://docs.python.org/3/library/profile.html) in unit tests
via the `profile` pytest fixture. This will automatically save a timestamped copy of a cProfile trace to this directory.
The intention is that we'll be able to compare project performance improvements over time as well as build a culture of
testing performance.

Here's how you can inspect the results of a `cProfile` profile dump:

```bash
uvx snakeviz prof/<your-profile-dump.prof>
```