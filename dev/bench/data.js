window.BENCHMARK_DATA = {
  "lastUpdate": 1741131981015,
  "repoUrl": "https://github.com/suryadheeshjith/Ocean_Emulator",
  "entries": {
    "Python Benchmark with pytest-benchmark": [
      {
        "commit": {
          "author": {
            "email": "alex@openathena.ai",
            "name": "Alex Merose",
            "username": "alxmrs"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5f0685bc93824c6020b68d0500e4950215fd61ec",
          "message": "Trying to prevent a git error during benchmark publishing. (#93)\n\nSee:\nhttps://docs.astral.sh/uv/concepts/projects/sync/#automatic-lock-and-sync",
          "timestamp": "2025-03-04T15:43:00-08:00",
          "tree_id": "5271b88b4abdabae9f8f4e7c615ed6c83428668c",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/5f0685bc93824c6020b68d0500e4950215fd61ec"
        },
        "date": 1741131980252,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cpu-train]",
            "value": 0.10129957381271672,
            "unit": "iter/sec",
            "range": "stddev: 0.15782607495643836",
            "extra": "mean: 9.871709843999998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cpu-val]",
            "value": 0.27875893873158974,
            "unit": "iter/sec",
            "range": "stddev: 0.027921421471974912",
            "extra": "mean: 3.587328910599979 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu]",
            "value": 0.23596931152918932,
            "unit": "iter/sec",
            "range": "stddev: 0.20254437427843017",
            "extra": "mean: 4.237839206799992 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}