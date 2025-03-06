window.BENCHMARK_DATA = {
  "lastUpdate": 1741272127123,
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
      },
      {
        "commit": {
          "author": {
            "name": "Jesse Rusak",
            "username": "jder",
            "email": "me@jesserusak.com"
          },
          "committer": {
            "name": "Jesse Rusak",
            "username": "jder",
            "email": "me@jesserusak.com"
          },
          "id": "e5b4747750ce6e204e6c64848da4303f48625cb6",
          "message": "Use GPU runners for benchmarks too",
          "timestamp": "2025-03-06T14:31:42Z",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/e5b4747750ce6e204e6c64848da4303f48625cb6"
        },
        "date": 1741272123691,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cpu-train]",
            "value": 0.19450538555785601,
            "unit": "iter/sec",
            "range": "stddev: 0.3578038321581784",
            "extra": "mean: 5.141245817599986 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cpu-val]",
            "value": 0.42210662317939274,
            "unit": "iter/sec",
            "range": "stddev: 0.14352713139068227",
            "extra": "mean: 2.369069673599995 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu]",
            "value": 0.16333163414175053,
            "unit": "iter/sec",
            "range": "stddev: 0.0747579601187738",
            "extra": "mean: 6.122512673400001 sec\nrounds: 5"
          }
        ]
      }
    ],
    "Python Benchmark with pytest-benchmark (GPU)": [
      {
        "commit": {
          "author": {
            "name": "Jesse Rusak",
            "username": "jder",
            "email": "me@jesserusak.com"
          },
          "committer": {
            "name": "Jesse Rusak",
            "username": "jder",
            "email": "me@jesserusak.com"
          },
          "id": "e5b4747750ce6e204e6c64848da4303f48625cb6",
          "message": "Use GPU runners for benchmarks too",
          "timestamp": "2025-03-06T14:31:42Z",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/e5b4747750ce6e204e6c64848da4303f48625cb6"
        },
        "date": 1741272126186,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cuda-train]",
            "value": 0.20077627560966077,
            "unit": "iter/sec",
            "range": "stddev: 0.2893057014535929",
            "extra": "mean: 4.980668143999992 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cuda-val]",
            "value": 0.45316694607284674,
            "unit": "iter/sec",
            "range": "stddev: 0.021339989101592623",
            "extra": "mean: 2.2066922768000152 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda]",
            "value": 0.16381264858865272,
            "unit": "iter/sec",
            "range": "stddev: 0.10637843211555174",
            "extra": "mean: 6.104534714599993 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}