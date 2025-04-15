window.BENCHMARK_DATA = {
  "lastUpdate": 1744741568292,
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
          "id": "ba1d64318fd3545b2e2d0ae6ff6efb6acbf8ca25",
          "message": "Using a perf counter to measure time for performance. (#191)",
          "timestamp": "2025-04-15T10:45:17-07:00",
          "tree_id": "3b3ba9fdf2f6e5f0d4f09d84060cbfb8d9e8ae7e",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/ba1d64318fd3545b2e2d0ae6ff6efb6acbf8ca25"
        },
        "date": 1744741565348,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-mock-train_default.test.yaml]",
            "value": 0.1867045272420116,
            "unit": "iter/sec",
            "range": "stddev: 0.38865046499367734",
            "extra": "mean: 5.356056517599984 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_LAZY-cpu-mock-train_default.test.yaml]",
            "value": 0.5278080334027114,
            "unit": "iter/sec",
            "range": "stddev: 0.011959208410435145",
            "extra": "mean: 1.8946282298000028 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-mock-train_default.test.yaml]",
            "value": 0.05415969245338399,
            "unit": "iter/sec",
            "range": "stddev: 1.1851745629302939",
            "extra": "mean: 18.463915777599993 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-mock-train_default.test.yaml]",
            "value": 0.1396868029869146,
            "unit": "iter/sec",
            "range": "stddev: 0.07271597757370207",
            "extra": "mean: 7.158872410399977 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-mock-train_default.test.yaml]",
            "value": 0.007876417639524295,
            "unit": "iter/sec",
            "range": "stddev: 0.28217871496990393",
            "extra": "mean: 126.96127170580003 sec\nrounds: 5"
          }
        ]
      }
    ],
    "Python Benchmark with pytest-benchmark (GPU)": [
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
          "id": "ba1d64318fd3545b2e2d0ae6ff6efb6acbf8ca25",
          "message": "Using a perf counter to measure time for performance. (#191)",
          "timestamp": "2025-04-15T10:45:17-07:00",
          "tree_id": "3b3ba9fdf2f6e5f0d4f09d84060cbfb8d9e8ae7e",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/ba1d64318fd3545b2e2d0ae6ff6efb6acbf8ca25"
        },
        "date": 1744741567380,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-mock-train_default.test.yaml]",
            "value": 0.20338158685571298,
            "unit": "iter/sec",
            "range": "stddev: 0.3223892422200483",
            "extra": "mean: 4.916865953600018 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_LAZY-cuda-mock-train_default.test.yaml]",
            "value": 0.5443005253788635,
            "unit": "iter/sec",
            "range": "stddev: 0.08561328857005351",
            "extra": "mean: 1.8372203468000408 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-mock-train_default.test.yaml]",
            "value": 0.051953912634401356,
            "unit": "iter/sec",
            "range": "stddev: 1.5616442555274344",
            "extra": "mean: 19.24782849440003 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-mock-train_default.test.yaml]",
            "value": 0.13046192235055382,
            "unit": "iter/sec",
            "range": "stddev: 0.08087941876395714",
            "extra": "mean: 7.665071784800011 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-mock-train_default.test.yaml]",
            "value": 0.011140746427156763,
            "unit": "iter/sec",
            "range": "stddev: 0.9186787717073944",
            "extra": "mean: 89.76059248259999 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}