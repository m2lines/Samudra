window.BENCHMARK_DATA = {
  "lastUpdate": 1741306784126,
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
      },
      {
        "commit": {
          "author": {
            "email": "jesse@openathena.ai",
            "name": "Jesse Rusak",
            "username": "jder"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c5d1c2d724d8abc6fb958e63520e32674d399af2",
          "message": "GPU Tests & benchmarks (#96)\n\nRuns tests on GPU. Also moves existing CPU benchmarks to a GPU runner\nmachine and runs GPU benchmarks there, too. This lets us have 1\nbenchmark job and (I would expect) a more consistent performance\nenvironment for CPU tests, too.",
          "timestamp": "2025-03-06T16:49:15-05:00",
          "tree_id": "563daeb6dd63c785b2cc788225b3587c0cb391de",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/c5d1c2d724d8abc6fb958e63520e32674d399af2"
        },
        "date": 1741298353195,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cpu-train]",
            "value": 0.17420555904308752,
            "unit": "iter/sec",
            "range": "stddev: 0.38049027027638777",
            "extra": "mean: 5.740344943600007 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cpu-val]",
            "value": 0.4000370094079227,
            "unit": "iter/sec",
            "range": "stddev: 0.1758521402419627",
            "extra": "mean: 2.4997687126000074 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu]",
            "value": 0.1633972642911852,
            "unit": "iter/sec",
            "range": "stddev: 0.08581570713335311",
            "extra": "mean: 6.1200535048 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "jesse@openathena.ai",
            "name": "Jesse Rusak",
            "username": "jder"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3600a25e2cd26d87135c62942eab6335efd51465",
          "message": "fix benchmarks when run together (#99)\n\nSorry, I clearly [did not\ntest](https://github.com/suryadheeshjith/Ocean_Emulator/actions/runs/13708924698/job/38341018074)\nthe benchmarks well enough when run together with the new trainer\nbenchmark.\n\nToday we can't create more than one trainer in a process. It's on my\nplate to fix\n(https://github.com/suryadheeshjith/Ocean_Emulator/issues/87) but in the\nmeantime let's unbreak the benchmark by re-using the existing trainer.",
          "timestamp": "2025-03-06T18:45:37-05:00",
          "tree_id": "c622727bc25ffa4cb99a684875652a48412ca546",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/3600a25e2cd26d87135c62942eab6335efd51465"
        },
        "date": 1741306783167,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cpu-train]",
            "value": 0.08298613778205478,
            "unit": "iter/sec",
            "range": "stddev: 0.09666503770369714",
            "extra": "mean: 12.050205332199994 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cpu-val]",
            "value": 0.2280556547853698,
            "unit": "iter/sec",
            "range": "stddev: 0.08263070259120724",
            "extra": "mean: 4.384894559800022 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu]",
            "value": 0.20572186388893965,
            "unit": "iter/sec",
            "range": "stddev: 0.12199368978710297",
            "extra": "mean: 4.860932042400009 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu]",
            "value": 0.008765476842917017,
            "unit": "iter/sec",
            "range": "stddev: 0.3767539694413662",
            "extra": "mean: 114.0839246878 sec\nrounds: 5"
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
      },
      {
        "commit": {
          "author": {
            "email": "jesse@openathena.ai",
            "name": "Jesse Rusak",
            "username": "jder"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c5d1c2d724d8abc6fb958e63520e32674d399af2",
          "message": "GPU Tests & benchmarks (#96)\n\nRuns tests on GPU. Also moves existing CPU benchmarks to a GPU runner\nmachine and runs GPU benchmarks there, too. This lets us have 1\nbenchmark job and (I would expect) a more consistent performance\nenvironment for CPU tests, too.",
          "timestamp": "2025-03-06T16:49:15-05:00",
          "tree_id": "563daeb6dd63c785b2cc788225b3587c0cb391de",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/c5d1c2d724d8abc6fb958e63520e32674d399af2"
        },
        "date": 1741298355307,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cuda-train]",
            "value": 0.2007299897193038,
            "unit": "iter/sec",
            "range": "stddev: 0.3335808387688165",
            "extra": "mean: 4.981816625400006 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cuda-val]",
            "value": 0.4558057791548714,
            "unit": "iter/sec",
            "range": "stddev: 0.02179595023817908",
            "extra": "mean: 2.1939168956000117 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda]",
            "value": 0.1661040596205076,
            "unit": "iter/sec",
            "range": "stddev: 0.10025466902673769",
            "extra": "mean: 6.020322454999996 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}