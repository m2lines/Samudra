window.BENCHMARK_DATA = {
  "lastUpdate": 1744842288311,
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
      },
      {
        "commit": {
          "author": {
            "email": "41594351+suryadheeshjith@users.noreply.github.com",
            "name": "Surya Dheeshjith",
            "username": "suryadheeshjith"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6f54e2a17d4fffe6740ef6e59a0b382def3304aa",
          "message": "Update max steps in constants.py (#216)",
          "timestamp": "2025-04-16T13:04:39-04:00",
          "tree_id": "89dd7069844cff48073196b14ee6aca4c96fef80",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/6f54e2a17d4fffe6740ef6e59a0b382def3304aa"
        },
        "date": 1744825755101,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-mock-train_default.test.yaml]",
            "value": 0.19387310271997207,
            "unit": "iter/sec",
            "range": "stddev: 0.23382894602171014",
            "extra": "mean: 5.158013081600018 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_LAZY-cpu-mock-train_default.test.yaml]",
            "value": 0.47247804457225173,
            "unit": "iter/sec",
            "range": "stddev: 0.07833463645319574",
            "extra": "mean: 2.116500462799979 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-mock-train_default.test.yaml]",
            "value": 0.05030697984866226,
            "unit": "iter/sec",
            "range": "stddev: 1.0894000371153418",
            "extra": "mean: 19.877957353199996 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-mock-train_default.test.yaml]",
            "value": 0.12841156429536216,
            "unit": "iter/sec",
            "range": "stddev: 0.26655604691610746",
            "extra": "mean: 7.7874606191999876 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-mock-train_default.test.yaml]",
            "value": 0.006936637916861929,
            "unit": "iter/sec",
            "range": "stddev: 0.5419443780709129",
            "extra": "mean: 144.16205833220005 sec\nrounds: 5"
          }
        ]
      },
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
          "id": "47cef09247ac73131582226ffa3dd7e94d65c391",
          "message": "Adding types and type-specific variable names to InferenceDataset  (#214)\n\nThis addresses a number of flaky mypy errors in datasets.py.",
          "timestamp": "2025-04-16T14:32:41-07:00",
          "tree_id": "5ba5ed14034fae6fc258253593ac8ec509a9e26c",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/47cef09247ac73131582226ffa3dd7e94d65c391"
        },
        "date": 1744841611929,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-mock-train_default.test.yaml]",
            "value": 0.19568162641321044,
            "unit": "iter/sec",
            "range": "stddev: 0.44295041232527155",
            "extra": "mean: 5.1103418258000035 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_LAZY-cpu-mock-train_default.test.yaml]",
            "value": 0.5123310988434614,
            "unit": "iter/sec",
            "range": "stddev: 0.007609669835326069",
            "extra": "mean: 1.9518627744000014 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-mock-train_default.test.yaml]",
            "value": 0.055264641202127915,
            "unit": "iter/sec",
            "range": "stddev: 1.329404683133495",
            "extra": "mean: 18.094752417599988 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-mock-train_default.test.yaml]",
            "value": 0.13727298623077874,
            "unit": "iter/sec",
            "range": "stddev: 0.08361939282318899",
            "extra": "mean: 7.2847544695999655 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-mock-train_default.test.yaml]",
            "value": 0.007690349436882988,
            "unit": "iter/sec",
            "range": "stddev: 0.1334348204246942",
            "extra": "mean: 130.0331029438 sec\nrounds: 5"
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
          "id": "ef402bd05adab5e866d9c2383b2a76f4672314fb",
          "message": "Use stricter notion of equality for data loaders (#187)\n\nI was trying to figure out why the lazy loader's losses were quite\ndifferent from eager and torch loaders. I was not successful but given\nthis test currently passes it seems like it was worth checking it in.\nThoughts?",
          "timestamp": "2025-04-16T14:44:16-07:00",
          "tree_id": "47e8225632082bcecab821e4ed3c888d9e6579e1",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/ef402bd05adab5e866d9c2383b2a76f4672314fb"
        },
        "date": 1744842287018,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-mock-train_default.test.yaml]",
            "value": 0.19235076693154302,
            "unit": "iter/sec",
            "range": "stddev: 0.34118133425533403",
            "extra": "mean: 5.198835523000002 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_LAZY-cpu-mock-train_default.test.yaml]",
            "value": 0.5152874325894424,
            "unit": "iter/sec",
            "range": "stddev: 0.09057245329488896",
            "extra": "mean: 1.9406644461999805 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-mock-train_default.test.yaml]",
            "value": 0.05430085072908913,
            "unit": "iter/sec",
            "range": "stddev: 0.7171642232109837",
            "extra": "mean: 18.415917735599987 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-mock-train_default.test.yaml]",
            "value": 0.13625283886972686,
            "unit": "iter/sec",
            "range": "stddev: 0.13234812615418312",
            "extra": "mean: 7.339296621600033 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-mock-train_default.test.yaml]",
            "value": 0.007832507456924676,
            "unit": "iter/sec",
            "range": "stddev: 0.5676131109063465",
            "extra": "mean: 127.67303516780001 sec\nrounds: 5"
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
      },
      {
        "commit": {
          "author": {
            "email": "41594351+suryadheeshjith@users.noreply.github.com",
            "name": "Surya Dheeshjith",
            "username": "suryadheeshjith"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6f54e2a17d4fffe6740ef6e59a0b382def3304aa",
          "message": "Update max steps in constants.py (#216)",
          "timestamp": "2025-04-16T13:04:39-04:00",
          "tree_id": "89dd7069844cff48073196b14ee6aca4c96fef80",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/6f54e2a17d4fffe6740ef6e59a0b382def3304aa"
        },
        "date": 1744825757559,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-mock-train_default.test.yaml]",
            "value": 0.19162669132517887,
            "unit": "iter/sec",
            "range": "stddev: 0.10918733338717698",
            "extra": "mean: 5.2184797070000055 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_LAZY-cuda-mock-train_default.test.yaml]",
            "value": 0.48946088304018087,
            "unit": "iter/sec",
            "range": "stddev: 0.0789668281311864",
            "extra": "mean: 2.0430641848000506 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-mock-train_default.test.yaml]",
            "value": 0.05086815807555744,
            "unit": "iter/sec",
            "range": "stddev: 1.7424044651393846",
            "extra": "mean: 19.658663451400024 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-mock-train_default.test.yaml]",
            "value": 0.12003941579973994,
            "unit": "iter/sec",
            "range": "stddev: 0.08738583942767349",
            "extra": "mean: 8.330597023799964 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-mock-train_default.test.yaml]",
            "value": 0.00980102664094515,
            "unit": "iter/sec",
            "range": "stddev: 0.18967606305125392",
            "extra": "mean: 102.03012772379998 sec\nrounds: 5"
          }
        ]
      },
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
          "id": "47cef09247ac73131582226ffa3dd7e94d65c391",
          "message": "Adding types and type-specific variable names to InferenceDataset  (#214)\n\nThis addresses a number of flaky mypy errors in datasets.py.",
          "timestamp": "2025-04-16T14:32:41-07:00",
          "tree_id": "5ba5ed14034fae6fc258253593ac8ec509a9e26c",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/47cef09247ac73131582226ffa3dd7e94d65c391"
        },
        "date": 1744841614038,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-mock-train_default.test.yaml]",
            "value": 0.19602406742600897,
            "unit": "iter/sec",
            "range": "stddev: 0.5641500895020005",
            "extra": "mean: 5.101414398400129 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_LAZY-cuda-mock-train_default.test.yaml]",
            "value": 0.5399224065501704,
            "unit": "iter/sec",
            "range": "stddev: 0.07523057848745841",
            "extra": "mean: 1.852117985599989 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-mock-train_default.test.yaml]",
            "value": 0.0558080488740754,
            "unit": "iter/sec",
            "range": "stddev: 0.4908260241764432",
            "extra": "mean: 17.918562289400008 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-mock-train_default.test.yaml]",
            "value": 0.1310450438404766,
            "unit": "iter/sec",
            "range": "stddev: 0.07947100916054976",
            "extra": "mean: 7.630963909000002 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-mock-train_default.test.yaml]",
            "value": 0.01092586878193464,
            "unit": "iter/sec",
            "range": "stddev: 0.6072026158263316",
            "extra": "mean: 91.525902421 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}