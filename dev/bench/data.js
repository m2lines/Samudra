window.BENCHMARK_DATA = {
  "lastUpdate": 1745685476695,
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
          "id": "dcc4fa780eeb21762ddae07405a9571a68a6f6dd",
          "message": "Removed the Lazy Loader from the codebase. (#213)\n\nFixes #207.",
          "timestamp": "2025-04-16T14:57:46-07:00",
          "tree_id": "790c3ea7d82a10d8bfbde947e144cde5c44b2c02",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/dcc4fa780eeb21762ddae07405a9571a68a6f6dd"
        },
        "date": 1744843006387,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-mock-train_default.test.yaml]",
            "value": 0.20011474602750523,
            "unit": "iter/sec",
            "range": "stddev: 0.2704052946793023",
            "extra": "mean: 4.997132994200001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-mock-train_default.test.yaml]",
            "value": 0.05789661991650419,
            "unit": "iter/sec",
            "range": "stddev: 1.0773368435918922",
            "extra": "mean: 17.2721654812 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-mock-train_default.test.yaml]",
            "value": 0.13949711653631966,
            "unit": "iter/sec",
            "range": "stddev: 0.10632463594171095",
            "extra": "mean: 7.168606956399981 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-mock-train_default.test.yaml]",
            "value": 0.007838094989608309,
            "unit": "iter/sec",
            "range": "stddev: 0.15793428798501996",
            "extra": "mean: 127.5820210556 sec\nrounds: 5"
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
          "id": "85f6f6ada579d695c859c5f033866a73f81f8ec7",
          "message": "Vectorizing the Torch Loader (#212)\n\nFixes #205.\n\n---------\n\nCo-authored-by: Jesse Rusak <jesse@openathena.ai>",
          "timestamp": "2025-04-16T15:11:06-07:00",
          "tree_id": "f1c6f092a292472a6268d8b37ebbd41d5d9f6712",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/85f6f6ada579d695c859c5f033866a73f81f8ec7"
        },
        "date": 1744843900601,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-mock-train_default.test.yaml]",
            "value": 0.18124662961425278,
            "unit": "iter/sec",
            "range": "stddev: 0.43888035255670027",
            "extra": "mean: 5.517343975599988 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-mock-train_default.test.yaml]",
            "value": 0.05263275031526606,
            "unit": "iter/sec",
            "range": "stddev: 0.7537985453500919",
            "extra": "mean: 18.999577145599993 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-mock-train_default.test.yaml]",
            "value": 0.13632990498399145,
            "unit": "iter/sec",
            "range": "stddev: 0.10803230973929084",
            "extra": "mean: 7.3351477808000025 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-mock-train_default.test.yaml]",
            "value": 0.007717954766841503,
            "unit": "iter/sec",
            "range": "stddev: 0.18517636039844462",
            "extra": "mean: 129.5680047642 sec\nrounds: 5"
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
          "id": "3e4ab9c74cb22e2ecb2670a4717dcd5a3536f539",
          "message": "Add missing Multiton typing (#217)\n\nNoticed while reviewing #214.",
          "timestamp": "2025-04-16T18:51:23-04:00",
          "tree_id": "6f22f225167c8f97c247f460854fe97ab05a5527",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/3e4ab9c74cb22e2ecb2670a4717dcd5a3536f539"
        },
        "date": 1744846507839,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-mock-train_default.test.yaml]",
            "value": 0.1590125736223508,
            "unit": "iter/sec",
            "range": "stddev: 0.511721468762918",
            "extra": "mean: 6.2888108608000035 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-mock-train_default.test.yaml]",
            "value": 0.05130323196343788,
            "unit": "iter/sec",
            "range": "stddev: 1.479031051551947",
            "extra": "mean: 19.491949370999997 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-mock-train_default.test.yaml]",
            "value": 0.13228085116127192,
            "unit": "iter/sec",
            "range": "stddev: 0.11120953394585689",
            "extra": "mean: 7.5596731592000195 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-mock-train_default.test.yaml]",
            "value": 0.007363709259193663,
            "unit": "iter/sec",
            "range": "stddev: 0.23283552297793952",
            "extra": "mean: 135.80112478659993 sec\nrounds: 5"
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
          "id": "0ac9469629781b19963f3eefea74a95a4fa3e8ed",
          "message": "Introducing `DataSource` to train and eval scripts. (#211)\n\nThis makes progress towards #190. Specifically, this PR argues that the\nthree Xarray datasets (data, means, stds) are actually part of the same\nconcern, one that is fundamentally related to normalization. I\nanticipate that this affordance will make it easier to modify core\naspects to the data loader in the future. In particular, the addition of\nthe `DataSource.filter()` method should provide a single entry point to\nadd data compaction support -- see #215 (initially prototyped in #175).\nIn addition, this change should make #208 trivial to implement.\n\nFixes #218.\n\n---------\n\nCo-authored-by: Jesse Rusak <jesse@openathena.ai>",
          "timestamp": "2025-04-18T14:00:47-07:00",
          "tree_id": "387a1a355aad9cb9c67d2f63ec35b7c33a8cf5bc",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/0ac9469629781b19963f3eefea74a95a4fa3e8ed"
        },
        "date": 1745012614640,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-mock-train_default.test.yaml]",
            "value": 0.1684152124770462,
            "unit": "iter/sec",
            "range": "stddev: 0.23706041890408053",
            "extra": "mean: 5.937705895399995 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-mock-train_default.test.yaml]",
            "value": 0.05149057677845876,
            "unit": "iter/sec",
            "range": "stddev: 0.9274375337423029",
            "extra": "mean: 19.4210292944 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-mock-train_default.test.yaml]",
            "value": 0.13255156751016528,
            "unit": "iter/sec",
            "range": "stddev: 0.06633189442387215",
            "extra": "mean: 7.544233680399975 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-mock-train_default.test.yaml]",
            "value": 0.0073484602579212185,
            "unit": "iter/sec",
            "range": "stddev: 1.8171510364044732",
            "extra": "mean: 136.08292960720001 sec\nrounds: 5"
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
          "id": "499db56063fe946f79b37da5b8241f484d02e0f8",
          "message": "pydantic-settings for config (#210)\n\nConfiguration via pydantic-settings, yaml files and command line-overrides.\n\nAn example invocation would be `uv run python -m ocean_emulators.train configs/train_cm4.yaml\n--experiment.cluster_data_dir=/tmp/data/dir --epochs=2`\n\nThis allows `!include` in YAML files (see\nconfigs/train_cm4.yaml and configs/data/surya.yaml for an example) and\nallows overriding bits on the command line either with values eg\n`--epochs 10` or with the contents of some file eg `--data\n@configs/data/jder.yaml`. It also has JSON schemas exported so that\nVSCode's YAML server will help you a bit, for example:\n\n<img width=\"666\" alt=\"Screenshot 2025-04-15 at 3 40 17 PM\"\nsrc=\"https://github.com/user-attachments/assets/0d512d7f-739f-47af-a9c0-b9696caaee44\"\n/>\n\nUnfortunately this doesn't know about the include syntax so it's a bit\nnoisy, but it still seems helpful.",
          "timestamp": "2025-04-18T18:31:07-04:00",
          "tree_id": "2834ea6b1d662d4e84b74181b6c4630506d58ff6",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/499db56063fe946f79b37da5b8241f484d02e0f8"
        },
        "date": 1745018117148,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-mock-train_default.test.yaml]",
            "value": 0.16763378155788883,
            "unit": "iter/sec",
            "range": "stddev: 0.128415190378753",
            "extra": "mean: 5.965384725600018 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-mock-train_default.test.yaml]",
            "value": 0.04859398795666355,
            "unit": "iter/sec",
            "range": "stddev: 2.144350145286177",
            "extra": "mean: 20.578677364200008 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-mock-train_default.test.yaml]",
            "value": 0.12147719039025452,
            "unit": "iter/sec",
            "range": "stddev: 0.1569591188533516",
            "extra": "mean: 8.23199809599996 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-mock-train_default.test.yaml]",
            "value": 0.006862715594810418,
            "unit": "iter/sec",
            "range": "stddev: 1.2965314937020067",
            "extra": "mean: 145.7149121488 sec\nrounds: 5"
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
          "id": "e01e28c1a414b86e56e2c3ccd3e3aa14a20727cc",
          "message": "Normalize prefill and fill value configuration (#219)\n\nAdd normalize pre fill boolean and fill value configuration for testing\npost fill configuration and multiple fill values\n\n- [x] Add tests for pre-fill and post-fill configuration with multiple\nfill values",
          "timestamp": "2025-04-22T10:39:07-04:00",
          "tree_id": "565a8e25bad0d8d56e8a64b5f00a859a9b72b738",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/e01e28c1a414b86e56e2c3ccd3e3aa14a20727cc"
        },
        "date": 1745335292579,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-mock-train_default.test.yaml]",
            "value": 0.17465823760534638,
            "unit": "iter/sec",
            "range": "stddev: 0.6380995106415528",
            "extra": "mean: 5.7254671392000205 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-mock-train_default.test.yaml]",
            "value": 0.046065139713128124,
            "unit": "iter/sec",
            "range": "stddev: 0.6583459544397046",
            "extra": "mean: 21.708389602800004 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-mock-train_default.test.yaml]",
            "value": 0.1380147638595732,
            "unit": "iter/sec",
            "range": "stddev: 0.0772877819134211",
            "extra": "mean: 7.24560164459997 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-mock-train_default.test.yaml]",
            "value": 0.007187277769725334,
            "unit": "iter/sec",
            "range": "stddev: 2.3328860996006187",
            "extra": "mean: 139.1347366888 sec\nrounds: 5"
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
          "id": "f2cabb570ca994ac0c334b5149ff02d8e532c117",
          "message": "Enable upgrade lints (#220)\n\nOn #210, @alxmrs was leaving comments about using `Union` rather than\nmodern`|`, and there are a variety of related things like dict/list vs\nDict/List and so on. This also turns on warnings about using `\"\".format` rather than\nf-strings, older import paths, inheriting from `object`, using\n`open(foo, \"r\")` rather than `open(foo)` and others. Full list is here\nhttps://docs.astral.sh/ruff/rules/#pyupgrade-up.\n\nAll these fixes were automated (and therefore should be safe) except for\none I will call out below.",
          "timestamp": "2025-04-22T13:28:25-04:00",
          "tree_id": "8d977694a8c8210a21e9ec9450eb0c3586d732e4",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/f2cabb570ca994ac0c334b5149ff02d8e532c117"
        },
        "date": 1745345477194,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-mock-train_default.test.yaml]",
            "value": 0.17634263985379522,
            "unit": "iter/sec",
            "range": "stddev: 0.27245885643788564",
            "extra": "mean: 5.6707782123999895 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-mock-train_default.test.yaml]",
            "value": 0.04538474323424702,
            "unit": "iter/sec",
            "range": "stddev: 0.8760868297855096",
            "extra": "mean: 22.0338362352 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-mock-train_default.test.yaml]",
            "value": 0.13783425336897048,
            "unit": "iter/sec",
            "range": "stddev: 0.12652846457352068",
            "extra": "mean: 7.255090629200026 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-mock-train_default.test.yaml]",
            "value": 0.007197720118009039,
            "unit": "iter/sec",
            "range": "stddev: 0.16293502752484837",
            "extra": "mean: 138.93288202439996 sec\nrounds: 5"
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
          "id": "baf745e6863bbf4fa749a2457a0661fc6a582fc2",
          "message": "Add aggregate data benchmarks to W&B Logs. (#192)\n\nI also added fine-grained debug logging statements to the core of each\ndata loader.",
          "timestamp": "2025-04-22T17:14:02-07:00",
          "tree_id": "0f5c8f401186d1df55079976645857dd47fb2a52",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/baf745e6863bbf4fa749a2457a0661fc6a582fc2"
        },
        "date": 1745369926699,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-mock-train_default.test.yaml]",
            "value": 0.17015797402976113,
            "unit": "iter/sec",
            "range": "stddev: 0.18939259381299503",
            "extra": "mean: 5.876891786600003 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-mock-train_default.test.yaml]",
            "value": 0.0421667156470555,
            "unit": "iter/sec",
            "range": "stddev: 1.4011167099784603",
            "extra": "mean: 23.715387472200007 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-mock-train_default.test.yaml]",
            "value": 0.13161326905764142,
            "unit": "iter/sec",
            "range": "stddev: 0.10080356952505631",
            "extra": "mean: 7.598018096200008 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-mock-train_default.test.yaml]",
            "value": 0.006856879818703837,
            "unit": "iter/sec",
            "range": "stddev: 0.2944949565517047",
            "extra": "mean: 145.83892768139998 sec\nrounds: 5"
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
          "id": "212d9da3b3954bee74342a10420a25a13cd6bd8b",
          "message": "Use dask for inference (#227)\n\nFixes memory issues with the torch loader, fixes #208. \n\nAfter this, the torch and eager loaders have the same peak memory usage\nduring inference. The issue was that the non-dask version was using\nabout 3x the memory of the dask version and getting OOM killed. This is\ndue to a combination of the InferenceDataset not being written with\neager evaluation in mind (eg slicing repeatedly) and due to what look\nlike some pretty bad performance bugs in xarray. (It was allocating a\nton of memory to lazily track reorderings of the data rather than\neagerly shuffling the arrays -- kinda looks like 6x more than the\nunderlying arrays but I could have misread and don't think it's worth\ndigging into this more now.) (And to add insult to injury, the shuffle\nin question was the identity transform…)",
          "timestamp": "2025-04-24T11:57:17-04:00",
          "tree_id": "00f0babf0b2c9662ad5320466b7459ea845ab3d5",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/212d9da3b3954bee74342a10420a25a13cd6bd8b"
        },
        "date": 1745512818344,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-mock-train_default.test.yaml]",
            "value": 0.17323228920122472,
            "unit": "iter/sec",
            "range": "stddev: 0.08372543566206345",
            "extra": "mean: 5.772595886200008 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-mock-train_default.test.yaml]",
            "value": 0.043826142763783764,
            "unit": "iter/sec",
            "range": "stddev: 0.8143787014379099",
            "extra": "mean: 22.8174312622 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-mock-train_default.test.yaml]",
            "value": 0.13627922727373712,
            "unit": "iter/sec",
            "range": "stddev: 0.09291817240686903",
            "extra": "mean: 7.337875478199999 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-mock-train_default.test.yaml]",
            "value": 0.007134197394690626,
            "unit": "iter/sec",
            "range": "stddev: 2.9826944397623745",
            "extra": "mean: 140.1699370898 sec\nrounds: 5"
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
          "id": "b929596c1d476649860758ddc80745891dbe94fd",
          "message": "Log data wait time and data load times separately (#231)\n\nWe now log how much time we spend waiting for data (stalling due to\ngetting ahead of the data loaders, which is what we were logging before)\nand also how much time the relevant data loader spent loading that data,\nwhich is not itself important for overall train time since it's in a\nseparate process and async, but is much more sensitive of a measure for\nfiguring out what is happening with load times.\n\nexample wandb run here:\nhttps://wandb.ai/m2lines/ocean-emulators/runs/o3gv8d0k\n\noutput looks like this now:\n\n```\n2025-04-25 15:18:01,323 - INFO - logging - Training Epoch: [1]  [   0/1412]  eta: 7:19:24  lr: 0.000200  iter_time: 18.672(18.672)  data_wait_time: 9.300(9.300)  data_load_time: 8.747(8.747)  loss: 1.4612 (1.4612)  max cpu mem: 1605  max gpu mem: 30424\n2025-04-25 15:18:01,970 - INFO - logging - Training Epoch: [1]  [   1/1412]  eta: 3:47:08  lr: 0.000200  iter_time: 0.645(0.645)  data_wait_time: 0.000(0.000)  data_load_time: 8.745(8.745)  loss: 1.4612 (1.4779)  max cpu mem: 1757  max gpu mem: 30424\n2025-04-25 15:18:02,617 - INFO - logging - Training Epoch: [1]  [   2/1412]  eta: 2:36:22  lr: 0.000200  iter_time: 0.646(0.646)  data_wait_time: 0.000(0.000)  data_load_time: 8.736(8.736)  loss: 1.4612 (1.3976)  max cpu mem: 1758  max gpu mem: 30424\n2025-04-25 15:18:03,265 - INFO - logging - Training Epoch: [1]  [   3/1412]  eta: 2:00:59  lr: 0.000200  iter_time: 0.647(0.647)  data_wait_time: 0.000(0.000)  data_load_time: 8.748(8.748)  loss: 1.2369 (1.3021)  max cpu mem: 1759  max gpu mem: 30424\n2025-04-25 15:18:03,961 - INFO - logging - Training Epoch: [1]  [   4/1412]  eta: 1:39:59  lr: 0.000200  iter_time: 0.695(0.695)  data_wait_time: 0.000(0.000)  data_load_time: 8.854(8.854)  loss: 1.2369 (1.2135)  max cpu mem: 1759  max gpu mem: 30424\n2025-04-25 15:18:04,612 - INFO - logging - Training Epoch: [1]  [   5/1412]  eta: 1:25:48  lr: 0.000200  iter_time: 0.649(0.649)  data_wait_time: 0.000(0.000)  data_load_time: 8.747(8.747)  loss: 1.0158 (1.1252)  max cpu mem: 1759  max gpu mem: 30424\n2025-04-25 15:18:05,263 - INFO - logging - Training Epoch: [1]  [   6/1412]  eta: 1:15:40  lr: 0.000200  iter_time: 0.650(0.650)  data_wait_time: 0.000(0.000)  data_load_time: 8.748(8.748)  loss: 1.0158 (1.0428)  max cpu mem: 1759  max gpu mem: 30424\n```",
          "timestamp": "2025-04-25T17:15:23-04:00",
          "tree_id": "76a1d196a08819dafd2fb5e423d98e6610fab5bd",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/b929596c1d476649860758ddc80745891dbe94fd"
        },
        "date": 1745618589137,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-mock-train_default.test.yaml]",
            "value": 0.17254092649381358,
            "unit": "iter/sec",
            "range": "stddev: 0.24300413516430022",
            "extra": "mean: 5.795726384 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-mock-train_default.test.yaml]",
            "value": 0.042723367519929376,
            "unit": "iter/sec",
            "range": "stddev: 0.5374417243200751",
            "extra": "mean: 23.4063946278 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-mock-train_default.test.yaml]",
            "value": 0.1338976466289525,
            "unit": "iter/sec",
            "range": "stddev: 0.10498516873847676",
            "extra": "mean: 7.468391156799998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-mock-train_default.test.yaml]",
            "value": 0.006888229390347656,
            "unit": "iter/sec",
            "range": "stddev: 0.31821358339536526",
            "extra": "mean: 145.1751884746 sec\nrounds: 5"
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
          "id": "a7a42c83b3ed6c3b7feb97d9000a8cd33b0092d3",
          "message": "revert vectorizing torch loader (#234)\n\nSorry, re-opening because I merged to the wrong branch. We should merge\n#231 before this one.\n\nThis loader appears to be dramatically faster *without* vectorization.\nSee:\nhttps://wandb.ai/m2lines/ocean-emulators/reports/Torch-Loader-with-and-Without-Vectorization--VmlldzoxMjQ3NzA2MA\n\nFixes #230 and reverts #212\n\nFrom comparing profiles, it appears the vectorized version is hitting\nsome kind of performance issue in xarray's indexing:\n\n\n[profile-novec](https://github.com/user-attachments/assets/5b006496-898c-4d5d-9e2c-eebaac0dfd43)\n\n[profile-vec](https://github.com/user-attachments/assets/065bbfc6-1d75-4e6c-8876-b016fa0a393c)",
          "timestamp": "2025-04-25T17:36:11-04:00",
          "tree_id": "aa75b471dedbccc3cd41f7642b776987e5c1d087",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/a7a42c83b3ed6c3b7feb97d9000a8cd33b0092d3"
        },
        "date": 1745619524933,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-mock-train_default.test.yaml]",
            "value": 0.18103898808457713,
            "unit": "iter/sec",
            "range": "stddev: 0.49174760046621374",
            "extra": "mean: 5.523672058600016 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-mock-train_default.test.yaml]",
            "value": 0.044692758076124495,
            "unit": "iter/sec",
            "range": "stddev: 0.7922695129588113",
            "extra": "mean: 22.37498966380001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-mock-train_default.test.yaml]",
            "value": 0.1357457371555006,
            "unit": "iter/sec",
            "range": "stddev: 0.0684189158614772",
            "extra": "mean: 7.366713835400015 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-mock-train_default.test.yaml]",
            "value": 0.007055286457514787,
            "unit": "iter/sec",
            "range": "stddev: 0.2865479359353482",
            "extra": "mean: 141.73768932299998 sec\nrounds: 5"
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
          "id": "c2fbe9bf215d428679dcfaf05f23d68ae70388fe",
          "message": "Finetune, Restarts and Preemptible Runs (#226)\n\nCloses #223 , #33 , #160",
          "timestamp": "2025-04-26T11:53:05-04:00",
          "tree_id": "e930269fb388568c98bea5bc2c516239803ce7a2",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/c2fbe9bf215d428679dcfaf05f23d68ae70388fe"
        },
        "date": 1745685473810,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-mock-train_default.test.yaml]",
            "value": 0.1793757440392398,
            "unit": "iter/sec",
            "range": "stddev: 0.24190517636522843",
            "extra": "mean: 5.5748897676000295 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-mock-train_default.test.yaml]",
            "value": 0.046734822291440214,
            "unit": "iter/sec",
            "range": "stddev: 0.5262448120087531",
            "extra": "mean: 21.39732111880003 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-mock-train_default.test.yaml]",
            "value": 0.134152268359711,
            "unit": "iter/sec",
            "range": "stddev: 0.11292971931492653",
            "extra": "mean: 7.454216109999993 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-mock-train_default.test.yaml]",
            "value": 0.007250170452408171,
            "unit": "iter/sec",
            "range": "stddev: 0.17736657756227542",
            "extra": "mean: 137.92779170699998 sec\nrounds: 5"
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
        "date": 1744842289381,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-mock-train_default.test.yaml]",
            "value": 0.19794353420320093,
            "unit": "iter/sec",
            "range": "stddev: 0.4227218242111661",
            "extra": "mean: 5.0519457683999915 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_LAZY-cuda-mock-train_default.test.yaml]",
            "value": 0.5384696279736079,
            "unit": "iter/sec",
            "range": "stddev: 0.09026794570660579",
            "extra": "mean: 1.857114956999976 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-mock-train_default.test.yaml]",
            "value": 0.05553368815266424,
            "unit": "iter/sec",
            "range": "stddev: 0.8786947570319963",
            "extra": "mean: 18.00708782839997 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-mock-train_default.test.yaml]",
            "value": 0.13076027836615073,
            "unit": "iter/sec",
            "range": "stddev: 0.07920218762702606",
            "extra": "mean: 7.647582373599971 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-mock-train_default.test.yaml]",
            "value": 0.010579509789224515,
            "unit": "iter/sec",
            "range": "stddev: 2.0603103517742234",
            "extra": "mean: 94.52233798379999 sec\nrounds: 5"
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
          "id": "dcc4fa780eeb21762ddae07405a9571a68a6f6dd",
          "message": "Removed the Lazy Loader from the codebase. (#213)\n\nFixes #207.",
          "timestamp": "2025-04-16T14:57:46-07:00",
          "tree_id": "790c3ea7d82a10d8bfbde947e144cde5c44b2c02",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/dcc4fa780eeb21762ddae07405a9571a68a6f6dd"
        },
        "date": 1744843008780,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-mock-train_default.test.yaml]",
            "value": 0.20446385722920238,
            "unit": "iter/sec",
            "range": "stddev: 0.3901619059730472",
            "extra": "mean: 4.890839943799984 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-mock-train_default.test.yaml]",
            "value": 0.05639689149006624,
            "unit": "iter/sec",
            "range": "stddev: 1.1542145203368046",
            "extra": "mean: 17.73147373159991 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-mock-train_default.test.yaml]",
            "value": 0.13244609717670677,
            "unit": "iter/sec",
            "range": "stddev: 0.09207346499483049",
            "extra": "mean: 7.550241353399952 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-mock-train_default.test.yaml]",
            "value": 0.011019075496156218,
            "unit": "iter/sec",
            "range": "stddev: 0.40887927752069914",
            "extra": "mean: 90.7517150916 sec\nrounds: 5"
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
          "id": "85f6f6ada579d695c859c5f033866a73f81f8ec7",
          "message": "Vectorizing the Torch Loader (#212)\n\nFixes #205.\n\n---------\n\nCo-authored-by: Jesse Rusak <jesse@openathena.ai>",
          "timestamp": "2025-04-16T15:11:06-07:00",
          "tree_id": "f1c6f092a292472a6268d8b37ebbd41d5d9f6712",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/85f6f6ada579d695c859c5f033866a73f81f8ec7"
        },
        "date": 1744843902859,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-mock-train_default.test.yaml]",
            "value": 0.1789094861475476,
            "unit": "iter/sec",
            "range": "stddev: 0.4646268461279602",
            "extra": "mean: 5.589418546400021 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-mock-train_default.test.yaml]",
            "value": 0.05673692551644761,
            "unit": "iter/sec",
            "range": "stddev: 0.45695104923921664",
            "extra": "mean: 17.62520599939994 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-mock-train_default.test.yaml]",
            "value": 0.1308310652317681,
            "unit": "iter/sec",
            "range": "stddev: 0.11053640462482019",
            "extra": "mean: 7.643444607200081 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-mock-train_default.test.yaml]",
            "value": 0.01025165149149829,
            "unit": "iter/sec",
            "range": "stddev: 5.591888457612751",
            "extra": "mean: 97.54525900819995 sec\nrounds: 5"
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
          "id": "3e4ab9c74cb22e2ecb2670a4717dcd5a3536f539",
          "message": "Add missing Multiton typing (#217)\n\nNoticed while reviewing #214.",
          "timestamp": "2025-04-16T18:51:23-04:00",
          "tree_id": "6f22f225167c8f97c247f460854fe97ab05a5527",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/3e4ab9c74cb22e2ecb2670a4717dcd5a3536f539"
        },
        "date": 1744846510084,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-mock-train_default.test.yaml]",
            "value": 0.16728412420165711,
            "unit": "iter/sec",
            "range": "stddev: 0.5676932850266345",
            "extra": "mean: 5.977853575600056 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-mock-train_default.test.yaml]",
            "value": 0.04965872010590991,
            "unit": "iter/sec",
            "range": "stddev: 0.47559197609353715",
            "extra": "mean: 20.137450136999995 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-mock-train_default.test.yaml]",
            "value": 0.1155716050622202,
            "unit": "iter/sec",
            "range": "stddev: 0.10838009836882366",
            "extra": "mean: 8.652644388399995 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-mock-train_default.test.yaml]",
            "value": 0.009704147695114233,
            "unit": "iter/sec",
            "range": "stddev: 2.136301796837951",
            "extra": "mean: 103.04872013680006 sec\nrounds: 5"
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
          "id": "0ac9469629781b19963f3eefea74a95a4fa3e8ed",
          "message": "Introducing `DataSource` to train and eval scripts. (#211)\n\nThis makes progress towards #190. Specifically, this PR argues that the\nthree Xarray datasets (data, means, stds) are actually part of the same\nconcern, one that is fundamentally related to normalization. I\nanticipate that this affordance will make it easier to modify core\naspects to the data loader in the future. In particular, the addition of\nthe `DataSource.filter()` method should provide a single entry point to\nadd data compaction support -- see #215 (initially prototyped in #175).\nIn addition, this change should make #208 trivial to implement.\n\nFixes #218.\n\n---------\n\nCo-authored-by: Jesse Rusak <jesse@openathena.ai>",
          "timestamp": "2025-04-18T14:00:47-07:00",
          "tree_id": "387a1a355aad9cb9c67d2f63ec35b7c33a8cf5bc",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/0ac9469629781b19963f3eefea74a95a4fa3e8ed"
        },
        "date": 1745012616840,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-mock-train_default.test.yaml]",
            "value": 0.17233357942867203,
            "unit": "iter/sec",
            "range": "stddev: 0.34521507991512745",
            "extra": "mean: 5.802699643999995 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-mock-train_default.test.yaml]",
            "value": 0.05263641488806646,
            "unit": "iter/sec",
            "range": "stddev: 1.1898318175205047",
            "extra": "mean: 18.998254385800056 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-mock-train_default.test.yaml]",
            "value": 0.12426730387133196,
            "unit": "iter/sec",
            "range": "stddev: 0.14665182680747468",
            "extra": "mean: 8.047169036800005 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-mock-train_default.test.yaml]",
            "value": 0.009975770663640455,
            "unit": "iter/sec",
            "range": "stddev: 2.458952421201442",
            "extra": "mean: 100.24288185019986 sec\nrounds: 5"
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
          "id": "499db56063fe946f79b37da5b8241f484d02e0f8",
          "message": "pydantic-settings for config (#210)\n\nConfiguration via pydantic-settings, yaml files and command line-overrides.\n\nAn example invocation would be `uv run python -m ocean_emulators.train configs/train_cm4.yaml\n--experiment.cluster_data_dir=/tmp/data/dir --epochs=2`\n\nThis allows `!include` in YAML files (see\nconfigs/train_cm4.yaml and configs/data/surya.yaml for an example) and\nallows overriding bits on the command line either with values eg\n`--epochs 10` or with the contents of some file eg `--data\n@configs/data/jder.yaml`. It also has JSON schemas exported so that\nVSCode's YAML server will help you a bit, for example:\n\n<img width=\"666\" alt=\"Screenshot 2025-04-15 at 3 40 17 PM\"\nsrc=\"https://github.com/user-attachments/assets/0d512d7f-739f-47af-a9c0-b9696caaee44\"\n/>\n\nUnfortunately this doesn't know about the include syntax so it's a bit\nnoisy, but it still seems helpful.",
          "timestamp": "2025-04-18T18:31:07-04:00",
          "tree_id": "2834ea6b1d662d4e84b74181b6c4630506d58ff6",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/499db56063fe946f79b37da5b8241f484d02e0f8"
        },
        "date": 1745018119322,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-mock-train_default.test.yaml]",
            "value": 0.16793746513389923,
            "unit": "iter/sec",
            "range": "stddev: 0.20882389939968413",
            "extra": "mean: 5.9545974402000414 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-mock-train_default.test.yaml]",
            "value": 0.050189165595039886,
            "unit": "iter/sec",
            "range": "stddev: 0.5756678637170669",
            "extra": "mean: 19.92461895200004 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-mock-train_default.test.yaml]",
            "value": 0.1168138454371749,
            "unit": "iter/sec",
            "range": "stddev: 0.10499232790857435",
            "extra": "mean: 8.560629061199961 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-mock-train_default.test.yaml]",
            "value": 0.009458789196233204,
            "unit": "iter/sec",
            "range": "stddev: 2.6512082979455145",
            "extra": "mean: 105.7217767786 sec\nrounds: 5"
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
          "id": "e01e28c1a414b86e56e2c3ccd3e3aa14a20727cc",
          "message": "Normalize prefill and fill value configuration (#219)\n\nAdd normalize pre fill boolean and fill value configuration for testing\npost fill configuration and multiple fill values\n\n- [x] Add tests for pre-fill and post-fill configuration with multiple\nfill values",
          "timestamp": "2025-04-22T10:39:07-04:00",
          "tree_id": "565a8e25bad0d8d56e8a64b5f00a859a9b72b738",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/e01e28c1a414b86e56e2c3ccd3e3aa14a20727cc"
        },
        "date": 1745335295298,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-mock-train_default.test.yaml]",
            "value": 0.18124641637637115,
            "unit": "iter/sec",
            "range": "stddev: 0.3853196558960974",
            "extra": "mean: 5.517350466800008 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-mock-train_default.test.yaml]",
            "value": 0.046593358132185375,
            "unit": "iter/sec",
            "range": "stddev: 0.5203017024190717",
            "extra": "mean: 21.462286473599942 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-mock-train_default.test.yaml]",
            "value": 0.12921717474761257,
            "unit": "iter/sec",
            "range": "stddev: 0.10748568602331393",
            "extra": "mean: 7.738909335800008 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-mock-train_default.test.yaml]",
            "value": 0.010544297333805632,
            "unit": "iter/sec",
            "range": "stddev: 0.3355670442380267",
            "extra": "mean: 94.83799330979996 sec\nrounds: 5"
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
          "id": "f2cabb570ca994ac0c334b5149ff02d8e532c117",
          "message": "Enable upgrade lints (#220)\n\nOn #210, @alxmrs was leaving comments about using `Union` rather than\nmodern`|`, and there are a variety of related things like dict/list vs\nDict/List and so on. This also turns on warnings about using `\"\".format` rather than\nf-strings, older import paths, inheriting from `object`, using\n`open(foo, \"r\")` rather than `open(foo)` and others. Full list is here\nhttps://docs.astral.sh/ruff/rules/#pyupgrade-up.\n\nAll these fixes were automated (and therefore should be safe) except for\none I will call out below.",
          "timestamp": "2025-04-22T13:28:25-04:00",
          "tree_id": "8d977694a8c8210a21e9ec9450eb0c3586d732e4",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/f2cabb570ca994ac0c334b5149ff02d8e532c117"
        },
        "date": 1745345479244,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-mock-train_default.test.yaml]",
            "value": 0.1754112183636589,
            "unit": "iter/sec",
            "range": "stddev: 0.31408560239348116",
            "extra": "mean: 5.7008896542 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-mock-train_default.test.yaml]",
            "value": 0.045929750202792004,
            "unit": "iter/sec",
            "range": "stddev: 0.8285893108004153",
            "extra": "mean: 21.772380550400023 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-mock-train_default.test.yaml]",
            "value": 0.13068887267657436,
            "unit": "iter/sec",
            "range": "stddev: 0.06842474896896379",
            "extra": "mean: 7.651760853999986 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-mock-train_default.test.yaml]",
            "value": 0.010482940558545846,
            "unit": "iter/sec",
            "range": "stddev: 0.31409967366641944",
            "extra": "mean: 95.39308120799993 sec\nrounds: 5"
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
          "id": "baf745e6863bbf4fa749a2457a0661fc6a582fc2",
          "message": "Add aggregate data benchmarks to W&B Logs. (#192)\n\nI also added fine-grained debug logging statements to the core of each\ndata loader.",
          "timestamp": "2025-04-22T17:14:02-07:00",
          "tree_id": "0f5c8f401186d1df55079976645857dd47fb2a52",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/baf745e6863bbf4fa749a2457a0661fc6a582fc2"
        },
        "date": 1745369928806,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-mock-train_default.test.yaml]",
            "value": 0.16288531091413397,
            "unit": "iter/sec",
            "range": "stddev: 0.4658991770640878",
            "extra": "mean: 6.139289014999986 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-mock-train_default.test.yaml]",
            "value": 0.04353155356395057,
            "unit": "iter/sec",
            "range": "stddev: 0.4734099643169331",
            "extra": "mean: 22.971842677999938 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-mock-train_default.test.yaml]",
            "value": 0.12426601253878036,
            "unit": "iter/sec",
            "range": "stddev: 0.03529023026743787",
            "extra": "mean: 8.0472526604 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-mock-train_default.test.yaml]",
            "value": 0.009869023152098796,
            "unit": "iter/sec",
            "range": "stddev: 0.3422894738996399",
            "extra": "mean: 101.3271510856001 sec\nrounds: 5"
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
          "id": "212d9da3b3954bee74342a10420a25a13cd6bd8b",
          "message": "Use dask for inference (#227)\n\nFixes memory issues with the torch loader, fixes #208. \n\nAfter this, the torch and eager loaders have the same peak memory usage\nduring inference. The issue was that the non-dask version was using\nabout 3x the memory of the dask version and getting OOM killed. This is\ndue to a combination of the InferenceDataset not being written with\neager evaluation in mind (eg slicing repeatedly) and due to what look\nlike some pretty bad performance bugs in xarray. (It was allocating a\nton of memory to lazily track reorderings of the data rather than\neagerly shuffling the arrays -- kinda looks like 6x more than the\nunderlying arrays but I could have misread and don't think it's worth\ndigging into this more now.) (And to add insult to injury, the shuffle\nin question was the identity transform…)",
          "timestamp": "2025-04-24T11:57:17-04:00",
          "tree_id": "00f0babf0b2c9662ad5320466b7459ea845ab3d5",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/212d9da3b3954bee74342a10420a25a13cd6bd8b"
        },
        "date": 1745512822029,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-mock-train_default.test.yaml]",
            "value": 0.16971433884005255,
            "unit": "iter/sec",
            "range": "stddev: 0.6084305091417417",
            "extra": "mean: 5.892254047799997 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-mock-train_default.test.yaml]",
            "value": 0.04481527072648967,
            "unit": "iter/sec",
            "range": "stddev: 0.21532675469841206",
            "extra": "mean: 22.31382258299991 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-mock-train_default.test.yaml]",
            "value": 0.13019361495194376,
            "unit": "iter/sec",
            "range": "stddev: 0.11370170925102274",
            "extra": "mean: 7.68086822360001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-mock-train_default.test.yaml]",
            "value": 0.010341177088091458,
            "unit": "iter/sec",
            "range": "stddev: 0.9121989341393858",
            "extra": "mean: 96.70079058519995 sec\nrounds: 5"
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
          "id": "b929596c1d476649860758ddc80745891dbe94fd",
          "message": "Log data wait time and data load times separately (#231)\n\nWe now log how much time we spend waiting for data (stalling due to\ngetting ahead of the data loaders, which is what we were logging before)\nand also how much time the relevant data loader spent loading that data,\nwhich is not itself important for overall train time since it's in a\nseparate process and async, but is much more sensitive of a measure for\nfiguring out what is happening with load times.\n\nexample wandb run here:\nhttps://wandb.ai/m2lines/ocean-emulators/runs/o3gv8d0k\n\noutput looks like this now:\n\n```\n2025-04-25 15:18:01,323 - INFO - logging - Training Epoch: [1]  [   0/1412]  eta: 7:19:24  lr: 0.000200  iter_time: 18.672(18.672)  data_wait_time: 9.300(9.300)  data_load_time: 8.747(8.747)  loss: 1.4612 (1.4612)  max cpu mem: 1605  max gpu mem: 30424\n2025-04-25 15:18:01,970 - INFO - logging - Training Epoch: [1]  [   1/1412]  eta: 3:47:08  lr: 0.000200  iter_time: 0.645(0.645)  data_wait_time: 0.000(0.000)  data_load_time: 8.745(8.745)  loss: 1.4612 (1.4779)  max cpu mem: 1757  max gpu mem: 30424\n2025-04-25 15:18:02,617 - INFO - logging - Training Epoch: [1]  [   2/1412]  eta: 2:36:22  lr: 0.000200  iter_time: 0.646(0.646)  data_wait_time: 0.000(0.000)  data_load_time: 8.736(8.736)  loss: 1.4612 (1.3976)  max cpu mem: 1758  max gpu mem: 30424\n2025-04-25 15:18:03,265 - INFO - logging - Training Epoch: [1]  [   3/1412]  eta: 2:00:59  lr: 0.000200  iter_time: 0.647(0.647)  data_wait_time: 0.000(0.000)  data_load_time: 8.748(8.748)  loss: 1.2369 (1.3021)  max cpu mem: 1759  max gpu mem: 30424\n2025-04-25 15:18:03,961 - INFO - logging - Training Epoch: [1]  [   4/1412]  eta: 1:39:59  lr: 0.000200  iter_time: 0.695(0.695)  data_wait_time: 0.000(0.000)  data_load_time: 8.854(8.854)  loss: 1.2369 (1.2135)  max cpu mem: 1759  max gpu mem: 30424\n2025-04-25 15:18:04,612 - INFO - logging - Training Epoch: [1]  [   5/1412]  eta: 1:25:48  lr: 0.000200  iter_time: 0.649(0.649)  data_wait_time: 0.000(0.000)  data_load_time: 8.747(8.747)  loss: 1.0158 (1.1252)  max cpu mem: 1759  max gpu mem: 30424\n2025-04-25 15:18:05,263 - INFO - logging - Training Epoch: [1]  [   6/1412]  eta: 1:15:40  lr: 0.000200  iter_time: 0.650(0.650)  data_wait_time: 0.000(0.000)  data_load_time: 8.748(8.748)  loss: 1.0158 (1.0428)  max cpu mem: 1759  max gpu mem: 30424\n```",
          "timestamp": "2025-04-25T17:15:23-04:00",
          "tree_id": "76a1d196a08819dafd2fb5e423d98e6610fab5bd",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/b929596c1d476649860758ddc80745891dbe94fd"
        },
        "date": 1745618591255,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-mock-train_default.test.yaml]",
            "value": 0.17107528645875836,
            "unit": "iter/sec",
            "range": "stddev: 0.29876999550321814",
            "extra": "mean: 5.845379661200059 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-mock-train_default.test.yaml]",
            "value": 0.04398548861742908,
            "unit": "iter/sec",
            "range": "stddev: 1.069544527025251",
            "extra": "mean: 22.734770749000017 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-mock-train_default.test.yaml]",
            "value": 0.12624328795872547,
            "unit": "iter/sec",
            "range": "stddev: 0.06917877265987145",
            "extra": "mean: 7.921213207999972 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-mock-train_default.test.yaml]",
            "value": 0.009970668970310148,
            "unit": "iter/sec",
            "range": "stddev: 0.173101960581308",
            "extra": "mean: 100.29417313700006 sec\nrounds: 5"
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
          "id": "a7a42c83b3ed6c3b7feb97d9000a8cd33b0092d3",
          "message": "revert vectorizing torch loader (#234)\n\nSorry, re-opening because I merged to the wrong branch. We should merge\n#231 before this one.\n\nThis loader appears to be dramatically faster *without* vectorization.\nSee:\nhttps://wandb.ai/m2lines/ocean-emulators/reports/Torch-Loader-with-and-Without-Vectorization--VmlldzoxMjQ3NzA2MA\n\nFixes #230 and reverts #212\n\nFrom comparing profiles, it appears the vectorized version is hitting\nsome kind of performance issue in xarray's indexing:\n\n\n[profile-novec](https://github.com/user-attachments/assets/5b006496-898c-4d5d-9e2c-eebaac0dfd43)\n\n[profile-vec](https://github.com/user-attachments/assets/065bbfc6-1d75-4e6c-8876-b016fa0a393c)",
          "timestamp": "2025-04-25T17:36:11-04:00",
          "tree_id": "aa75b471dedbccc3cd41f7642b776987e5c1d087",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/a7a42c83b3ed6c3b7feb97d9000a8cd33b0092d3"
        },
        "date": 1745619526863,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-mock-train_default.test.yaml]",
            "value": 0.18150736153021071,
            "unit": "iter/sec",
            "range": "stddev: 0.4906494179828221",
            "extra": "mean: 5.5094184145999865 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-mock-train_default.test.yaml]",
            "value": 0.04553285872505972,
            "unit": "iter/sec",
            "range": "stddev: 0.4669205771994029",
            "extra": "mean: 21.962161568600003 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-mock-train_default.test.yaml]",
            "value": 0.12907468063175684,
            "unit": "iter/sec",
            "range": "stddev: 0.09524987160459554",
            "extra": "mean: 7.747452831999999 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-mock-train_default.test.yaml]",
            "value": 0.010222177669354243,
            "unit": "iter/sec",
            "range": "stddev: 0.31483789226674924",
            "extra": "mean: 97.82651332679998 sec\nrounds: 5"
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
          "id": "c2fbe9bf215d428679dcfaf05f23d68ae70388fe",
          "message": "Finetune, Restarts and Preemptible Runs (#226)\n\nCloses #223 , #33 , #160",
          "timestamp": "2025-04-26T11:53:05-04:00",
          "tree_id": "e930269fb388568c98bea5bc2c516239803ce7a2",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/c2fbe9bf215d428679dcfaf05f23d68ae70388fe"
        },
        "date": 1745685475878,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-mock-train_default.test.yaml]",
            "value": 0.18206885519125104,
            "unit": "iter/sec",
            "range": "stddev: 0.41243211061886226",
            "extra": "mean: 5.492427570599967 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-mock-train_default.test.yaml]",
            "value": 0.04625795882520392,
            "unit": "iter/sec",
            "range": "stddev: 0.6931535498481457",
            "extra": "mean: 21.61790155460003 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-mock-train_default.test.yaml]",
            "value": 0.12999669250923046,
            "unit": "iter/sec",
            "range": "stddev: 0.03703700358141157",
            "extra": "mean: 7.692503406800097 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-mock-train_default.test.yaml]",
            "value": 0.010545724307978387,
            "unit": "iter/sec",
            "range": "stddev: 0.42191740339431894",
            "extra": "mean: 94.82516049120004 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}