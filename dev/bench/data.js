window.BENCHMARK_DATA = {
  "lastUpdate": 1756252246988,
  "repoUrl": "https://github.com/Open-Athena/Ocean_Emulator",
  "entries": {
    "Python Benchmark with pytest-benchmark": [
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
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/6f54e2a17d4fffe6740ef6e59a0b382def3304aa"
        },
        "date": 1744825755101,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19387310271997207,
            "unit": "iter/sec",
            "range": "stddev: 0.23382894602171014",
            "extra": "mean: 5.158013081600018 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.05030697984866226,
            "unit": "iter/sec",
            "range": "stddev: 1.0894000371153418",
            "extra": "mean: 19.877957353199996 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.12841156429536216,
            "unit": "iter/sec",
            "range": "stddev: 0.26655604691610746",
            "extra": "mean: 7.7874606191999876 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
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
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/47cef09247ac73131582226ffa3dd7e94d65c391"
        },
        "date": 1744841611929,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19568162641321044,
            "unit": "iter/sec",
            "range": "stddev: 0.44295041232527155",
            "extra": "mean: 5.1103418258000035 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.055264641202127915,
            "unit": "iter/sec",
            "range": "stddev: 1.329404683133495",
            "extra": "mean: 18.094752417599988 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.13727298623077874,
            "unit": "iter/sec",
            "range": "stddev: 0.08361939282318899",
            "extra": "mean: 7.2847544695999655 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
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
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/ef402bd05adab5e866d9c2383b2a76f4672314fb"
        },
        "date": 1744842287018,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19235076693154302,
            "unit": "iter/sec",
            "range": "stddev: 0.34118133425533403",
            "extra": "mean: 5.198835523000002 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.05430085072908913,
            "unit": "iter/sec",
            "range": "stddev: 0.7171642232109837",
            "extra": "mean: 18.415917735599987 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.13625283886972686,
            "unit": "iter/sec",
            "range": "stddev: 0.13234812615418312",
            "extra": "mean: 7.339296621600033 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
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
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/dcc4fa780eeb21762ddae07405a9571a68a6f6dd"
        },
        "date": 1744843006387,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.20011474602750523,
            "unit": "iter/sec",
            "range": "stddev: 0.2704052946793023",
            "extra": "mean: 4.997132994200001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.05789661991650419,
            "unit": "iter/sec",
            "range": "stddev: 1.0773368435918922",
            "extra": "mean: 17.2721654812 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.13949711653631966,
            "unit": "iter/sec",
            "range": "stddev: 0.10632463594171095",
            "extra": "mean: 7.168606956399981 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
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
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/85f6f6ada579d695c859c5f033866a73f81f8ec7"
        },
        "date": 1744843900601,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.18124662961425278,
            "unit": "iter/sec",
            "range": "stddev: 0.43888035255670027",
            "extra": "mean: 5.517343975599988 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.05263275031526606,
            "unit": "iter/sec",
            "range": "stddev: 0.7537985453500919",
            "extra": "mean: 18.999577145599993 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.13632990498399145,
            "unit": "iter/sec",
            "range": "stddev: 0.10803230973929084",
            "extra": "mean: 7.3351477808000025 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
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
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/3e4ab9c74cb22e2ecb2670a4717dcd5a3536f539"
        },
        "date": 1744846507839,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1590125736223508,
            "unit": "iter/sec",
            "range": "stddev: 0.511721468762918",
            "extra": "mean: 6.2888108608000035 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.05130323196343788,
            "unit": "iter/sec",
            "range": "stddev: 1.479031051551947",
            "extra": "mean: 19.491949370999997 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.13228085116127192,
            "unit": "iter/sec",
            "range": "stddev: 0.11120953394585689",
            "extra": "mean: 7.5596731592000195 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
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
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/0ac9469629781b19963f3eefea74a95a4fa3e8ed"
        },
        "date": 1745012614640,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1684152124770462,
            "unit": "iter/sec",
            "range": "stddev: 0.23706041890408053",
            "extra": "mean: 5.937705895399995 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.05149057677845876,
            "unit": "iter/sec",
            "range": "stddev: 0.9274375337423029",
            "extra": "mean: 19.4210292944 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.13255156751016528,
            "unit": "iter/sec",
            "range": "stddev: 0.06633189442387215",
            "extra": "mean: 7.544233680399975 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
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
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/499db56063fe946f79b37da5b8241f484d02e0f8"
        },
        "date": 1745018117148,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.16763378155788883,
            "unit": "iter/sec",
            "range": "stddev: 0.128415190378753",
            "extra": "mean: 5.965384725600018 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.04859398795666355,
            "unit": "iter/sec",
            "range": "stddev: 2.144350145286177",
            "extra": "mean: 20.578677364200008 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.12147719039025452,
            "unit": "iter/sec",
            "range": "stddev: 0.1569591188533516",
            "extra": "mean: 8.23199809599996 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
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
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/e01e28c1a414b86e56e2c3ccd3e3aa14a20727cc"
        },
        "date": 1745335292579,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.17465823760534638,
            "unit": "iter/sec",
            "range": "stddev: 0.6380995106415528",
            "extra": "mean: 5.7254671392000205 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.046065139713128124,
            "unit": "iter/sec",
            "range": "stddev: 0.6583459544397046",
            "extra": "mean: 21.708389602800004 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1380147638595732,
            "unit": "iter/sec",
            "range": "stddev: 0.0772877819134211",
            "extra": "mean: 7.24560164459997 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
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
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/f2cabb570ca994ac0c334b5149ff02d8e532c117"
        },
        "date": 1745345477194,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.17634263985379522,
            "unit": "iter/sec",
            "range": "stddev: 0.27245885643788564",
            "extra": "mean: 5.6707782123999895 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.04538474323424702,
            "unit": "iter/sec",
            "range": "stddev: 0.8760868297855096",
            "extra": "mean: 22.0338362352 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.13783425336897048,
            "unit": "iter/sec",
            "range": "stddev: 0.12652846457352068",
            "extra": "mean: 7.255090629200026 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
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
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/baf745e6863bbf4fa749a2457a0661fc6a582fc2"
        },
        "date": 1745369926699,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.17015797402976113,
            "unit": "iter/sec",
            "range": "stddev: 0.18939259381299503",
            "extra": "mean: 5.876891786600003 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.0421667156470555,
            "unit": "iter/sec",
            "range": "stddev: 1.4011167099784603",
            "extra": "mean: 23.715387472200007 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.13161326905764142,
            "unit": "iter/sec",
            "range": "stddev: 0.10080356952505631",
            "extra": "mean: 7.598018096200008 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
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
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/212d9da3b3954bee74342a10420a25a13cd6bd8b"
        },
        "date": 1745512818344,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.17323228920122472,
            "unit": "iter/sec",
            "range": "stddev: 0.08372543566206345",
            "extra": "mean: 5.772595886200008 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.043826142763783764,
            "unit": "iter/sec",
            "range": "stddev: 0.8143787014379099",
            "extra": "mean: 22.8174312622 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.13627922727373712,
            "unit": "iter/sec",
            "range": "stddev: 0.09291817240686903",
            "extra": "mean: 7.337875478199999 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
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
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/b929596c1d476649860758ddc80745891dbe94fd"
        },
        "date": 1745618589137,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.17254092649381358,
            "unit": "iter/sec",
            "range": "stddev: 0.24300413516430022",
            "extra": "mean: 5.795726384 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.042723367519929376,
            "unit": "iter/sec",
            "range": "stddev: 0.5374417243200751",
            "extra": "mean: 23.4063946278 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1338976466289525,
            "unit": "iter/sec",
            "range": "stddev: 0.10498516873847676",
            "extra": "mean: 7.468391156799998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
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
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/a7a42c83b3ed6c3b7feb97d9000a8cd33b0092d3"
        },
        "date": 1745619524933,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.18103898808457713,
            "unit": "iter/sec",
            "range": "stddev: 0.49174760046621374",
            "extra": "mean: 5.523672058600016 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.044692758076124495,
            "unit": "iter/sec",
            "range": "stddev: 0.7922695129588113",
            "extra": "mean: 22.37498966380001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1357457371555006,
            "unit": "iter/sec",
            "range": "stddev: 0.0684189158614772",
            "extra": "mean: 7.366713835400015 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
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
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/c2fbe9bf215d428679dcfaf05f23d68ae70388fe"
        },
        "date": 1745685473810,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1793757440392398,
            "unit": "iter/sec",
            "range": "stddev: 0.24190517636522843",
            "extra": "mean: 5.5748897676000295 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.046734822291440214,
            "unit": "iter/sec",
            "range": "stddev: 0.5262448120087531",
            "extra": "mean: 21.39732111880003 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.134152268359711,
            "unit": "iter/sec",
            "range": "stddev: 0.11292971931492653",
            "extra": "mean: 7.454216109999993 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.007250170452408171,
            "unit": "iter/sec",
            "range": "stddev: 0.17736657756227542",
            "extra": "mean: 137.92779170699998 sec\nrounds: 5"
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
          "id": "17d4187244ffb5dd8f45d62cdb0d570796c98770",
          "message": "[Clean PR] Ocean Heat Corrector  (#224)\n\nThis PR replaces #165\n\n---------\n\nCo-authored-by: Alex Merose <alex@openathena.ai>\nCo-authored-by: Jesse Rusak <jesse@openathena.ai>",
          "timestamp": "2025-04-26T13:09:51-04:00",
          "tree_id": "3603d044dd950eb742f66f22e5db01eff53aba9f",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/17d4187244ffb5dd8f45d62cdb0d570796c98770"
        },
        "date": 1745690013883,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.18627397922258063,
            "unit": "iter/sec",
            "range": "stddev: 0.24710248245205338",
            "extra": "mean: 5.368436343999986 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.045857236098406834,
            "unit": "iter/sec",
            "range": "stddev: 0.49492965811542744",
            "extra": "mean: 21.8068092428 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.13211779341864152,
            "unit": "iter/sec",
            "range": "stddev: 0.14918409522880988",
            "extra": "mean: 7.569003191199999 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.0068529750982426925,
            "unit": "iter/sec",
            "range": "stddev: 0.8596470634757793",
            "extra": "mean: 145.92202447319994 sec\nrounds: 5"
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
          "id": "4a7b5a01e1e7add44b9f5b671c22e9a9173b4bef",
          "message": "Fix wandb nested config logging (#236)\n\nThe config system broke the nice logging of nested config structures to\nwandb. For example, see the \"data\" key in this run:\n\nhttps://wandb.ai/m2lines/ocean-emulators/runs/2i57cuzt/overview\n\n<img width=\"667\" alt=\"Screenshot 2025-04-29 at 11 21 59 AM\"\nsrc=\"https://github.com/user-attachments/assets/9fa91699-bb21-4d28-882a-87465efc87ac\"\n/>\n\n\nThis PR uses pydantic serialization to correctly output nested keys, for\nexample this run:\n\nhttps://wandb.ai/m2lines/ocean-emulators/runs/twx4knvp/overview\n\n<img width=\"572\" alt=\"Screenshot 2025-04-29 at 11 22 19 AM\"\nsrc=\"https://github.com/user-attachments/assets/574147ee-02fb-4198-b57f-43e756d3b51e\"\n/>\n\nFixes #228",
          "timestamp": "2025-04-29T15:53:52-04:00",
          "tree_id": "0ebf59a0a078cdac0df28657f899e83a587c30c0",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/4a7b5a01e1e7add44b9f5b671c22e9a9173b4bef"
        },
        "date": 1745959230540,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.18392365835070393,
            "unit": "iter/sec",
            "range": "stddev: 0.5756939228387485",
            "extra": "mean: 5.437038437399986 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.045176166569820334,
            "unit": "iter/sec",
            "range": "stddev: 0.33732151359092216",
            "extra": "mean: 22.135565629600013 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1351456694369026,
            "unit": "iter/sec",
            "range": "stddev: 0.12345306917190713",
            "extra": "mean: 7.399423186600029 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.006582401342688222,
            "unit": "iter/sec",
            "range": "stddev: 9.148125324022272",
            "extra": "mean: 151.92024125220001 sec\nrounds: 5"
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
          "id": "ff81c07df95dda17ba1dfb9971412c08c781c751",
          "message": "Add & document some profiling tools (#233)\n\nDocument some helpful profiling tools I've been using and how to call\nthem.",
          "timestamp": "2025-05-14T12:12:01-04:00",
          "tree_id": "fd878e3503e902e8193f3214a27215347d2cf9c3",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/ff81c07df95dda17ba1dfb9971412c08c781c751"
        },
        "date": 1747241805582,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.17804869260418602,
            "unit": "iter/sec",
            "range": "stddev: 0.3886065760003961",
            "extra": "mean: 5.6164411284000035 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.04468997466337762,
            "unit": "iter/sec",
            "range": "stddev: 0.2890292398238716",
            "extra": "mean: 22.37638323880001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.13516294602911502,
            "unit": "iter/sec",
            "range": "stddev: 0.09872667488298757",
            "extra": "mean: 7.398477388799984 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.006817417083930728,
            "unit": "iter/sec",
            "range": "stddev: 0.1790249119126206",
            "extra": "mean: 146.68311879539993 sec\nrounds: 5"
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
          "id": "d93abe463079a04cdd70baf24bd56edcda7f9b8f",
          "message": "CUDA Memory profiler (#242)\n\nLets us turn on CUDA memory snapshots via config.\n\n---------\n\nCo-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>",
          "timestamp": "2025-05-19T16:16:56-04:00",
          "tree_id": "976d6aa522f8d1e5f0cdec1d477541c47a89b745",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/d93abe463079a04cdd70baf24bd56edcda7f9b8f"
        },
        "date": 1747688784944,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.16728931727134566,
            "unit": "iter/sec",
            "range": "stddev: 0.7194851867956632",
            "extra": "mean: 5.977668008399997 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.04349046781415819,
            "unit": "iter/sec",
            "range": "stddev: 0.5784599734736856",
            "extra": "mean: 22.993544338799985 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.13087546048563198,
            "unit": "iter/sec",
            "range": "stddev: 0.17134422155592863",
            "extra": "mean: 7.640851816599979 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.006698627166167523,
            "unit": "iter/sec",
            "range": "stddev: 1.158003883839965",
            "extra": "mean: 149.28431978580005 sec\nrounds: 5"
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
          "id": "50579904a32d156dbb0accb473f4cabb23f2738d",
          "message": "Apply same fix to benchmarks, too (#245)\n\nLooks like this wasn't a perf regression, it was a failure. Sorry, I\nshould have checked here after finding it in the other two places!",
          "timestamp": "2025-05-19T19:00:44-04:00",
          "tree_id": "eaf4c42454c0be041b139e98fae1da1f31c209a9",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/50579904a32d156dbb0accb473f4cabb23f2738d"
        },
        "date": 1747698328666,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19060785333375374,
            "unit": "iter/sec",
            "range": "stddev: 0.18975980636759385",
            "extra": "mean: 5.2463735492000065 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.047810928950504976,
            "unit": "iter/sec",
            "range": "stddev: 0.633377614180422",
            "extra": "mean: 20.915719939999995 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1351348531817425,
            "unit": "iter/sec",
            "range": "stddev: 0.11804814526190831",
            "extra": "mean: 7.400015439799995 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.006765920058316589,
            "unit": "iter/sec",
            "range": "stddev: 0.2785232522528074",
            "extra": "mean: 147.79955887460002 sec\nrounds: 5"
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
          "id": "d6830a1c89aaad5ac0b1488cb19f45960b48f166",
          "message": "Activation Checkpointing (#243)\n\nI am looking into fitting the high res models on our GPUs, so I tried a\ncouple options for activation checkpointing: checkpoint each\nConvNeXtBlock and other top-level layers (\"all\") or checkpointing just\nsimple scalings/nonlinearities (\"simple\").\n\nResults are here:\nhttps://wandb.ai/m2lines/ocean-emulators/reports/Checkpointing-2025-05-16--VmlldzoxMjgwOTIyOA\n\ntl;dr: \"all\" saves about 67% of GPU memory usage (!) but at a\nperformance cost of about 10%. \"simple\" saves about 20% of GPU memory\nbut has practically no performance degradation. There are probably\nstrategies in between those two we could look at, too. (Perhaps also\nsome wins to be had by using the extra space to pre-load future steps\nworth of data or similar.)",
          "timestamp": "2025-05-20T15:10:49-04:00",
          "tree_id": "d0f7ce37b57625846e6198b323e9edcb4529910f",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/d6830a1c89aaad5ac0b1488cb19f45960b48f166"
        },
        "date": 1747771060830,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19912305538153013,
            "unit": "iter/sec",
            "range": "stddev: 0.40149215470488164",
            "extra": "mean: 5.0220201677999965 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.04814422178843577,
            "unit": "iter/sec",
            "range": "stddev: 0.4886003860496873",
            "extra": "mean: 20.770924585600007 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.13635660957656973,
            "unit": "iter/sec",
            "range": "stddev: 0.11540477796200382",
            "extra": "mean: 7.333711237800026 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.006898894790418253,
            "unit": "iter/sec",
            "range": "stddev: 1.4224786853745308",
            "extra": "mean: 144.9507537626 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "m@mihasya.com",
            "name": "Mikhail P",
            "username": "mihasya"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f3f47b96bedd2e1fb0bf6a45205e57ebb9ed5afa",
          "message": "Revert \"Switch to using EC2 for CI workers\" (#249)\n\nReverting while we figure out why EC2 runners are not registering\ncorrectly when running on the parent repo.\n\nReverts suryadheeshjith/Ocean_Emulator#246",
          "timestamp": "2025-05-22T08:21:10-07:00",
          "tree_id": "d0f7ce37b57625846e6198b323e9edcb4529910f",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/f3f47b96bedd2e1fb0bf6a45205e57ebb9ed5afa"
        },
        "date": 1747929832251,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1884313043010343,
            "unit": "iter/sec",
            "range": "stddev: 0.515699577521025",
            "extra": "mean: 5.306973826399985 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.049503129345499636,
            "unit": "iter/sec",
            "range": "stddev: 0.7974173614700626",
            "extra": "mean: 20.200743129199985 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.13805140077170294,
            "unit": "iter/sec",
            "range": "stddev: 0.11207753949948228",
            "extra": "mean: 7.243678763200023 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.007150754893198183,
            "unit": "iter/sec",
            "range": "stddev: 0.48677097898961175",
            "extra": "mean: 139.8453750598 sec\nrounds: 5"
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
          "id": "ea6c3ccf85283472fe7dea10c421310c7309458e",
          "message": "Torch Loader works with compact data. (#215)\n\nThis PR depends on #211. Fixes #204.\n\nHere is the WandB profile results:\nhttps://wandb.ai/m2lines/ocean-emulators/reports/Compaction-and-Vectorization-on-Data-Load-Time--VmlldzoxMjg1MDE4NQ\n\nConclusions: compaction is at least as fast as before, but often\npredictably faster.",
          "timestamp": "2025-05-22T14:42:30-07:00",
          "tree_id": "3ac0be2c892c274e4df61226b84c28f86ca6a928",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/ea6c3ccf85283472fe7dea10c421310c7309458e"
        },
        "date": 1747952710191,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19763234985555772,
            "unit": "iter/sec",
            "range": "stddev: 0.13890165388740922",
            "extra": "mean: 5.059900369199999 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.04839226472213922,
            "unit": "iter/sec",
            "range": "stddev: 0.561742846737054",
            "extra": "mean: 20.664459614400005 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1350834731289528,
            "unit": "iter/sec",
            "range": "stddev: 0.09347638751865638",
            "extra": "mean: 7.402830093399984 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.007181711180854332,
            "unit": "iter/sec",
            "range": "stddev: 0.13480467696066134",
            "extra": "mean: 139.242580886 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "mihasya",
            "username": "mihasya",
            "email": "m@mihasya.com"
          },
          "committer": {
            "name": "mihasya",
            "username": "mihasya",
            "email": "m@mihasya.com"
          },
          "id": "a4e8a8862791a04104ac27a7d98a32155c7f221e",
          "message": "use fixed/vendored version of gha-runner on benchmark workflow",
          "timestamp": "2025-05-22T18:24:50Z",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/a4e8a8862791a04104ac27a7d98a32155c7f221e"
        },
        "date": 1748007638886,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.2617661501641519,
            "unit": "iter/sec",
            "range": "stddev: 0.3091794003539715",
            "extra": "mean: 3.820203641200004 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06663613527422867,
            "unit": "iter/sec",
            "range": "stddev: 0.33575227343370356",
            "extra": "mean: 15.006872710800007 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19279759459940143,
            "unit": "iter/sec",
            "range": "stddev: 0.10105792660984869",
            "extra": "mean: 5.186786702799997 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.008704564066344268,
            "unit": "iter/sec",
            "range": "stddev: 0.23081076676807682",
            "extra": "mean: 114.88226088960005 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "m@mihasya.com",
            "name": "Mikhail P",
            "username": "mihasya"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d586ae5475dc034925ee58d35e9376a73c6d0981",
          "message": "Switch to using EC2 for CI workers, take 2 (#250)\n\nThis changeset switches the repo to use OpenAthena's EC2 account to\nperform CI checks.\n\nUnlike its predecessor, #246, this PR depends on a branch of\n`start-gha-ec2-runner` which contains a fix for a bug that prevented the\nrunner machines from being detected by GHA. We will upstream this fix\nand send a follow-up PR when the fix has been accepted.\n\nWe observe a modest improvement in benchmarks when running on AWS\n`g6.xlarge` VMs vs `g2-standard-8` Google Cloud VMs.",
          "timestamp": "2025-05-27T06:38:11-07:00",
          "tree_id": "0a2f733705754f56bc04f44be0e10fd48ad40fa2",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/d586ae5475dc034925ee58d35e9376a73c6d0981"
        },
        "date": 1748355004406,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.2738620160721509,
            "unit": "iter/sec",
            "range": "stddev: 0.19640697017166367",
            "extra": "mean: 3.651473885800004 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.0676651675080029,
            "unit": "iter/sec",
            "range": "stddev: 0.6406258523156678",
            "extra": "mean: 14.778652545000023 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.20428218936064174,
            "unit": "iter/sec",
            "range": "stddev: 0.08524761439452808",
            "extra": "mean: 4.8951893609999955 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.00947935829601696,
            "unit": "iter/sec",
            "range": "stddev: 0.2904810441943721",
            "extra": "mean: 105.49237287719998 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "mihasya",
            "username": "mihasya",
            "email": "m@mihasya.com"
          },
          "committer": {
            "name": "mihasya",
            "username": "mihasya",
            "email": "m@mihasya.com"
          },
          "id": "c5a6f952a2b2c43315c3cd50226ce601fbda7f36",
          "message": "try out 6g.2xlarge to see what that does to benchmark results",
          "timestamp": "2025-05-28T14:19:14Z",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/c5a6f952a2b2c43315c3cd50226ce601fbda7f36"
        },
        "date": 1748443888928,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.2895639527720061,
            "unit": "iter/sec",
            "range": "stddev: 0.23283181839994882",
            "extra": "mean: 3.453468535799999 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07043990395439581,
            "unit": "iter/sec",
            "range": "stddev: 0.5198133186570716",
            "extra": "mean: 14.196498630200006 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.20118770729239324,
            "unit": "iter/sec",
            "range": "stddev: 0.1357089986876567",
            "extra": "mean: 4.970482607800011 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.009685292517153634,
            "unit": "iter/sec",
            "range": "stddev: 0.16305057916098264",
            "extra": "mean: 103.24933379439999 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "m@mihasya.com",
            "name": "Mikhail P",
            "username": "mihasya"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "cd9c83eef13b76242972bb05ca858bfc872618bb",
          "message": "Remove pointer to patched version of GH Action (#254)\n\nMinor cleanup from #250 removing a reference to a patched CI dependency.",
          "timestamp": "2025-05-29T08:39:15-07:00",
          "tree_id": "c8ac84cac2c120a97b3b75ff9d967ab5fce952aa",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/cd9c83eef13b76242972bb05ca858bfc872618bb"
        },
        "date": 1748535146669,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.26972835353855,
            "unit": "iter/sec",
            "range": "stddev: 0.262568311224251",
            "extra": "mean: 3.7074337453999933 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06624735514435279,
            "unit": "iter/sec",
            "range": "stddev: 0.9537382849676402",
            "extra": "mean: 15.094942248199994 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1924269539376995,
            "unit": "iter/sec",
            "range": "stddev: 0.15767553294733316",
            "extra": "mean: 5.196777164199989 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.008853005104733642,
            "unit": "iter/sec",
            "range": "stddev: 0.4602156783364526",
            "extra": "mean: 112.95599496099996 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Jesse Rusak",
            "username": "jder",
            "email": "jesse@openathena.ai"
          },
          "committer": {
            "name": "Jesse Rusak",
            "username": "jder",
            "email": "jesse@openathena.ai"
          },
          "id": "57579549ffb2d962977b629f3fc7801397ebca6e",
          "message": "fix",
          "timestamp": "2025-06-06T13:42:54Z",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/57579549ffb2d962977b629f3fc7801397ebca6e"
        },
        "date": 1749220405982,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.26470706724094395,
            "unit": "iter/sec",
            "range": "stddev: 0.3335491154228296",
            "extra": "mean: 3.7777608676 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07073527790143688,
            "unit": "iter/sec",
            "range": "stddev: 0.4634304277133242",
            "extra": "mean: 14.1372173782 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.20594677881164064,
            "unit": "iter/sec",
            "range": "stddev: 0.0831457976898759",
            "extra": "mean: 4.855623407999997 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.009617933557805606,
            "unit": "iter/sec",
            "range": "stddev: 0.15899376633004914",
            "extra": "mean: 103.97243794519999 sec\nrounds: 5"
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
          "id": "cd2e5d64d8d9b6ceb2a1ed6220c622dc00763ee9",
          "message": "Switch to torch loader and reduce default number of workers per GPU (#255)\n\nI think the torch loader is a good default now, and I found 4 workers by\ndefault overloads the 8-GPU systems I've been using, resulting in very\nlong waits at the start of an epoch. 1-worker-per-GPU seems to be too\nfew on my other host, so I was thinking this might be a better default.\n(Though of course might need tuning.)",
          "timestamp": "2025-06-10T13:00:58-04:00",
          "tree_id": "6cfee0c0e32770bde2438ef83e8574ba905f607f",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/cd2e5d64d8d9b6ceb2a1ed6220c622dc00763ee9"
        },
        "date": 1749576182462,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2797808605636831,
            "unit": "iter/sec",
            "range": "stddev: 0.05026706096497794",
            "extra": "mean: 781.3837750000005 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.24295918281438056,
            "unit": "iter/sec",
            "range": "stddev: 0.13111095302224376",
            "extra": "mean: 4.1159176962 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.20973270126245713,
            "unit": "iter/sec",
            "range": "stddev: 0.072762964817674",
            "extra": "mean: 4.767973682599984 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.01493338665091722,
            "unit": "iter/sec",
            "range": "stddev: 0.10905029229295636",
            "extra": "mean: 66.9640466276 sec\nrounds: 5"
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
          "id": "07aa2191c4c8c85a267c5fdcd54c201fb0cec1f6",
          "message": "Complain if the validation or inference time ranges overlap with train (#258)\n\nI did this by accident last week so let's prevent people from doing it.\n\n---------\n\nCo-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>",
          "timestamp": "2025-06-10T14:20:15-04:00",
          "tree_id": "39ff4517aeaf326cbff211d3b6f8035a28a5e4ce",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/07aa2191c4c8c85a267c5fdcd54c201fb0cec1f6"
        },
        "date": 1749580908370,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2888490131262715,
            "unit": "iter/sec",
            "range": "stddev: 0.014646213528990187",
            "extra": "mean: 775.8860733999938 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.25214472838481844,
            "unit": "iter/sec",
            "range": "stddev: 0.0032797482536670317",
            "extra": "mean: 3.9659762327999943 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.20703178089127164,
            "unit": "iter/sec",
            "range": "stddev: 0.010202874067946477",
            "extra": "mean: 4.830176293200014 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.014696650718316428,
            "unit": "iter/sec",
            "range": "stddev: 0.22438019575081325",
            "extra": "mean: 68.04271389219998 sec\nrounds: 5"
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
          "id": "87e1ec3a77ea6a1fde81946786cc3d0873cf268a",
          "message": "Deduplicate start/stop jobs, remove patched version of action (#256)\n\nI resisted doing this because this is more moving pieces and also more\ncode, but this is also at least the third time we've had them drift out\nof sync (in this case, one was using the mihasya patched version and the\nother not), so I think it's probably worth it.",
          "timestamp": "2025-06-10T14:19:58-04:00",
          "tree_id": "0fb347ff79be7fba168adb4c0760c6a343a62f12",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/87e1ec3a77ea6a1fde81946786cc3d0873cf268a"
        },
        "date": 1749580960313,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.243005701386733,
            "unit": "iter/sec",
            "range": "stddev: 0.03860602103021056",
            "extra": "mean: 804.5015391999982 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.25203642747955335,
            "unit": "iter/sec",
            "range": "stddev: 0.00806869610681853",
            "extra": "mean: 3.967680426200002 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.20336328141140542,
            "unit": "iter/sec",
            "range": "stddev: 0.09823045063119513",
            "extra": "mean: 4.917308537999998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.014193459551834094,
            "unit": "iter/sec",
            "range": "stddev: 0.12069271545744033",
            "extra": "mean: 70.45498642160001 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Jesse Rusak",
            "username": "jder",
            "email": "jesse@openathena.ai"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "07aa2191c4c8c85a267c5fdcd54c201fb0cec1f6",
          "message": "Complain if the validation or inference time ranges overlap with train (#258)\n\nI did this by accident last week so let's prevent people from doing it.\n\n---------\n\nCo-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>",
          "timestamp": "2025-06-10T18:20:15Z",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/07aa2191c4c8c85a267c5fdcd54c201fb0cec1f6"
        },
        "date": 1749759002887,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2776643478017173,
            "unit": "iter/sec",
            "range": "stddev: 0.027592160253701842",
            "extra": "mean: 782.6781749999896 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.25577454496468427,
            "unit": "iter/sec",
            "range": "stddev: 0.005297014647398327",
            "extra": "mean: 3.9096932032000042 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.20920866387359222,
            "unit": "iter/sec",
            "range": "stddev: 0.06905945962220844",
            "extra": "mean: 4.779916765799999 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.014654542174261011,
            "unit": "iter/sec",
            "range": "stddev: 0.11764469889675631",
            "extra": "mean: 68.238228674 sec\nrounds: 5"
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
          "id": "a644488d63d087f52514e991d2ca60937f56e212",
          "message": "Use passed loader version to decide whether to use dask (#260)\n\n@alxmrs pretty sure this is why our benchmarks did something surprising\nwhen we changed the default loader version. Previously the passed\n`version` was ignored for the purpose of deciding whether to use dask or\nnot. Seems like good additional evidence that we should try and tackle\nthis duplication sometime soon.\n\nThis also makes me want to move to a world where we're not using config\nobjects in most tests. It's hard to tell whether this change is\nsufficient -- if any other code we're calling here transitively uses\n`cfg.data.loader_version`, then it's probably broken. We could also make\na new train config but then if the *caller* is using\n`cfg.data.loader_version` we have the same problem. We could update all\ncallers to mutate train config rather than passing in verison and\ntime_config separately but I would rather move away from manipulating\nconfig rather than towards. (But could be convinced this is still worth\ndoing in the short term.)",
          "timestamp": "2025-06-16T11:57:54-04:00",
          "tree_id": "b1f88c1e9950ab1dba0d7d5cc696362e6df605d8",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/a644488d63d087f52514e991d2ca60937f56e212"
        },
        "date": 1750090974488,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.3136650908991965,
            "unit": "iter/sec",
            "range": "stddev: 0.023498250258816286",
            "extra": "mean: 761.2290278000046 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07036300291880149,
            "unit": "iter/sec",
            "range": "stddev: 0.6128717650384378",
            "extra": "mean: 14.212014247800004 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19643050120881841,
            "unit": "iter/sec",
            "range": "stddev: 0.18103979376403853",
            "extra": "mean: 5.0908590765999975 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.014303610361879627,
            "unit": "iter/sec",
            "range": "stddev: 0.15330692516465308",
            "extra": "mean: 69.9124189418 sec\nrounds: 5"
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
          "id": "9f3b92249932b442b16a93744398acd50aabcc9c",
          "message": "Notes on how to reproduce OM4 half-deg preprocessing. (#261)\n\nI've refreshed the contributing guide, adding a data engineering\nsection. Here, I explain how to clone the data and how to start a coiled\nnotebook to create the half-deg data. The docs indicate that this all\nmay change soon -- and that is my plan. I am happy to refresh these docs\noverall.\n\n---------\n\nCo-authored-by: Surya <surya.dheeshjith@gmail.com>",
          "timestamp": "2025-06-18T11:52:16-07:00",
          "tree_id": "96598efc2baf77a86569cde8e32850a308b0cfed",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/9f3b92249932b442b16a93744398acd50aabcc9c"
        },
        "date": 1750274287793,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.279651482393934,
            "unit": "iter/sec",
            "range": "stddev: 0.013527511473291953",
            "extra": "mean: 781.462776199993 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06842798008947724,
            "unit": "iter/sec",
            "range": "stddev: 0.7494418126394848",
            "extra": "mean: 14.613904994599988 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.2025737917742135,
            "unit": "iter/sec",
            "range": "stddev: 0.09870940585952467",
            "extra": "mean: 4.936472735399991 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.013687404163350968,
            "unit": "iter/sec",
            "range": "stddev: 0.14165644183692938",
            "extra": "mean: 73.05987227860003 sec\nrounds: 5"
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
          "id": "5f662af829c4088ae43c55060a116d7bb09aaffa",
          "message": "Avoid failing benchmarks when they regress (#264)\n\nThis is a bit weird since I always think something went wrong, but I\nthink regressions are \"normal\" and not failures. In addition, failing\nthe first \"store\" task causes the second one to never run, which seems\nbad.",
          "timestamp": "2025-06-18T15:01:41-04:00",
          "tree_id": "7deb607a3746aef4172aedcb6a2c5e4d5c0459d6",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/5f662af829c4088ae43c55060a116d7bb09aaffa"
        },
        "date": 1750274845227,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.277756420008245,
            "unit": "iter/sec",
            "range": "stddev: 0.019760080264875437",
            "extra": "mean: 782.6217769999914 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06728627100128563,
            "unit": "iter/sec",
            "range": "stddev: 0.8334095483031617",
            "extra": "mean: 14.861872787999994 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.20589841229669018,
            "unit": "iter/sec",
            "range": "stddev: 0.09486406029743173",
            "extra": "mean: 4.856764017000023 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.013861848617101852,
            "unit": "iter/sec",
            "range": "stddev: 0.24263095963673764",
            "extra": "mean: 72.1404502114 sec\nrounds: 5"
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
          "id": "72cb9be4c09bcad079e6648a880b3cbc6e24b361",
          "message": "Support absolute URLs for file paths, remove '*' support (#262)\n\nAt the moment, we don't support `root` being a URL but this seems like a\nstep in the right direction.\n\nI also don't think we need the \"*\" support, and defaulting to `debug:\nfalse` in eval is consistent with the other configurations.\n\n---------\n\nCo-authored-by: Alex Merose <alex@openathena.ai>",
          "timestamp": "2025-06-23T13:08:51-04:00",
          "tree_id": "5a023c1cd0f4ac43133276ef8f72ac1d23c53e52",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/72cb9be4c09bcad079e6648a880b3cbc6e24b361"
        },
        "date": 1750700097505,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.282435280003015,
            "unit": "iter/sec",
            "range": "stddev: 0.029712756338617",
            "extra": "mean: 779.7664456000064 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07025347018085569,
            "unit": "iter/sec",
            "range": "stddev: 0.3582300928276252",
            "extra": "mean: 14.234172310999998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1997026008897286,
            "unit": "iter/sec",
            "range": "stddev: 0.20813233671647652",
            "extra": "mean: 5.00744605000001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.013407374845859608,
            "unit": "iter/sec",
            "range": "stddev: 0.41703317111339155",
            "extra": "mean: 74.58581649999996 sec\nrounds: 5"
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
          "id": "32f3043d20bb43b24a1a097004f5c0f34f013075",
          "message": "Tweaks while running half-degree experiments (#283)\n\nThe 3 commits are independent so probably easiest to review that way.\n(Happy to break into separate PRs but I think they're all pretty\nuncontroversial.)\n\n* Updates config for the half-degree experiments to load remotely when\ndesired and to avoid doing inference during training.\n* Adds a flag to load xarray variables concurrently based on [this\nsuggestion](https://github.com/pydata/xarray/issues/8965#issuecomment-2083787484).\nThis helps a lot locally with small datasizes (3x improvement for me)\nbut with larger batch sizes it seems to not help (maybe GIL\ncontention?).\n* Swaps to the the \"spawn\" multiprocessing type since the s3fs requires\nthis when passing those stores around.\n* Adds more logging statements so you can tell what's going on more\neasily\n* Removes some logic we had about tweaking the number of workers based\non the number of GPUs in use -- the worker counts are are *already* per\nGPU so this was using `num_gpus * num_gpus * num_workers` workers in\ntotal. Also allows us to use multiple workers for CPU jobs, for testing\npurposes.",
          "timestamp": "2025-06-25T13:20:35-04:00",
          "tree_id": "83e2145e1efb9308a0172521dcd9a66d10070aca",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/32f3043d20bb43b24a1a097004f5c0f34f013075"
        },
        "date": 1750874163301,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2548113151832494,
            "unit": "iter/sec",
            "range": "stddev: 0.04056835319914753",
            "extra": "mean: 796.9325649999917 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06959269397631941,
            "unit": "iter/sec",
            "range": "stddev: 0.5504767462045158",
            "extra": "mean: 14.369324463000009 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.08058037036409946,
            "unit": "iter/sec",
            "range": "stddev: 0.08856686796862123",
            "extra": "mean: 12.409970262999991 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.009288079900235759,
            "unit": "iter/sec",
            "range": "stddev: 0.1585540164941068",
            "extra": "mean: 107.66487915060002 sec\nrounds: 5"
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
          "id": "8b0601504cae8d4940cd645928e53660f6a886a4",
          "message": "Put OSN Bucket keys in environment variables. (#285)\n\nI plan to scrub the access keys that I checked in. For good measure, I\nalso plan to invalidate these keys and get new ones for myself.\n\nThis PR also adds a script that converts jupyter notebooks into Python\nscripts. It fixes #287.\n\n---------\n\nCo-authored-by: Ryan Williams <nobigdealstyle@gmail.com>",
          "timestamp": "2025-06-30T12:52:21-07:00",
          "tree_id": "c5c5b0946a4178d50c6a0b380166935812bfbf44",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/8b0601504cae8d4940cd645928e53660f6a886a4"
        },
        "date": 1751315395864,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.3024160606146504,
            "unit": "iter/sec",
            "range": "stddev: 0.006885193647572271",
            "extra": "mean: 767.8037996000057 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06677421815838273,
            "unit": "iter/sec",
            "range": "stddev: 0.4468789154920082",
            "extra": "mean: 14.975839891200007 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.079074574047734,
            "unit": "iter/sec",
            "range": "stddev: 0.18467526790378164",
            "extra": "mean: 12.646290062799983 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.008524999739728518,
            "unit": "iter/sec",
            "range": "stddev: 4.920336872321891",
            "extra": "mean: 117.3020563672 sec\nrounds: 5"
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
          "id": "541044de5af013efb189b91ebd7807d39752370d",
          "message": "Put OSN Bucket keys in environment variables. (#285)\n\nI plan to scrub the access keys that I checked in. For good measure, I\nalso plan to invalidate these keys and get new ones for myself.\n\nThis PR also adds a script that converts jupyter notebooks into Python\nscripts. It fixes #287.\n\n---------\n\nCo-authored-by: Ryan Williams <nobigdealstyle@gmail.com>",
          "timestamp": "2025-06-30T12:52:21-07:00",
          "tree_id": "c5c5b0946a4178d50c6a0b380166935812bfbf44",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/541044de5af013efb189b91ebd7807d39752370d"
        },
        "date": 1751914007035,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2307488053377837,
            "unit": "iter/sec",
            "range": "stddev: 0.02357251072361151",
            "extra": "mean: 812.513484200008 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06780935896615534,
            "unit": "iter/sec",
            "range": "stddev: 0.4953510278633571",
            "extra": "mean: 14.747226861399986 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07731304874625645,
            "unit": "iter/sec",
            "range": "stddev: 0.15848420831701246",
            "extra": "mean: 12.934427191999987 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.008686729970502235,
            "unit": "iter/sec",
            "range": "stddev: 0.9577207582242829",
            "extra": "mean: 115.11811733479999 sec\nrounds: 5"
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
          "id": "52ae948638fd948154d8aa445a146cdb411b3346",
          "message": "Updated `concurrent_compute` logic to create a thread pool only once. (#288)\n\nI think that it's a good idea to try to load all the data variables\nconcurrently. However, before this PR, we create and then tear down the\nexecutor way too many times (per loop per getitem). This change will\nallow to to see how effective we can concurrently load Xarray data.",
          "timestamp": "2025-07-07T14:10:50-07:00",
          "tree_id": "528b34aa5b5b696c4bbbe921eb5716497b18ec9a",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/52ae948638fd948154d8aa445a146cdb411b3346"
        },
        "date": 1751924794252,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2683366206069064,
            "unit": "iter/sec",
            "range": "stddev: 0.03737432159552265",
            "extra": "mean: 788.4342246000074 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06863162678721287,
            "unit": "iter/sec",
            "range": "stddev: 0.7218410681293501",
            "extra": "mean: 14.570541990800013 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.08026914643705897,
            "unit": "iter/sec",
            "range": "stddev: 0.1428850163290985",
            "extra": "mean: 12.458086878799998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.009114763454235243,
            "unit": "iter/sec",
            "range": "stddev: 0.2412905434797989",
            "extra": "mean: 109.71211760140002 sec\nrounds: 5"
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
          "id": "7555580c3f9be1f1fb7658c24d798eaaa847c5ed",
          "message": "Ignore some more things (#294)",
          "timestamp": "2025-07-08T15:14:27-04:00",
          "tree_id": "795235b648ce3bb3f130385fdc6db4fb21012be1",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/7555580c3f9be1f1fb7658c24d798eaaa847c5ed"
        },
        "date": 1752004167017,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2896813540713021,
            "unit": "iter/sec",
            "range": "stddev: 0.020008024674997616",
            "extra": "mean: 775.3853282000023 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06962100945203059,
            "unit": "iter/sec",
            "range": "stddev: 0.4377782849892488",
            "extra": "mean: 14.3634803326 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.08137648097262781,
            "unit": "iter/sec",
            "range": "stddev: 0.09201308144920951",
            "extra": "mean: 12.288562838400015 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.009425767142559723,
            "unit": "iter/sec",
            "range": "stddev: 0.35237170686674646",
            "extra": "mean: 106.09216044440002 sec\nrounds: 5"
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
          "id": "cc0d2b287407dd91f914074ec94f4e75da12779c",
          "message": "Add a \"Location\" to support S3 and local paths (#284)\n\nThis lets us:\n* Add structured info for locations (eg s3 endpoint URLs)\n* Still use raw strings most of the time\n* Get some type safety to know if we've correctly resolved these against\nthe root before opening.\n* (In a future PR) detect the need for \"spawn\" multiprocessing (required\nfor S3 but not for local files) which has a performance cost.\n\nAlso removes the \"gantry\" boolean because I think we can use\n`data_root:/` if we want that",
          "timestamp": "2025-07-09T09:55:11-04:00",
          "tree_id": "50a1e982f6219a0b40fd1a418f47ee6ecafb894e",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/cc0d2b287407dd91f914074ec94f4e75da12779c"
        },
        "date": 1752071462469,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.3006482586738843,
            "unit": "iter/sec",
            "range": "stddev: 0.02848003202743674",
            "extra": "mean: 768.8473754000029 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06934207658397938,
            "unit": "iter/sec",
            "range": "stddev: 0.39743518518007204",
            "extra": "mean: 14.421258336400001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.08119363245453748,
            "unit": "iter/sec",
            "range": "stddev: 0.08298976921675245",
            "extra": "mean: 12.316236751200005 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.00908505650966302,
            "unit": "iter/sec",
            "range": "stddev: 2.021888294566076",
            "extra": "mean: 110.07086185280005 sec\nrounds: 5"
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
          "id": "628310a0e41f387ea808db2027112753ff6c7820",
          "message": "Fix performance regression by moving towards a more buildy world with DataContainer (#295)\n\nFixes the performance regression from\n[here](https://github.com/LaureZanna/Ocean_Emulator/commit/32f3043d20bb43b24a1a097004f5c0f34f013075#commitcomment-160869800)\nby detecting if the data source supports forking or not.\n\nDoing this required a bit of code movement so I took the opportunity to\nmove us closer towards a world where creating the trainer is `trainer =\ncfg.build()` by moving a bunch of the stuff in `Trainer.__init__` into\n`DataConfig.build` and creating a new class which is the result of that\nmethod, `DataContainer`. This also reduces some of the duplication\nbetween train/eval/tests.\n\nLast commit is some drive-by fixes for warnings that were being printed\nin tests.",
          "timestamp": "2025-07-09T19:10:04-04:00",
          "tree_id": "9ef480f8a8af2a3a899f1af138ef0f5623910c5b",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/628310a0e41f387ea808db2027112753ff6c7820"
        },
        "date": 1752103960417,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2480129359556094,
            "unit": "iter/sec",
            "range": "stddev: 0.02787960164737591",
            "extra": "mean: 801.2737457999947 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.296168717673847,
            "unit": "iter/sec",
            "range": "stddev: 0.026364366137923242",
            "extra": "mean: 771.5045011999962 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19243796126118884,
            "unit": "iter/sec",
            "range": "stddev: 0.08745411964840068",
            "extra": "mean: 5.196479911999989 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.013110605734168453,
            "unit": "iter/sec",
            "range": "stddev: 0.9335722901170258",
            "extra": "mean: 76.27412648020001 sec\nrounds: 5"
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
          "id": "1946c397256e79c51032d671dae1c58466a27af3",
          "message": "Fix bug which ignored loader version in tests (#297)\n\nOur benchmarks showed a surprising jump in one of them after #295:\n<img width=\"1028\" height=\"514\" alt=\"Screenshot 2025-07-10 at 10 26\n37 AM\"\nsrc=\"https://github.com/user-attachments/assets/3322d95a-817e-4113-be9a-704ed2b3638c\"\n/>\n\nTurns out we were always building a torch loader, not an eager loader.\nRunning the benchmarks locally after this change again shows a big\nperformance gap between eager and torch loaders.",
          "timestamp": "2025-07-11T12:44:20-04:00",
          "tree_id": "bfd6a865745f3919d0465de8fe384d3b749a6cbb",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/1946c397256e79c51032d671dae1c58466a27af3"
        },
        "date": 1752253799840,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2811474542032897,
            "unit": "iter/sec",
            "range": "stddev: 0.02835997300445083",
            "extra": "mean: 780.5502768000053 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06960449475883854,
            "unit": "iter/sec",
            "range": "stddev: 0.5361193482834711",
            "extra": "mean: 14.366888280200005 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19276934162074785,
            "unit": "iter/sec",
            "range": "stddev: 0.05945805669464332",
            "extra": "mean: 5.187546897200013 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.013455071952436667,
            "unit": "iter/sec",
            "range": "stddev: 0.2224248722832982",
            "extra": "mean: 74.32141600839998 sec\nrounds: 5"
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
          "id": "3683713b49d46850d0f38310c278c7e1b78de68a",
          "message": "Improvements from trying to understand the model (#296)\n\n* Makes the model summary include grid sizes by delaying calling it\nuntil we have data to run a forward pass (this will probably slow down\nbenchmarks but we can ignore)\n* This required making TrainData a typing.Mapping to satisfy the summary\nfunction (though this still isn't perfect, it somehow can't figure out\nhow much memory it uses)\n* When debug = true, include all the layers not just top ones\n* Rename and add type annotations to the core samudra model.",
          "timestamp": "2025-07-14T10:15:12-04:00",
          "tree_id": "aa79754e00d8aefa067e3de159cde356a2d9ddb0",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/3683713b49d46850d0f38310c278c7e1b78de68a"
        },
        "date": 1752504044742,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2641438551937585,
            "unit": "iter/sec",
            "range": "stddev: 0.0312502696878527",
            "extra": "mean: 791.0492115999944 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07034175867969182,
            "unit": "iter/sec",
            "range": "stddev: 0.37022395026101806",
            "extra": "mean: 14.21630648380003 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1942839249436438,
            "unit": "iter/sec",
            "range": "stddev: 0.10070330250807862",
            "extra": "mean: 5.1471062276 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.013454822134931247,
            "unit": "iter/sec",
            "range": "stddev: 0.16151178302164781",
            "extra": "mean: 74.32279594419997 sec\nrounds: 5"
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
          "id": "2c6ea98cc5807206b0b2ee84249ab3eb075197b3",
          "message": "The default chunks of `clone_data` should be time=1. (#306)\n\nI just verified this and it looks like in the OSN pod, chunks are\n`time=1`, but by default, we write `time=10` chunks. I want to remedy\nthis so we don't make any accidents when automating are testing setup.",
          "timestamp": "2025-07-14T15:16:21-07:00",
          "tree_id": "e7090b1bcca421369346568cca8089ad091ff662",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/2c6ea98cc5807206b0b2ee84249ab3eb075197b3"
        },
        "date": 1752532900568,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2837439405700637,
            "unit": "iter/sec",
            "range": "stddev: 0.0229138388091571",
            "extra": "mean: 778.9715443999967 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07067450640025408,
            "unit": "iter/sec",
            "range": "stddev: 0.24123765235890765",
            "extra": "mean: 14.149373669999978 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1908897672136177,
            "unit": "iter/sec",
            "range": "stddev: 0.10653017932125851",
            "extra": "mean: 5.2386254883999985 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.013378247621908467,
            "unit": "iter/sec",
            "range": "stddev: 0.06730585842608666",
            "extra": "mean: 74.74820531519998 sec\nrounds: 5"
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
          "id": "227c5a7be10702ef2dca2b285166f53f889d4c94",
          "message": "The default chunks of `clone_data` should be time=1. (#306)\n\nI just verified this and it looks like in the OSN pod, chunks are\n`time=1`, but by default, we write `time=10` chunks. I want to remedy\nthis so we don't make any accidents when automating are testing setup.",
          "timestamp": "2025-07-14T15:16:21-07:00",
          "tree_id": "e7090b1bcca421369346568cca8089ad091ff662",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/227c5a7be10702ef2dca2b285166f53f889d4c94"
        },
        "date": 1752812755886,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.22986991900524,
            "unit": "iter/sec",
            "range": "stddev: 0.02400030602895909",
            "extra": "mean: 813.0941204000123 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.0694194235399118,
            "unit": "iter/sec",
            "range": "stddev: 0.7816636018677365",
            "extra": "mean: 14.405190204799998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1894634233102771,
            "unit": "iter/sec",
            "range": "stddev: 0.07658199694690336",
            "extra": "mean: 5.278063610000004 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.012904567777697816,
            "unit": "iter/sec",
            "range": "stddev: 0.14808009516629886",
            "extra": "mean: 77.49194062339998 sec\nrounds: 5"
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
          "id": "a8eca0d2e569a74291d1df4d6853506871c7904f",
          "message": "Avoid LocalLocation relative footgun (#309)\n\nPreviously, we permitted you to make *relative* local locations as in:\n\n```yaml\ndata_location:\n  type: local\n  path: foo/bar\n```\n\nBut the rule is that if you give a ResolvedLocation (type: local is a\nResolvedLocation) then it's *not* resolved relative to the data_root. So\nthis is actually relative to the cwd rather than data_root. This is just\nconfusing so we are now forbidding relative local locations; instead you\nwant:\n\n```yaml\ndata_location: \"foo/bar\"\n```\n\nThis means we can no longer express \"resolved paths which are relative\nto cwd\" but this seems preferable to having this footgun.\n\n---------\n\nCo-authored-by: Alex Merose <alex@openathena.ai>",
          "timestamp": "2025-07-22T17:39:29Z",
          "tree_id": "fa6e4a01bd0246e61bd7566ccbbf968152db6951",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/a8eca0d2e569a74291d1df4d6853506871c7904f"
        },
        "date": 1753207533048,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.232869093467998,
            "unit": "iter/sec",
            "range": "stddev: 0.017817592005739985",
            "extra": "mean: 811.116123599993 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07085317832130113,
            "unit": "iter/sec",
            "range": "stddev: 0.22808454537843714",
            "extra": "mean: 14.113692902600002 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19188691227290458,
            "unit": "iter/sec",
            "range": "stddev: 0.09430403750004134",
            "extra": "mean: 5.211402842200016 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.013070131872284778,
            "unit": "iter/sec",
            "range": "stddev: 0.12248228254426942",
            "extra": "mean: 76.510322143 sec\nrounds: 5"
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
          "id": "8d93b1e51592819c9c85a1b0d5342545bdb0006f",
          "message": "First pass at setting up one-command training with skypilot. (#298)\n\nWrote entrypoint for performing training runs with skypilot. The `sky`\ncommand is documented within the `train.sky.yaml` file and contributing\nguide.\n\nI did not successfully do a full training run, but that is because I hit\ndata engineering errors (no static_data!).\n\n<details>\n<summary>(Latest error trace related to data engineering)</summary\n\nUltimately, hitting this error seems like a good sign because the error\nis in our code, not in the provisioning system.\n```\nAssertionError: : Static data variable sea_surface_fraction not found in dataStatic data variable sea_surface_fraction not found in data\n```\n\n</details<",
          "timestamp": "2025-07-22T20:53:08Z",
          "tree_id": "6327f17fe130707f862aaa437aea12b22d259e56",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/8d93b1e51592819c9c85a1b0d5342545bdb0006f"
        },
        "date": 1753219198966,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2149770331627956,
            "unit": "iter/sec",
            "range": "stddev: 0.02824251740513975",
            "extra": "mean: 823.0608255999925 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06431722171934344,
            "unit": "iter/sec",
            "range": "stddev: 1.0322193734124974",
            "extra": "mean: 15.547935269400005 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1748169716977023,
            "unit": "iter/sec",
            "range": "stddev: 0.16374379357115573",
            "extra": "mean: 5.720268405800004 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.012583934571242383,
            "unit": "iter/sec",
            "range": "stddev: 4.1569945538398505",
            "extra": "mean: 79.4664017314 sec\nrounds: 5"
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
          "id": "19c72a4b92ea5a55405704f61a9ec41edc804339",
          "message": "Skypilot 0.10.0 just came out!  (#314)\n\nPinning to that stable version instead of depending on nightly.",
          "timestamp": "2025-07-23T14:41:46Z",
          "tree_id": "d5efb931789a4189eac83520a16db5ad34576e0e",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/19c72a4b92ea5a55405704f61a9ec41edc804339"
        },
        "date": 1753283396666,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2757420812700262,
            "unit": "iter/sec",
            "range": "stddev: 0.02916073087140848",
            "extra": "mean: 783.8575011999922 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06642359863142058,
            "unit": "iter/sec",
            "range": "stddev: 0.45203398261267735",
            "extra": "mean: 15.054890439599978 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.18229255870770725,
            "unit": "iter/sec",
            "range": "stddev: 0.10604018430046128",
            "extra": "mean: 5.485687441599998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.011839430397777916,
            "unit": "iter/sec",
            "range": "stddev: 0.2892855371964303",
            "extra": "mean: 84.46352285559996 sec\nrounds: 5"
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
          "id": "5a290621e470ab5f4abdecd9f2d1bbf0c8b19e79",
          "message": "AGENTS.md (#316)\n\nI keep wanting to put things in here so made a first cut at it so we can\ngrow it over time.\n\n---------\n\nCo-authored-by: Alex Merose <alex@openathena.ai>",
          "timestamp": "2025-07-26T13:03:11Z",
          "tree_id": "93de5b5c63179f0e0e9007484dbe1043bbe90816",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/5a290621e470ab5f4abdecd9f2d1bbf0c8b19e79"
        },
        "date": 1753536513865,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2694256371589177,
            "unit": "iter/sec",
            "range": "stddev: 0.024929906337208436",
            "extra": "mean: 787.7578415999892 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06958313353454455,
            "unit": "iter/sec",
            "range": "stddev: 0.26273496268288066",
            "extra": "mean: 14.371298750200003 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19122238070112274,
            "unit": "iter/sec",
            "range": "stddev: 0.09274693412386541",
            "extra": "mean: 5.229513388200007 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.013384509530338122,
            "unit": "iter/sec",
            "range": "stddev: 0.14353258130254698",
            "extra": "mean: 74.7132345592 sec\nrounds: 5"
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
          "id": "22812c426eebfcc158fcd32d4529ce25a3092742",
          "message": "Hopefully fix data race in cache reading/writing (#318)\n\nBreaks out the test change from #313 and fixes the bug (see eg [this\nactions\nrun](https://github.com/Open-Athena/Ocean_Emulator/actions/runs/16573115391/job/46880287790))\nthat exposed.\n\nWe now ensure that only one process (of possibly-many pytest processes)\nwrite to a given data source cache at once. We also ensure that\nprocesses that would like to `open` a given cache wait for any writers\nof that cache to finish. (They also wait for other readers, though this\nis unnecessary.) This locking is also hoisted upwards such that only one\nprocess will try to create the remote/compact/mock data in the case when\nit is not cached, which should hopefully reduce the chance of that\nfailing.\n\nFixes #257",
          "timestamp": "2025-07-29T14:41:42Z",
          "tree_id": "7c74980a4e88603a4e838fe3bdd4b64fe75ea52b",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/22812c426eebfcc158fcd32d4529ce25a3092742"
        },
        "date": 1753801740939,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2501478853064825,
            "unit": "iter/sec",
            "range": "stddev: 0.0401749976141187",
            "extra": "mean: 799.9053645999993 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06898865419756849,
            "unit": "iter/sec",
            "range": "stddev: 0.23574103023742624",
            "extra": "mean: 14.495137086400002 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.18561344951610215,
            "unit": "iter/sec",
            "range": "stddev: 0.11194285093598426",
            "extra": "mean: 5.387540625999998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.012397376901096654,
            "unit": "iter/sec",
            "range": "stddev: 0.5263348664330443",
            "extra": "mean: 80.66222459619999 sec\nrounds: 5"
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
          "id": "fca69bcf55ee435edceb52595536b733242cfebc",
          "message": "Fix EMA state restore when resuming training from a checkpoint (#313)\n\nThis fixes a silly bug which overwrote all EMA state after resuming from\na checkpoint. It also now declares the two kinds of checkpoints to be\nfor different uses:\n\n* EMA checkpoints: these bake in the averaged weights into the model\nitself, and so are useful for inference that wants to start from that\naveraged model.\n* Non-EMA checkpoints: these now save the averaged model weights\nseparately from the main model weights so we can resume training without\nstarting that averaging over again.\n\nThese maybe could be renamed \"inference checkpoints\" and \"training\ncheckpoints\" or something if this turns out to be a useful distinction.\n\nI ended up not adding a test that training continues exactly as it would\nhave without checkpointing/restoring because I think there are other\nbugs that are not critical to solve right now like rand seeds not being\nsaved. (I started a [discussion\nhere](https://openathena.slack.com/archives/C0884476QSC/p1753229734048809))\n\nFixes #282",
          "timestamp": "2025-07-29T15:00:44Z",
          "tree_id": "284b778022da90fdfb58e7ec80b48727fb5e176c",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/fca69bcf55ee435edceb52595536b733242cfebc"
        },
        "date": 1753802809279,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2493053946334087,
            "unit": "iter/sec",
            "range": "stddev: 0.03211125570025654",
            "extra": "mean: 800.4447945999914 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06881988463672949,
            "unit": "iter/sec",
            "range": "stddev: 0.39192583369116374",
            "extra": "mean: 14.530684049799982 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.18905205655206694,
            "unit": "iter/sec",
            "range": "stddev: 0.07249380584320339",
            "extra": "mean: 5.289548382800001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.013039170384282029,
            "unit": "iter/sec",
            "range": "stddev: 0.17136603733218406",
            "extra": "mean: 76.69199577340001 sec\nrounds: 5"
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
          "id": "ac92b9abf9c99815dae6b9ae91027ea6ce7ec05f",
          "message": "Fix UnresolvedLocation JSON schema to permit string (#325)\n\nThis was previously causing red squigglies in VS Code. It's annoying\nthat this annotation and the two `@model_*` annotations on this class\nare all required -- I spent some time looking for a more direct way to\nexpress \"this type serializes to/from a string\" in the pydantic docs but\neverything I tried was broken or more complicated than this.",
          "timestamp": "2025-08-04T12:04:10-04:00",
          "tree_id": "27307cf6f78e7b02e3b4a73430ebd435dbfd791b",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/ac92b9abf9c99815dae6b9ae91027ea6ce7ec05f"
        },
        "date": 1754325024091,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2438566978082914,
            "unit": "iter/sec",
            "range": "stddev: 0.02815242649124351",
            "extra": "mean: 803.9511318000109 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06998520205327614,
            "unit": "iter/sec",
            "range": "stddev: 0.3670058221912347",
            "extra": "mean: 14.288734913399997 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.18521078694772086,
            "unit": "iter/sec",
            "range": "stddev: 0.1783613763460729",
            "extra": "mean: 5.39925355580001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.013075527775960286,
            "unit": "iter/sec",
            "range": "stddev: 0.20167642142228218",
            "extra": "mean: 76.47874847840004 sec\nrounds: 5"
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
          "id": "97147185c31a4de7192cbd74e1de485391eb43c2",
          "message": "MSE Dynamic Loss (#310)\n\nAdds a new loss function to dynamically re-weight loss scaling to make\neach channel contribute equally to the loss.",
          "timestamp": "2025-08-04T19:22:39Z",
          "tree_id": "37dcb9e3e141071f2da6809e342144d2d2897d3c",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/97147185c31a4de7192cbd74e1de485391eb43c2"
        },
        "date": 1754336954371,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.305461744240566,
            "unit": "iter/sec",
            "range": "stddev: 0.02884408662970903",
            "extra": "mean: 766.012489000002 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.0701210166175281,
            "unit": "iter/sec",
            "range": "stddev: 0.3081279292277422",
            "extra": "mean: 14.261059640000008 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.18322672605458803,
            "unit": "iter/sec",
            "range": "stddev: 0.25838037325061897",
            "extra": "mean: 5.4577190868 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.013014275606498956,
            "unit": "iter/sec",
            "range": "stddev: 0.09620982025994937",
            "extra": "mean: 76.83869853659998 sec\nrounds: 5"
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
          "id": "624a32a1a10d4e7fd3330079f6d8d99aff013a03",
          "message": "Sky Train: Cloning data faster via setup section instead of file mount (#319)\n\nSince `file_mount` options don't let you pass arguments to rclone, this\nPR opts to manually `rclone` data during the `setup` stage of the job.\nThis allows us to make faster file transfers with the `--transfers`\nflag.\n\nIt also has required that I map the `data/` dir to home and not root. \nWe also set the default number of workers to 4.",
          "timestamp": "2025-08-04T20:58:11Z",
          "tree_id": "6729e53b5adf4b38607cec411fd3be8a77c44535",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/624a32a1a10d4e7fd3330079f6d8d99aff013a03"
        },
        "date": 1754342606687,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2645068867628333,
            "unit": "iter/sec",
            "range": "stddev: 0.03030718161384631",
            "extra": "mean: 790.8221066000067 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07135583208539337,
            "unit": "iter/sec",
            "range": "stddev: 0.4866418597419702",
            "extra": "mean: 14.01427144459999 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19271075284353342,
            "unit": "iter/sec",
            "range": "stddev: 0.10854949240367774",
            "extra": "mean: 5.189124038200009 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.013335990461259157,
            "unit": "iter/sec",
            "range": "stddev: 0.2973168510599323",
            "extra": "mean: 74.98505663339999 sec\nrounds: 5"
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
          "id": "43b541756668c9062e25026177d197b7c9941114",
          "message": "Fix initial ema creation in distributed mode (#326)",
          "timestamp": "2025-08-04T23:28:10Z",
          "tree_id": "fe07c806709f894fe23173b0a224c3375672af53",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/43b541756668c9062e25026177d197b7c9941114"
        },
        "date": 1754351614567,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.274365567953231,
            "unit": "iter/sec",
            "range": "stddev: 0.027471123454690147",
            "extra": "mean: 784.7041893999915 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07040381332610124,
            "unit": "iter/sec",
            "range": "stddev: 0.6619732008225289",
            "extra": "mean: 14.203776084799994 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19316267778558519,
            "unit": "iter/sec",
            "range": "stddev: 0.08772517063644016",
            "extra": "mean: 5.176983522199987 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.013324467828204048,
            "unit": "iter/sec",
            "range": "stddev: 0.07370883812669014",
            "extra": "mean: 75.04990164660002 sec\nrounds: 5"
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
          "id": "3904f8c8e5eb19e0cc0c9db7f3cf4dc568c0391f",
          "message": "Standard deviations should be stds, not means. 😬 (#327)",
          "timestamp": "2025-08-05T14:56:15Z",
          "tree_id": "11793635557474a6b941fe4b5717a5f6ab4e589a",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/3904f8c8e5eb19e0cc0c9db7f3cf4dc568c0391f"
        },
        "date": 1754407410022,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2453559025185177,
            "unit": "iter/sec",
            "range": "stddev: 0.02414274248151024",
            "extra": "mean: 802.9833061999966 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06768094857763071,
            "unit": "iter/sec",
            "range": "stddev: 0.6918857122298251",
            "extra": "mean: 14.775206627799998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1856147582461778,
            "unit": "iter/sec",
            "range": "stddev: 0.12778560991826757",
            "extra": "mean: 5.387502639600006 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.012259393721245009,
            "unit": "iter/sec",
            "range": "stddev: 0.14689436365552955",
            "extra": "mean: 81.5701023018 sec\nrounds: 5"
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
          "id": "f4adaeacda884121f7b1fa9ca7c54dad16743346",
          "message": "Add .git to .gitignore for the sake of skypilot (#328)\n\nWithout this, it seems that skypilot attempts to upload the whole .git\nfolder. Adding this makes it skip doing so. To me, this is a bug in\nskypilot; I'll file an issue over there and use this as a workaround for\nthe time being.",
          "timestamp": "2025-08-05T08:28:55-07:00",
          "tree_id": "a620d42ea51ca5eaf5f8306da3c022ea0cc09662",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/f4adaeacda884121f7b1fa9ca7c54dad16743346"
        },
        "date": 1754409297946,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.266312971676194,
            "unit": "iter/sec",
            "range": "stddev: 0.026765102380862972",
            "extra": "mean: 789.694192799999 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06931073405674053,
            "unit": "iter/sec",
            "range": "stddev: 0.4495548011038106",
            "extra": "mean: 14.427779673800023 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.18990276599449046,
            "unit": "iter/sec",
            "range": "stddev: 0.11525138277210392",
            "extra": "mean: 5.265852736600015 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.013244027854802746,
            "unit": "iter/sec",
            "range": "stddev: 0.08568033523496131",
            "extra": "mean: 75.50573065559999 sec\nrounds: 5"
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
          "id": "3b96a83248b45b68304764e0a2122b54535f86d3",
          "message": "Stop complaining about long lines (#333)\n\nThis rule seems low-value to me, and we already enforce `ruff format`\nwhich will wrap lines when reasonable. If ruff format is OK with the\nline length I don't think we should be wrapping things by hand.",
          "timestamp": "2025-08-06T12:57:24-04:00",
          "tree_id": "062a5871b5796fbe6a941a083e3934e5caf5fd67",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/3b96a83248b45b68304764e0a2122b54535f86d3"
        },
        "date": 1754501034466,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2673108062955551,
            "unit": "iter/sec",
            "range": "stddev: 0.031342933901319925",
            "extra": "mean: 789.072416199997 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.0683988793532257,
            "unit": "iter/sec",
            "range": "stddev: 0.8111530612073059",
            "extra": "mean: 14.620122573000021 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.18862455766396724,
            "unit": "iter/sec",
            "range": "stddev: 0.10580453298438187",
            "extra": "mean: 5.3015366206000065 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.012635649229365435,
            "unit": "iter/sec",
            "range": "stddev: 0.694854040770107",
            "extra": "mean: 79.14116495700002 sec\nrounds: 5"
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
          "id": "c9b025a34d85f91d65aca4724a6c0a8aa41a1d65",
          "message": "By default, our training runs should use A100 machines with high GPU memory (#332)",
          "timestamp": "2025-08-06T17:14:28Z",
          "tree_id": "05b48b967dca12d7516baea586ef0aaaffe36035",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/c9b025a34d85f91d65aca4724a6c0a8aa41a1d65"
        },
        "date": 1754502093926,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2308008465209943,
            "unit": "iter/sec",
            "range": "stddev: 0.05009152802836502",
            "extra": "mean: 812.4791291999998 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06806758018718016,
            "unit": "iter/sec",
            "range": "stddev: 1.047111870399672",
            "extra": "mean: 14.69128177099999 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.18291742779210438,
            "unit": "iter/sec",
            "range": "stddev: 0.2606676620301233",
            "extra": "mean: 5.466947639000011 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.012391548610060303,
            "unit": "iter/sec",
            "range": "stddev: 0.15119792510262484",
            "extra": "mean: 80.70016359280001 sec\nrounds: 5"
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
          "id": "ba0c18f9979fcd0f7143e09d7179035e2a693744",
          "message": "Work around ema name issues (#335)\n\nI would love a more principled strategy here, but this:\n* Ensures that checkpoints which include ema params are rewritten on\nload to not have a \"module\" prefix on those params\n* Ensures that EmaTrackers always store parameters without a \"module\"\nprefix.\n* Ensures that using an EmaTracker works either with or without a\n\"module.\" prefix on the requested parameters.\n\nCloses #329",
          "timestamp": "2025-08-08T14:11:52-04:00",
          "tree_id": "67ba61c66ed69a2fe95c3d1aeb4a8fcdbbcb36dd",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/ba0c18f9979fcd0f7143e09d7179035e2a693744"
        },
        "date": 1754678265234,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.259721512463275,
            "unit": "iter/sec",
            "range": "stddev: 0.034886757749258014",
            "extra": "mean: 793.8262466000026 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06996510689680083,
            "unit": "iter/sec",
            "range": "stddev: 0.7313794526213525",
            "extra": "mean: 14.292838878600003 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19251923310003108,
            "unit": "iter/sec",
            "range": "stddev: 0.09550950295121247",
            "extra": "mean: 5.194286222199992 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.012856013614358258,
            "unit": "iter/sec",
            "range": "stddev: 0.17918772697627341",
            "extra": "mean: 77.78460959960003 sec\nrounds: 5"
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
          "id": "67dcd0676a9a6bef712c6d29ee0abeb6220c98b7",
          "message": "Add a scheduler which can run a cosine schedule followed by a flat tail (#334)\n\nTo try extending the behavior seen greatly improving loss at low\nlearning rates from [this\nrun](https://openathena.slack.com/archives/C08CYM42DT3/p1753308638843249?thread_ts=1752275713.570969&cid=C08CYM42DT3).",
          "timestamp": "2025-08-08T18:19:36Z",
          "tree_id": "5aef0b8e92659742f4e1c09d7a901ae43089f905",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/67dcd0676a9a6bef712c6d29ee0abeb6220c98b7"
        },
        "date": 1754678704939,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2972799664223447,
            "unit": "iter/sec",
            "range": "stddev: 0.02088998883185652",
            "extra": "mean: 770.843631199989 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07273211779713577,
            "unit": "iter/sec",
            "range": "stddev: 0.34575121551448756",
            "extra": "mean: 13.749084040000008 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.18582482438250822,
            "unit": "iter/sec",
            "range": "stddev: 0.1423458553849846",
            "extra": "mean: 5.381412323799998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.013546098233727759,
            "unit": "iter/sec",
            "range": "stddev: 0.18081139360215046",
            "extra": "mean: 73.82199528939998 sec\nrounds: 5"
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
          "id": "6e5ba43915107283105350114a69966f932e3645",
          "message": "Un-ignoring .git folder for skypilot syncs (#338)\n\nI think this could address #337.",
          "timestamp": "2025-08-08T18:34:59Z",
          "tree_id": "83455ec4edd235c794aa4ce54c82240c15c974c4",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/6e5ba43915107283105350114a69966f932e3645"
        },
        "date": 1754679638108,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2282794247766138,
            "unit": "iter/sec",
            "range": "stddev: 0.031035255582474382",
            "extra": "mean: 814.1469928000049 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07035775881751544,
            "unit": "iter/sec",
            "range": "stddev: 0.24727426305008451",
            "extra": "mean: 14.213073537399998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19053900935114101,
            "unit": "iter/sec",
            "range": "stddev: 0.16602617309553583",
            "extra": "mean: 5.248269125599984 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.013185562979790644,
            "unit": "iter/sec",
            "range": "stddev: 0.35001505717680925",
            "extra": "mean: 75.84052357360001 sec\nrounds: 5"
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
          "id": "dd3d28bffaf1a5b4a46e3d5b6288583afc0207fb",
          "message": "Avoid chunking lat/lon (#342)",
          "timestamp": "2025-08-11T17:54:37-04:00",
          "tree_id": "260baac65f6e988c4442f12409b08abb08b08c64",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/dd3d28bffaf1a5b4a46e3d5b6288583afc0207fb"
        },
        "date": 1754950844235,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.247574797557253,
            "unit": "iter/sec",
            "range": "stddev: 0.024377439891637067",
            "extra": "mean: 801.5551468000126 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07067153091219096,
            "unit": "iter/sec",
            "range": "stddev: 0.7392577343757167",
            "extra": "mean: 14.14996940200001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19309283597775778,
            "unit": "iter/sec",
            "range": "stddev: 0.09231335742760768",
            "extra": "mean: 5.178856040600022 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.013335916337060645,
            "unit": "iter/sec",
            "range": "stddev: 0.15885132821764505",
            "extra": "mean: 74.98547341819999 sec\nrounds: 5"
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
          "id": "ea4fb9343b5b9c2fb5da5fd9789fefc7c818c995",
          "message": "Vizualization code (#339)\n\nThis is an import and reorganization of the code from [this\nnotebook](https://github.com/Open-Athena/Ocean_Emulator/blob/fixed-branch/Notebooks/notebooks/2025-04-08-Samudra_Eval_OM4.ipynb).\nThe goal was to get it to a state we could check it in and iterate from\nthere. This PR:\n\n* Moves that code from a notebook to python code in\nsrc/ocean_emulators/viz/core.py\n* Creates a new TopLevelConfig for visualization in\nsrc/ocean_emulators/viz/config.py (see CONTRIBUTING.md for an example\ninvocation)\n* Does a minimal amount of deduplication, cleanup, and typing of the\ncode there to get it passing CI and to allow the config to configure all\nthe main knobs. Also added some TODOs about clear things we might want\nto do next.\n* Includes a tool to compare old and new viz output -- this is how I\nverified that the previous notebook and this code produce nearly\nidentical results (modulo some FP error due to slight changes in\nordering, I think).\n* Includes pointers to the basin data, now in the emulator pod, which\nwas created by the new `notebooks/regrid_basins.py` code, which I\nincluded for posterity but don't think we really need to maintain per\nse.\n\nI would not try to review core.py too deeply but skimming the structure\nis good; same with notebooks/regrid_basins.py. Everything else is fair\ngame :)",
          "timestamp": "2025-08-12T13:09:11-04:00",
          "tree_id": "bbf84611af74016865ffa2229aea8022e132b186",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/ea4fb9343b5b9c2fb5da5fd9789fefc7c818c995"
        },
        "date": 1755020177469,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.267660098943949,
            "unit": "iter/sec",
            "range": "stddev: 0.01533733105776233",
            "extra": "mean: 788.8549941999997 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06993248855154394,
            "unit": "iter/sec",
            "range": "stddev: 0.35970136809217257",
            "extra": "mean: 14.29950543320001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.18405103633906586,
            "unit": "iter/sec",
            "range": "stddev: 0.19343129763615163",
            "extra": "mean: 5.433275573399987 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.012549580130180066,
            "unit": "iter/sec",
            "range": "stddev: 1.532887148081376",
            "extra": "mean: 79.6839407874 sec\nrounds: 5"
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
          "id": "34e217b5b958a2eedad46369abe00a8c643092ff",
          "message": "New baseline model: Wide Samudra (training and eval config) (#341)\n\nSince I'm about to fork off experiments from these configurations, I\nthought it should be merged into main.\n\n---------\n\nCo-authored-by: Jesse Rusak <jesse@openathena.ai>",
          "timestamp": "2025-08-12T17:45:25Z",
          "tree_id": "bae84318a9d1acd821c6ffb35a2a7e2829d09028",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/34e217b5b958a2eedad46369abe00a8c643092ff"
        },
        "date": 1755022317815,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2732303172750543,
            "unit": "iter/sec",
            "range": "stddev: 0.020223650416903428",
            "extra": "mean: 785.4038554000056 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06997879765453789,
            "unit": "iter/sec",
            "range": "stddev: 0.3363235301090989",
            "extra": "mean: 14.290042606000009 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19235362356387206,
            "unit": "iter/sec",
            "range": "stddev: 0.10529459108314856",
            "extra": "mean: 5.198758315399994 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.012905661255035596,
            "unit": "iter/sec",
            "range": "stddev: 0.2657167367896565",
            "extra": "mean: 77.48537484740001 sec\nrounds: 5"
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
          "id": "e4e5756144a42c22b2cd0043854da28cad0bd44d",
          "message": "minor tweaks to skypilot setup (#348)\n\n* Fixes numba bug with print being defined inside a function rather than\na module.\n* Fixes sqlalchemy required by skypilot\n* Some docs and convenience updates",
          "timestamp": "2025-08-14T14:19:31-07:00",
          "tree_id": "90bf8f2e0400125696c85234419ae9622f042201",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/e4e5756144a42c22b2cd0043854da28cad0bd44d"
        },
        "date": 1755207954958,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2866634631594256,
            "unit": "iter/sec",
            "range": "stddev: 0.031795665992096496",
            "extra": "mean: 777.2040075999996 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06974341112749494,
            "unit": "iter/sec",
            "range": "stddev: 0.47886562345454436",
            "extra": "mean: 14.338272014999996 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.18824883288866448,
            "unit": "iter/sec",
            "range": "stddev: 0.21120030883571803",
            "extra": "mean: 5.312117927399993 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.01278732655334228,
            "unit": "iter/sec",
            "range": "stddev: 0.11645410389140541",
            "extra": "mean: 78.20242924340002 sec\nrounds: 5"
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
          "id": "4a73c3e4b7a5792158ecb87170efc3bfdc6d5f5a",
          "message": "Tweak AGENTS.md for some common annoyances (#349)",
          "timestamp": "2025-08-21T13:42:57-04:00",
          "tree_id": "ccdb95535f057b4c7dc696cb521358936914481a",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/4a73c3e4b7a5792158ecb87170efc3bfdc6d5f5a"
        },
        "date": 1755799804077,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.3135440051200775,
            "unit": "iter/sec",
            "range": "stddev: 0.01790865343419854",
            "extra": "mean: 761.2991997999984 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07161299278209528,
            "unit": "iter/sec",
            "range": "stddev: 0.327884874016702",
            "extra": "mean: 13.963946501199995 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1862194331573269,
            "unit": "iter/sec",
            "range": "stddev: 0.22352806474081985",
            "extra": "mean: 5.370008827999993 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.012654116698482813,
            "unit": "iter/sec",
            "range": "stddev: 0.1548767026505357",
            "extra": "mean: 79.02566602059997 sec\nrounds: 5"
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
          "id": "70b29b97f87db3a89dc0ec107edbfaec772474c4",
          "message": "Eval + Viz Skypilot (#352)",
          "timestamp": "2025-08-26T19:55:18Z",
          "tree_id": "0ab45c6cbe943e2753d4f33103262119ce0c2f7e",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/70b29b97f87db3a89dc0ec107edbfaec772474c4"
        },
        "date": 1756239780482,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.246567026087767,
            "unit": "iter/sec",
            "range": "stddev: 0.04583503363158478",
            "extra": "mean: 802.2031539999944 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06720597251219441,
            "unit": "iter/sec",
            "range": "stddev: 0.7473862934130544",
            "extra": "mean: 14.879629929000009 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.17702922583823927,
            "unit": "iter/sec",
            "range": "stddev: 0.12672407689629178",
            "extra": "mean: 5.648784799599992 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.011946166650068344,
            "unit": "iter/sec",
            "range": "stddev: 0.4740221263922304",
            "extra": "mean: 83.70886069920002 sec\nrounds: 5"
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
          "id": "41564666409ad63c3b60609a6ca4713390e9b0a8",
          "message": "Implementing Warmup + Cosine Decay LR schedule from ConvNeXt paper. (#343)\n\nHere is a Warmup implementation. This was added to match the LR schedule\nof the original ConvNeXt paper. Following the citations, warmup is\ntheorhetical grounded and empirically verified to help the loss curves\nof networks with _large batch sizes_ -- see this paper:\nhttps://arxiv.org/pdf/1706.02677\n\nWe have very small batch sizes, though each sample is fairly large (~80\nMB). This technique is not likely to lead to a modeling improvement;\nhowever, I've queued up a few jobs to test if this improves anything.\nWhile this might not make it into the final baseline Samudra config, I\ncould see it being valuable for the next iteration of the model.\n\n---------\n\nCo-authored-by: Jesse Rusak <jesse@openathena.ai>",
          "timestamp": "2025-08-26T23:23:38Z",
          "tree_id": "38a4a731e25139e1ad166a25ede9cb543eee09bd",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/41564666409ad63c3b60609a6ca4713390e9b0a8"
        },
        "date": 1756252246085,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2847879720140074,
            "unit": "iter/sec",
            "range": "stddev: 0.027197973823160578",
            "extra": "mean: 778.3385443999919 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06757707540364018,
            "unit": "iter/sec",
            "range": "stddev: 0.6411075744872337",
            "extra": "mean: 14.797917696600006 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.18124901281297834,
            "unit": "iter/sec",
            "range": "stddev: 0.11900070176557843",
            "extra": "mean: 5.5172714294 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.012187453112426576,
            "unit": "iter/sec",
            "range": "stddev: 0.19901788418466237",
            "extra": "mean: 82.051597719 sec\nrounds: 5"
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
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/ba1d64318fd3545b2e2d0ae6ff6efb6acbf8ca25"
        },
        "date": 1744741567380,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.20338158685571298,
            "unit": "iter/sec",
            "range": "stddev: 0.3223892422200483",
            "extra": "mean: 4.916865953600018 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.051953912634401356,
            "unit": "iter/sec",
            "range": "stddev: 1.5616442555274344",
            "extra": "mean: 19.24782849440003 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.13046192235055382,
            "unit": "iter/sec",
            "range": "stddev: 0.08087941876395714",
            "extra": "mean: 7.665071784800011 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
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
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/6f54e2a17d4fffe6740ef6e59a0b382def3304aa"
        },
        "date": 1744825757559,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19162669132517887,
            "unit": "iter/sec",
            "range": "stddev: 0.10918733338717698",
            "extra": "mean: 5.2184797070000055 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.05086815807555744,
            "unit": "iter/sec",
            "range": "stddev: 1.7424044651393846",
            "extra": "mean: 19.658663451400024 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.12003941579973994,
            "unit": "iter/sec",
            "range": "stddev: 0.08738583942767349",
            "extra": "mean: 8.330597023799964 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
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
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/47cef09247ac73131582226ffa3dd7e94d65c391"
        },
        "date": 1744841614038,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19602406742600897,
            "unit": "iter/sec",
            "range": "stddev: 0.5641500895020005",
            "extra": "mean: 5.101414398400129 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.0558080488740754,
            "unit": "iter/sec",
            "range": "stddev: 0.4908260241764432",
            "extra": "mean: 17.918562289400008 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1310450438404766,
            "unit": "iter/sec",
            "range": "stddev: 0.07947100916054976",
            "extra": "mean: 7.630963909000002 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
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
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/ef402bd05adab5e866d9c2383b2a76f4672314fb"
        },
        "date": 1744842289381,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19794353420320093,
            "unit": "iter/sec",
            "range": "stddev: 0.4227218242111661",
            "extra": "mean: 5.0519457683999915 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.05553368815266424,
            "unit": "iter/sec",
            "range": "stddev: 0.8786947570319963",
            "extra": "mean: 18.00708782839997 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.13076027836615073,
            "unit": "iter/sec",
            "range": "stddev: 0.07920218762702606",
            "extra": "mean: 7.647582373599971 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
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
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/dcc4fa780eeb21762ddae07405a9571a68a6f6dd"
        },
        "date": 1744843008780,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.20446385722920238,
            "unit": "iter/sec",
            "range": "stddev: 0.3901619059730472",
            "extra": "mean: 4.890839943799984 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.05639689149006624,
            "unit": "iter/sec",
            "range": "stddev: 1.1542145203368046",
            "extra": "mean: 17.73147373159991 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.13244609717670677,
            "unit": "iter/sec",
            "range": "stddev: 0.09207346499483049",
            "extra": "mean: 7.550241353399952 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
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
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/85f6f6ada579d695c859c5f033866a73f81f8ec7"
        },
        "date": 1744843902859,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1789094861475476,
            "unit": "iter/sec",
            "range": "stddev: 0.4646268461279602",
            "extra": "mean: 5.589418546400021 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.05673692551644761,
            "unit": "iter/sec",
            "range": "stddev: 0.45695104923921664",
            "extra": "mean: 17.62520599939994 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1308310652317681,
            "unit": "iter/sec",
            "range": "stddev: 0.11053640462482019",
            "extra": "mean: 7.643444607200081 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
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
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/3e4ab9c74cb22e2ecb2670a4717dcd5a3536f539"
        },
        "date": 1744846510084,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.16728412420165711,
            "unit": "iter/sec",
            "range": "stddev: 0.5676932850266345",
            "extra": "mean: 5.977853575600056 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.04965872010590991,
            "unit": "iter/sec",
            "range": "stddev: 0.47559197609353715",
            "extra": "mean: 20.137450136999995 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1155716050622202,
            "unit": "iter/sec",
            "range": "stddev: 0.10838009836882366",
            "extra": "mean: 8.652644388399995 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
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
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/0ac9469629781b19963f3eefea74a95a4fa3e8ed"
        },
        "date": 1745012616840,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.17233357942867203,
            "unit": "iter/sec",
            "range": "stddev: 0.34521507991512745",
            "extra": "mean: 5.802699643999995 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.05263641488806646,
            "unit": "iter/sec",
            "range": "stddev: 1.1898318175205047",
            "extra": "mean: 18.998254385800056 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.12426730387133196,
            "unit": "iter/sec",
            "range": "stddev: 0.14665182680747468",
            "extra": "mean: 8.047169036800005 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
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
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/499db56063fe946f79b37da5b8241f484d02e0f8"
        },
        "date": 1745018119322,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.16793746513389923,
            "unit": "iter/sec",
            "range": "stddev: 0.20882389939968413",
            "extra": "mean: 5.9545974402000414 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.050189165595039886,
            "unit": "iter/sec",
            "range": "stddev: 0.5756678637170669",
            "extra": "mean: 19.92461895200004 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1168138454371749,
            "unit": "iter/sec",
            "range": "stddev: 0.10499232790857435",
            "extra": "mean: 8.560629061199961 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
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
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/e01e28c1a414b86e56e2c3ccd3e3aa14a20727cc"
        },
        "date": 1745335295298,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.18124641637637115,
            "unit": "iter/sec",
            "range": "stddev: 0.3853196558960974",
            "extra": "mean: 5.517350466800008 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.046593358132185375,
            "unit": "iter/sec",
            "range": "stddev: 0.5203017024190717",
            "extra": "mean: 21.462286473599942 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.12921717474761257,
            "unit": "iter/sec",
            "range": "stddev: 0.10748568602331393",
            "extra": "mean: 7.738909335800008 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
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
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/f2cabb570ca994ac0c334b5149ff02d8e532c117"
        },
        "date": 1745345479244,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1754112183636589,
            "unit": "iter/sec",
            "range": "stddev: 0.31408560239348116",
            "extra": "mean: 5.7008896542 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.045929750202792004,
            "unit": "iter/sec",
            "range": "stddev: 0.8285893108004153",
            "extra": "mean: 21.772380550400023 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.13068887267657436,
            "unit": "iter/sec",
            "range": "stddev: 0.06842474896896379",
            "extra": "mean: 7.651760853999986 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
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
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/baf745e6863bbf4fa749a2457a0661fc6a582fc2"
        },
        "date": 1745369928806,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.16288531091413397,
            "unit": "iter/sec",
            "range": "stddev: 0.4658991770640878",
            "extra": "mean: 6.139289014999986 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.04353155356395057,
            "unit": "iter/sec",
            "range": "stddev: 0.4734099643169331",
            "extra": "mean: 22.971842677999938 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.12426601253878036,
            "unit": "iter/sec",
            "range": "stddev: 0.03529023026743787",
            "extra": "mean: 8.0472526604 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
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
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/212d9da3b3954bee74342a10420a25a13cd6bd8b"
        },
        "date": 1745512822029,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.16971433884005255,
            "unit": "iter/sec",
            "range": "stddev: 0.6084305091417417",
            "extra": "mean: 5.892254047799997 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.04481527072648967,
            "unit": "iter/sec",
            "range": "stddev: 0.21532675469841206",
            "extra": "mean: 22.31382258299991 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.13019361495194376,
            "unit": "iter/sec",
            "range": "stddev: 0.11370170925102274",
            "extra": "mean: 7.68086822360001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
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
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/b929596c1d476649860758ddc80745891dbe94fd"
        },
        "date": 1745618591255,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.17107528645875836,
            "unit": "iter/sec",
            "range": "stddev: 0.29876999550321814",
            "extra": "mean: 5.845379661200059 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.04398548861742908,
            "unit": "iter/sec",
            "range": "stddev: 1.069544527025251",
            "extra": "mean: 22.734770749000017 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.12624328795872547,
            "unit": "iter/sec",
            "range": "stddev: 0.06917877265987145",
            "extra": "mean: 7.921213207999972 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
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
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/a7a42c83b3ed6c3b7feb97d9000a8cd33b0092d3"
        },
        "date": 1745619526863,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.18150736153021071,
            "unit": "iter/sec",
            "range": "stddev: 0.4906494179828221",
            "extra": "mean: 5.5094184145999865 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.04553285872505972,
            "unit": "iter/sec",
            "range": "stddev: 0.4669205771994029",
            "extra": "mean: 21.962161568600003 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.12907468063175684,
            "unit": "iter/sec",
            "range": "stddev: 0.09524987160459554",
            "extra": "mean: 7.747452831999999 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
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
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/c2fbe9bf215d428679dcfaf05f23d68ae70388fe"
        },
        "date": 1745685475878,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.18206885519125104,
            "unit": "iter/sec",
            "range": "stddev: 0.41243211061886226",
            "extra": "mean: 5.492427570599967 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.04625795882520392,
            "unit": "iter/sec",
            "range": "stddev: 0.6931535498481457",
            "extra": "mean: 21.61790155460003 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.12999669250923046,
            "unit": "iter/sec",
            "range": "stddev: 0.03703700358141157",
            "extra": "mean: 7.692503406800097 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.010545724307978387,
            "unit": "iter/sec",
            "range": "stddev: 0.42191740339431894",
            "extra": "mean: 94.82516049120004 sec\nrounds: 5"
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
          "id": "17d4187244ffb5dd8f45d62cdb0d570796c98770",
          "message": "[Clean PR] Ocean Heat Corrector  (#224)\n\nThis PR replaces #165\n\n---------\n\nCo-authored-by: Alex Merose <alex@openathena.ai>\nCo-authored-by: Jesse Rusak <jesse@openathena.ai>",
          "timestamp": "2025-04-26T13:09:51-04:00",
          "tree_id": "3603d044dd950eb742f66f22e5db01eff53aba9f",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/17d4187244ffb5dd8f45d62cdb0d570796c98770"
        },
        "date": 1745690016004,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.17442742765085914,
            "unit": "iter/sec",
            "range": "stddev: 0.3446935486717081",
            "extra": "mean: 5.7330433262000495 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.0457889387524756,
            "unit": "iter/sec",
            "range": "stddev: 0.7631732538016524",
            "extra": "mean: 21.83933559599991 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.12642856614698264,
            "unit": "iter/sec",
            "range": "stddev: 0.1729663934073463",
            "extra": "mean: 7.909604850199957 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.009844761506184392,
            "unit": "iter/sec",
            "range": "stddev: 0.6446493894114709",
            "extra": "mean: 101.57686393639997 sec\nrounds: 5"
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
          "id": "4a7b5a01e1e7add44b9f5b671c22e9a9173b4bef",
          "message": "Fix wandb nested config logging (#236)\n\nThe config system broke the nice logging of nested config structures to\nwandb. For example, see the \"data\" key in this run:\n\nhttps://wandb.ai/m2lines/ocean-emulators/runs/2i57cuzt/overview\n\n<img width=\"667\" alt=\"Screenshot 2025-04-29 at 11 21 59 AM\"\nsrc=\"https://github.com/user-attachments/assets/9fa91699-bb21-4d28-882a-87465efc87ac\"\n/>\n\n\nThis PR uses pydantic serialization to correctly output nested keys, for\nexample this run:\n\nhttps://wandb.ai/m2lines/ocean-emulators/runs/twx4knvp/overview\n\n<img width=\"572\" alt=\"Screenshot 2025-04-29 at 11 22 19 AM\"\nsrc=\"https://github.com/user-attachments/assets/574147ee-02fb-4198-b57f-43e756d3b51e\"\n/>\n\nFixes #228",
          "timestamp": "2025-04-29T15:53:52-04:00",
          "tree_id": "0ebf59a0a078cdac0df28657f899e83a587c30c0",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/4a7b5a01e1e7add44b9f5b671c22e9a9173b4bef"
        },
        "date": 1745959232723,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1709637678562266,
            "unit": "iter/sec",
            "range": "stddev: 0.24184444872591127",
            "extra": "mean: 5.849192566000056 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.039844456138748284,
            "unit": "iter/sec",
            "range": "stddev: 1.7334009758680708",
            "extra": "mean: 25.097594418599964 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.11537679738626186,
            "unit": "iter/sec",
            "range": "stddev: 0.07258796325324866",
            "extra": "mean: 8.667253924999931 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.009082597442271714,
            "unit": "iter/sec",
            "range": "stddev: 2.25787340421501",
            "extra": "mean: 110.10066298280007 sec\nrounds: 5"
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
          "id": "ff81c07df95dda17ba1dfb9971412c08c781c751",
          "message": "Add & document some profiling tools (#233)\n\nDocument some helpful profiling tools I've been using and how to call\nthem.",
          "timestamp": "2025-05-14T12:12:01-04:00",
          "tree_id": "fd878e3503e902e8193f3214a27215347d2cf9c3",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/ff81c07df95dda17ba1dfb9971412c08c781c751"
        },
        "date": 1747241807700,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.17267110135061173,
            "unit": "iter/sec",
            "range": "stddev: 0.7502925839736372",
            "extra": "mean: 5.791357049200042 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.04390378179388642,
            "unit": "iter/sec",
            "range": "stddev: 0.721063408262901",
            "extra": "mean: 22.777081133799946 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.12847493478873134,
            "unit": "iter/sec",
            "range": "stddev: 0.10961939719760452",
            "extra": "mean: 7.783619440200027 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.009720141144220717,
            "unit": "iter/sec",
            "range": "stddev: 2.3425248771308143",
            "extra": "mean: 102.87916452679988 sec\nrounds: 5"
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
          "id": "d93abe463079a04cdd70baf24bd56edcda7f9b8f",
          "message": "CUDA Memory profiler (#242)\n\nLets us turn on CUDA memory snapshots via config.\n\n---------\n\nCo-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>",
          "timestamp": "2025-05-19T16:16:56-04:00",
          "tree_id": "976d6aa522f8d1e5f0cdec1d477541c47a89b745",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/d93abe463079a04cdd70baf24bd56edcda7f9b8f"
        },
        "date": 1747688787026,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.18705245521163422,
            "unit": "iter/sec",
            "range": "stddev: 0.2108310595110439",
            "extra": "mean: 5.346093954599974 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.04310672466861702,
            "unit": "iter/sec",
            "range": "stddev: 0.604814642436247",
            "extra": "mean: 23.198236648400005 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.12296778625927976,
            "unit": "iter/sec",
            "range": "stddev: 0.11271335384734628",
            "extra": "mean: 8.13221112960009 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.009651613320130151,
            "unit": "iter/sec",
            "range": "stddev: 1.4437811656932669",
            "extra": "mean: 103.60962119300021 sec\nrounds: 5"
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
          "id": "50579904a32d156dbb0accb473f4cabb23f2738d",
          "message": "Apply same fix to benchmarks, too (#245)\n\nLooks like this wasn't a perf regression, it was a failure. Sorry, I\nshould have checked here after finding it in the other two places!",
          "timestamp": "2025-05-19T19:00:44-04:00",
          "tree_id": "eaf4c42454c0be041b139e98fae1da1f31c209a9",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/50579904a32d156dbb0accb473f4cabb23f2738d"
        },
        "date": 1747698330949,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19280430743899504,
            "unit": "iter/sec",
            "range": "stddev: 0.24502839882247845",
            "extra": "mean: 5.186606115200038 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.046562747005027286,
            "unit": "iter/sec",
            "range": "stddev: 0.4783840935314069",
            "extra": "mean: 21.476396138999963 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.12400506981642946,
            "unit": "iter/sec",
            "range": "stddev: 0.10668534587643125",
            "extra": "mean: 8.06418641979999 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.009590209262365113,
            "unit": "iter/sec",
            "range": "stddev: 0.6877916349379576",
            "extra": "mean: 104.27301142679994 sec\nrounds: 5"
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
          "id": "d6830a1c89aaad5ac0b1488cb19f45960b48f166",
          "message": "Activation Checkpointing (#243)\n\nI am looking into fitting the high res models on our GPUs, so I tried a\ncouple options for activation checkpointing: checkpoint each\nConvNeXtBlock and other top-level layers (\"all\") or checkpointing just\nsimple scalings/nonlinearities (\"simple\").\n\nResults are here:\nhttps://wandb.ai/m2lines/ocean-emulators/reports/Checkpointing-2025-05-16--VmlldzoxMjgwOTIyOA\n\ntl;dr: \"all\" saves about 67% of GPU memory usage (!) but at a\nperformance cost of about 10%. \"simple\" saves about 20% of GPU memory\nbut has practically no performance degradation. There are probably\nstrategies in between those two we could look at, too. (Perhaps also\nsome wins to be had by using the extra space to pre-load future steps\nworth of data or similar.)",
          "timestamp": "2025-05-20T15:10:49-04:00",
          "tree_id": "d0f7ce37b57625846e6198b323e9edcb4529910f",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/d6830a1c89aaad5ac0b1488cb19f45960b48f166"
        },
        "date": 1747771062884,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.18491686286574113,
            "unit": "iter/sec",
            "range": "stddev: 0.2921219332469261",
            "extra": "mean: 5.407835632200022 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.04868497420598307,
            "unit": "iter/sec",
            "range": "stddev: 0.9275715236559205",
            "extra": "mean: 20.54021833859997 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1278092627861711,
            "unit": "iter/sec",
            "range": "stddev: 0.089728646994505",
            "extra": "mean: 7.824159049199989 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.009942562699931963,
            "unit": "iter/sec",
            "range": "stddev: 0.6037746029239257",
            "extra": "mean: 100.57769110239988 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "m@mihasya.com",
            "name": "Mikhail P",
            "username": "mihasya"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f3f47b96bedd2e1fb0bf6a45205e57ebb9ed5afa",
          "message": "Revert \"Switch to using EC2 for CI workers\" (#249)\n\nReverting while we figure out why EC2 runners are not registering\ncorrectly when running on the parent repo.\n\nReverts suryadheeshjith/Ocean_Emulator#246",
          "timestamp": "2025-05-22T08:21:10-07:00",
          "tree_id": "d0f7ce37b57625846e6198b323e9edcb4529910f",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/f3f47b96bedd2e1fb0bf6a45205e57ebb9ed5afa"
        },
        "date": 1747929834442,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19110160272176757,
            "unit": "iter/sec",
            "range": "stddev: 0.3307962426482684",
            "extra": "mean: 5.23281848900001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.04805523137889997,
            "unit": "iter/sec",
            "range": "stddev: 0.8388242066228467",
            "extra": "mean: 20.80938893239995 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.12989231475549146,
            "unit": "iter/sec",
            "range": "stddev: 0.09261669277074258",
            "extra": "mean: 7.69868488279999 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.010155420092186599,
            "unit": "iter/sec",
            "range": "stddev: 0.42053263538538505",
            "extra": "mean: 98.46958480519997 sec\nrounds: 5"
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
          "id": "ea6c3ccf85283472fe7dea10c421310c7309458e",
          "message": "Torch Loader works with compact data. (#215)\n\nThis PR depends on #211. Fixes #204.\n\nHere is the WandB profile results:\nhttps://wandb.ai/m2lines/ocean-emulators/reports/Compaction-and-Vectorization-on-Data-Load-Time--VmlldzoxMjg1MDE4NQ\n\nConclusions: compaction is at least as fast as before, but often\npredictably faster.",
          "timestamp": "2025-05-22T14:42:30-07:00",
          "tree_id": "3ac0be2c892c274e4df61226b84c28f86ca6a928",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/ea6c3ccf85283472fe7dea10c421310c7309458e"
        },
        "date": 1747952712127,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1953602256331596,
            "unit": "iter/sec",
            "range": "stddev: 0.38284148713957716",
            "extra": "mean: 5.118749206799976 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.048917463218172176,
            "unit": "iter/sec",
            "range": "stddev: 1.1110152688933874",
            "extra": "mean: 20.442597269200043 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1281596065898112,
            "unit": "iter/sec",
            "range": "stddev: 0.023254332155223222",
            "extra": "mean: 7.802770518800116 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.010246132388796453,
            "unit": "iter/sec",
            "range": "stddev: 0.6880261137824993",
            "extra": "mean: 97.59780198560011 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "mihasya",
            "username": "mihasya",
            "email": "m@mihasya.com"
          },
          "committer": {
            "name": "mihasya",
            "username": "mihasya",
            "email": "m@mihasya.com"
          },
          "id": "a4e8a8862791a04104ac27a7d98a32155c7f221e",
          "message": "use fixed/vendored version of gha-runner on benchmark workflow",
          "timestamp": "2025-05-22T18:24:50Z",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/a4e8a8862791a04104ac27a7d98a32155c7f221e"
        },
        "date": 1748007640742,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.2535820683975566,
            "unit": "iter/sec",
            "range": "stddev: 0.20270975687549908",
            "extra": "mean: 3.9434965032000493 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06742043402380439,
            "unit": "iter/sec",
            "range": "stddev: 0.5226778257270897",
            "extra": "mean: 14.832298463799953 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1838940306579116,
            "unit": "iter/sec",
            "range": "stddev: 0.0705121926144088",
            "extra": "mean: 5.437914414200032 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.011701732134672037,
            "unit": "iter/sec",
            "range": "stddev: 0.6767896641855309",
            "extra": "mean: 85.45743386459999 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "m@mihasya.com",
            "name": "Mikhail P",
            "username": "mihasya"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d586ae5475dc034925ee58d35e9376a73c6d0981",
          "message": "Switch to using EC2 for CI workers, take 2 (#250)\n\nThis changeset switches the repo to use OpenAthena's EC2 account to\nperform CI checks.\n\nUnlike its predecessor, #246, this PR depends on a branch of\n`start-gha-ec2-runner` which contains a fix for a bug that prevented the\nrunner machines from being detected by GHA. We will upstream this fix\nand send a follow-up PR when the fix has been accepted.\n\nWe observe a modest improvement in benchmarks when running on AWS\n`g6.xlarge` VMs vs `g2-standard-8` Google Cloud VMs.",
          "timestamp": "2025-05-27T06:38:11-07:00",
          "tree_id": "0a2f733705754f56bc04f44be0e10fd48ad40fa2",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/d586ae5475dc034925ee58d35e9376a73c6d0981"
        },
        "date": 1748355006053,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.2879019876220538,
            "unit": "iter/sec",
            "range": "stddev: 0.2123753961156003",
            "extra": "mean: 3.473404293800013 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07192143839846069,
            "unit": "iter/sec",
            "range": "stddev: 0.5441461141256482",
            "extra": "mean: 13.904060072600032 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19395529780361837,
            "unit": "iter/sec",
            "range": "stddev: 0.09336021008967484",
            "extra": "mean: 5.155827200000021 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.012699829861767181,
            "unit": "iter/sec",
            "range": "stddev: 0.4117022051818577",
            "extra": "mean: 78.74121235359999 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "mihasya",
            "username": "mihasya",
            "email": "m@mihasya.com"
          },
          "committer": {
            "name": "mihasya",
            "username": "mihasya",
            "email": "m@mihasya.com"
          },
          "id": "c5a6f952a2b2c43315c3cd50226ce601fbda7f36",
          "message": "try out 6g.2xlarge to see what that does to benchmark results",
          "timestamp": "2025-05-28T14:19:14Z",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/c5a6f952a2b2c43315c3cd50226ce601fbda7f36"
        },
        "date": 1748443890660,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.27333604961470004,
            "unit": "iter/sec",
            "range": "stddev: 0.23796517919192492",
            "extra": "mean: 3.6585002285999964 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07089347473770445,
            "unit": "iter/sec",
            "range": "stddev: 0.3810742335418997",
            "extra": "mean: 14.105670567000061 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.193307963920903,
            "unit": "iter/sec",
            "range": "stddev: 0.10101394344093963",
            "extra": "mean: 5.173092612000073 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.01362725137228367,
            "unit": "iter/sec",
            "range": "stddev: 0.3575905419060481",
            "extra": "mean: 73.38236983239995 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "m@mihasya.com",
            "name": "Mikhail P",
            "username": "mihasya"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "cd9c83eef13b76242972bb05ca858bfc872618bb",
          "message": "Remove pointer to patched version of GH Action (#254)\n\nMinor cleanup from #250 removing a reference to a patched CI dependency.",
          "timestamp": "2025-05-29T08:39:15-07:00",
          "tree_id": "c8ac84cac2c120a97b3b75ff9d967ab5fce952aa",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/cd9c83eef13b76242972bb05ca858bfc872618bb"
        },
        "date": 1748535148310,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.25493553941094166,
            "unit": "iter/sec",
            "range": "stddev: 0.12352234543405288",
            "extra": "mean: 3.922560198199972 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06686470306366986,
            "unit": "iter/sec",
            "range": "stddev: 0.40299779826144594",
            "extra": "mean: 14.955573780800023 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.18774462971066186,
            "unit": "iter/sec",
            "range": "stddev: 0.08668800917285621",
            "extra": "mean: 5.326384043799953 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.011968950384872234,
            "unit": "iter/sec",
            "range": "stddev: 0.4958067812813021",
            "extra": "mean: 83.54951502380004 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Jesse Rusak",
            "username": "jder",
            "email": "jesse@openathena.ai"
          },
          "committer": {
            "name": "Jesse Rusak",
            "username": "jder",
            "email": "jesse@openathena.ai"
          },
          "id": "57579549ffb2d962977b629f3fc7801397ebca6e",
          "message": "fix",
          "timestamp": "2025-06-06T13:42:54Z",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/57579549ffb2d962977b629f3fc7801397ebca6e"
        },
        "date": 1749220407715,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.2616260095265049,
            "unit": "iter/sec",
            "range": "stddev: 0.2886881857473763",
            "extra": "mean: 3.822249942999997 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06967641493434427,
            "unit": "iter/sec",
            "range": "stddev: 0.33228524114751823",
            "extra": "mean: 14.352058741000018 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19077413513951155,
            "unit": "iter/sec",
            "range": "stddev: 0.07726722492389228",
            "extra": "mean: 5.241800725599978 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.013012495049751395,
            "unit": "iter/sec",
            "range": "stddev: 0.4083603376072332",
            "extra": "mean: 76.84921271260005 sec\nrounds: 5"
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
          "id": "cd2e5d64d8d9b6ceb2a1ed6220c622dc00763ee9",
          "message": "Switch to torch loader and reduce default number of workers per GPU (#255)\n\nI think the torch loader is a good default now, and I found 4 workers by\ndefault overloads the 8-GPU systems I've been using, resulting in very\nlong waits at the start of an epoch. 1-worker-per-GPU seems to be too\nfew on my other host, so I was thinking this might be a better default.\n(Though of course might need tuning.)",
          "timestamp": "2025-06-10T13:00:58-04:00",
          "tree_id": "6cfee0c0e32770bde2438ef83e8574ba905f607f",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/cd2e5d64d8d9b6ceb2a1ed6220c622dc00763ee9"
        },
        "date": 1749576184034,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2632050507398964,
            "unit": "iter/sec",
            "range": "stddev: 0.029235975444639076",
            "extra": "mean: 791.6371133999746 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.2556804163078871,
            "unit": "iter/sec",
            "range": "stddev: 0.006876906669233305",
            "extra": "mean: 3.9111325553999903 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1970428845844339,
            "unit": "iter/sec",
            "range": "stddev: 0.05609207267115057",
            "extra": "mean: 5.075037356000007 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.016879869303636365,
            "unit": "iter/sec",
            "range": "stddev: 0.17302647685957595",
            "extra": "mean: 59.24216485400002 sec\nrounds: 5"
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
          "id": "07aa2191c4c8c85a267c5fdcd54c201fb0cec1f6",
          "message": "Complain if the validation or inference time ranges overlap with train (#258)\n\nI did this by accident last week so let's prevent people from doing it.\n\n---------\n\nCo-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>",
          "timestamp": "2025-06-10T14:20:15-04:00",
          "tree_id": "39ff4517aeaf326cbff211d3b6f8035a28a5e4ce",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/07aa2191c4c8c85a267c5fdcd54c201fb0cec1f6"
        },
        "date": 1749580909686,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.30339494781557,
            "unit": "iter/sec",
            "range": "stddev: 0.03299218900497334",
            "extra": "mean: 767.2271567999815 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.2483562697126072,
            "unit": "iter/sec",
            "range": "stddev: 0.04210633238543347",
            "extra": "mean: 4.026473747399973 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19775311430748366,
            "unit": "iter/sec",
            "range": "stddev: 0.07384074187340443",
            "extra": "mean: 5.0568103744000386 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.016584643033715363,
            "unit": "iter/sec",
            "range": "stddev: 0.25224580658194756",
            "extra": "mean: 60.29674548720002 sec\nrounds: 5"
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
          "id": "87e1ec3a77ea6a1fde81946786cc3d0873cf268a",
          "message": "Deduplicate start/stop jobs, remove patched version of action (#256)\n\nI resisted doing this because this is more moving pieces and also more\ncode, but this is also at least the third time we've had them drift out\nof sync (in this case, one was using the mihasya patched version and the\nother not), so I think it's probably worth it.",
          "timestamp": "2025-06-10T14:19:58-04:00",
          "tree_id": "0fb347ff79be7fba168adb4c0760c6a343a62f12",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/87e1ec3a77ea6a1fde81946786cc3d0873cf268a"
        },
        "date": 1749580961744,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2759609418413964,
            "unit": "iter/sec",
            "range": "stddev: 0.011016778692026821",
            "extra": "mean: 783.7230491999662 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.25331023152060533,
            "unit": "iter/sec",
            "range": "stddev: 0.013908027640403917",
            "extra": "mean: 3.9477284197999554 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1948612868008294,
            "unit": "iter/sec",
            "range": "stddev: 0.07908910852860435",
            "extra": "mean: 5.131855672400002 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.015623856096446427,
            "unit": "iter/sec",
            "range": "stddev: 0.9326206153696651",
            "extra": "mean: 64.00468577199999 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Jesse Rusak",
            "username": "jder",
            "email": "jesse@openathena.ai"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "07aa2191c4c8c85a267c5fdcd54c201fb0cec1f6",
          "message": "Complain if the validation or inference time ranges overlap with train (#258)\n\nI did this by accident last week so let's prevent people from doing it.\n\n---------\n\nCo-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>",
          "timestamp": "2025-06-10T18:20:15Z",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/07aa2191c4c8c85a267c5fdcd54c201fb0cec1f6"
        },
        "date": 1749759004654,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.3284278672315972,
            "unit": "iter/sec",
            "range": "stddev: 0.019957785174541023",
            "extra": "mean: 752.7695140000105 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.24926218857056953,
            "unit": "iter/sec",
            "range": "stddev: 0.07501504430662692",
            "extra": "mean: 4.0118399254000225 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.198444088479734,
            "unit": "iter/sec",
            "range": "stddev: 0.06401666900045658",
            "extra": "mean: 5.039202768199994 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.016488823627640183,
            "unit": "iter/sec",
            "range": "stddev: 0.13873674831466504",
            "extra": "mean: 60.6471403044 sec\nrounds: 5"
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
          "id": "a644488d63d087f52514e991d2ca60937f56e212",
          "message": "Use passed loader version to decide whether to use dask (#260)\n\n@alxmrs pretty sure this is why our benchmarks did something surprising\nwhen we changed the default loader version. Previously the passed\n`version` was ignored for the purpose of deciding whether to use dask or\nnot. Seems like good additional evidence that we should try and tackle\nthis duplication sometime soon.\n\nThis also makes me want to move to a world where we're not using config\nobjects in most tests. It's hard to tell whether this change is\nsufficient -- if any other code we're calling here transitively uses\n`cfg.data.loader_version`, then it's probably broken. We could also make\na new train config but then if the *caller* is using\n`cfg.data.loader_version` we have the same problem. We could update all\ncallers to mutate train config rather than passing in verison and\ntime_config separately but I would rather move away from manipulating\nconfig rather than towards. (But could be convinced this is still worth\ndoing in the short term.)",
          "timestamp": "2025-06-16T11:57:54-04:00",
          "tree_id": "b1f88c1e9950ab1dba0d7d5cc696362e6df605d8",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/a644488d63d087f52514e991d2ca60937f56e212"
        },
        "date": 1750090975963,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.286008379957789,
            "unit": "iter/sec",
            "range": "stddev: 0.01937676619228541",
            "extra": "mean: 777.599909599985 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07044165578120938,
            "unit": "iter/sec",
            "range": "stddev: 0.33365539503642194",
            "extra": "mean: 14.196145574799994 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1971834154747392,
            "unit": "iter/sec",
            "range": "stddev: 0.0795033265624155",
            "extra": "mean: 5.071420421400035 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.01589746663838833,
            "unit": "iter/sec",
            "range": "stddev: 1.2269667394225694",
            "extra": "mean: 62.9031041704 sec\nrounds: 5"
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
          "id": "9f3b92249932b442b16a93744398acd50aabcc9c",
          "message": "Notes on how to reproduce OM4 half-deg preprocessing. (#261)\n\nI've refreshed the contributing guide, adding a data engineering\nsection. Here, I explain how to clone the data and how to start a coiled\nnotebook to create the half-deg data. The docs indicate that this all\nmay change soon -- and that is my plan. I am happy to refresh these docs\noverall.\n\n---------\n\nCo-authored-by: Surya <surya.dheeshjith@gmail.com>",
          "timestamp": "2025-06-18T11:52:16-07:00",
          "tree_id": "96598efc2baf77a86569cde8e32850a308b0cfed",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/9f3b92249932b442b16a93744398acd50aabcc9c"
        },
        "date": 1750274289472,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2600873933610808,
            "unit": "iter/sec",
            "range": "stddev: 0.035119516628354504",
            "extra": "mean: 793.5957500000541 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07051789012702615,
            "unit": "iter/sec",
            "range": "stddev: 0.7963645988480913",
            "extra": "mean: 14.180798634200027 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19150619920016068,
            "unit": "iter/sec",
            "range": "stddev: 0.09677539280957344",
            "extra": "mean: 5.221763076999968 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.01544718719494644,
            "unit": "iter/sec",
            "range": "stddev: 0.1156607438926149",
            "extra": "mean: 64.73670496640003 sec\nrounds: 5"
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
          "id": "5f662af829c4088ae43c55060a116d7bb09aaffa",
          "message": "Avoid failing benchmarks when they regress (#264)\n\nThis is a bit weird since I always think something went wrong, but I\nthink regressions are \"normal\" and not failures. In addition, failing\nthe first \"store\" task causes the second one to never run, which seems\nbad.",
          "timestamp": "2025-06-18T15:01:41-04:00",
          "tree_id": "7deb607a3746aef4172aedcb6a2c5e4d5c0459d6",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/5f662af829c4088ae43c55060a116d7bb09aaffa"
        },
        "date": 1750274846819,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2638761950114328,
            "unit": "iter/sec",
            "range": "stddev: 0.028107516980727748",
            "extra": "mean: 791.2167377999822 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07192833580357433,
            "unit": "iter/sec",
            "range": "stddev: 0.34220090226190236",
            "extra": "mean: 13.902726774199982 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.197177214716821,
            "unit": "iter/sec",
            "range": "stddev: 0.07115261501091602",
            "extra": "mean: 5.07157990559997 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.015563976504020166,
            "unit": "iter/sec",
            "range": "stddev: 0.1507209267650743",
            "extra": "mean: 64.25093225639993 sec\nrounds: 5"
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
          "id": "72cb9be4c09bcad079e6648a880b3cbc6e24b361",
          "message": "Support absolute URLs for file paths, remove '*' support (#262)\n\nAt the moment, we don't support `root` being a URL but this seems like a\nstep in the right direction.\n\nI also don't think we need the \"*\" support, and defaulting to `debug:\nfalse` in eval is consistent with the other configurations.\n\n---------\n\nCo-authored-by: Alex Merose <alex@openathena.ai>",
          "timestamp": "2025-06-23T13:08:51-04:00",
          "tree_id": "5a023c1cd0f4ac43133276ef8f72ac1d23c53e52",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/72cb9be4c09bcad079e6648a880b3cbc6e24b361"
        },
        "date": 1750700098985,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2814319663104945,
            "unit": "iter/sec",
            "range": "stddev: 0.030038240744911245",
            "extra": "mean: 780.3769738000256 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07015861529216302,
            "unit": "iter/sec",
            "range": "stddev: 0.7694681617172575",
            "extra": "mean: 14.25341700140002 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19477896240962894,
            "unit": "iter/sec",
            "range": "stddev: 0.10199154872348914",
            "extra": "mean: 5.134024679199979 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.015277103530821177,
            "unit": "iter/sec",
            "range": "stddev: 0.22814152076892982",
            "extra": "mean: 65.45743425660007 sec\nrounds: 5"
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
          "id": "32f3043d20bb43b24a1a097004f5c0f34f013075",
          "message": "Tweaks while running half-degree experiments (#283)\n\nThe 3 commits are independent so probably easiest to review that way.\n(Happy to break into separate PRs but I think they're all pretty\nuncontroversial.)\n\n* Updates config for the half-degree experiments to load remotely when\ndesired and to avoid doing inference during training.\n* Adds a flag to load xarray variables concurrently based on [this\nsuggestion](https://github.com/pydata/xarray/issues/8965#issuecomment-2083787484).\nThis helps a lot locally with small datasizes (3x improvement for me)\nbut with larger batch sizes it seems to not help (maybe GIL\ncontention?).\n* Swaps to the the \"spawn\" multiprocessing type since the s3fs requires\nthis when passing those stores around.\n* Adds more logging statements so you can tell what's going on more\neasily\n* Removes some logic we had about tweaking the number of workers based\non the number of GPUs in use -- the worker counts are are *already* per\nGPU so this was using `num_gpus * num_gpus * num_workers` workers in\ntotal. Also allows us to use multiple workers for CPU jobs, for testing\npurposes.",
          "timestamp": "2025-06-25T13:20:35-04:00",
          "tree_id": "83e2145e1efb9308a0172521dcd9a66d10070aca",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/32f3043d20bb43b24a1a097004f5c0f34f013075"
        },
        "date": 1750874165617,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.242908630794658,
            "unit": "iter/sec",
            "range": "stddev: 0.028888238209976814",
            "extra": "mean: 804.5643703999758 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06919374438085289,
            "unit": "iter/sec",
            "range": "stddev: 0.41912162127384534",
            "extra": "mean: 14.452173515800041 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.08084525855235832,
            "unit": "iter/sec",
            "range": "stddev: 0.06083002859071351",
            "extra": "mean: 12.369309195199913 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.010170908708447808,
            "unit": "iter/sec",
            "range": "stddev: 0.19842781812418364",
            "extra": "mean: 98.31963187020001 sec\nrounds: 5"
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
          "id": "8b0601504cae8d4940cd645928e53660f6a886a4",
          "message": "Put OSN Bucket keys in environment variables. (#285)\n\nI plan to scrub the access keys that I checked in. For good measure, I\nalso plan to invalidate these keys and get new ones for myself.\n\nThis PR also adds a script that converts jupyter notebooks into Python\nscripts. It fixes #287.\n\n---------\n\nCo-authored-by: Ryan Williams <nobigdealstyle@gmail.com>",
          "timestamp": "2025-06-30T12:52:21-07:00",
          "tree_id": "c5c5b0946a4178d50c6a0b380166935812bfbf44",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/8b0601504cae8d4940cd645928e53660f6a886a4"
        },
        "date": 1751315397116,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2387626952096484,
            "unit": "iter/sec",
            "range": "stddev: 0.06146599312902097",
            "extra": "mean: 807.2571153999434 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.0671345842278371,
            "unit": "iter/sec",
            "range": "stddev: 0.618875101264874",
            "extra": "mean: 14.895452343999978 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07602654874238748,
            "unit": "iter/sec",
            "range": "stddev: 0.12419660550675399",
            "extra": "mean: 13.153299953000033 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.009294919919765962,
            "unit": "iter/sec",
            "range": "stddev: 0.21585204783912043",
            "extra": "mean: 107.58564986379993 sec\nrounds: 5"
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
          "id": "541044de5af013efb189b91ebd7807d39752370d",
          "message": "Put OSN Bucket keys in environment variables. (#285)\n\nI plan to scrub the access keys that I checked in. For good measure, I\nalso plan to invalidate these keys and get new ones for myself.\n\nThis PR also adds a script that converts jupyter notebooks into Python\nscripts. It fixes #287.\n\n---------\n\nCo-authored-by: Ryan Williams <nobigdealstyle@gmail.com>",
          "timestamp": "2025-06-30T12:52:21-07:00",
          "tree_id": "c5c5b0946a4178d50c6a0b380166935812bfbf44",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/541044de5af013efb189b91ebd7807d39752370d"
        },
        "date": 1751914008429,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2433555335177084,
            "unit": "iter/sec",
            "range": "stddev: 0.025364741160628625",
            "extra": "mean: 804.2751835999752 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.0683661899258133,
            "unit": "iter/sec",
            "range": "stddev: 0.19413338567410024",
            "extra": "mean: 14.627113213199937 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.0783206116113354,
            "unit": "iter/sec",
            "range": "stddev: 0.11203531932667467",
            "extra": "mean: 12.768031038399977 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.009625755717765196,
            "unit": "iter/sec",
            "range": "stddev: 0.3280760474311185",
            "extra": "mean: 103.88794701639999 sec\nrounds: 5"
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
          "id": "52ae948638fd948154d8aa445a146cdb411b3346",
          "message": "Updated `concurrent_compute` logic to create a thread pool only once. (#288)\n\nI think that it's a good idea to try to load all the data variables\nconcurrently. However, before this PR, we create and then tear down the\nexecutor way too many times (per loop per getitem). This change will\nallow to to see how effective we can concurrently load Xarray data.",
          "timestamp": "2025-07-07T14:10:50-07:00",
          "tree_id": "528b34aa5b5b696c4bbbe921eb5716497b18ec9a",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/52ae948638fd948154d8aa445a146cdb411b3346"
        },
        "date": 1751924795777,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2917348284995018,
            "unit": "iter/sec",
            "range": "stddev: 0.03979475569281503",
            "extra": "mean: 774.1526960000101 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06830809700412027,
            "unit": "iter/sec",
            "range": "stddev: 0.7132817860707142",
            "extra": "mean: 14.639552906000016 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.08022508119885201,
            "unit": "iter/sec",
            "range": "stddev: 0.10546882333147832",
            "extra": "mean: 12.464929733400004 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.009958370667425796,
            "unit": "iter/sec",
            "range": "stddev: 0.28910666771540494",
            "extra": "mean: 100.41803357159998 sec\nrounds: 5"
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
          "id": "7555580c3f9be1f1fb7658c24d798eaaa847c5ed",
          "message": "Ignore some more things (#294)",
          "timestamp": "2025-07-08T15:14:27-04:00",
          "tree_id": "795235b648ce3bb3f130385fdc6db4fb21012be1",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/7555580c3f9be1f1fb7658c24d798eaaa847c5ed"
        },
        "date": 1752004168300,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2902770633330904,
            "unit": "iter/sec",
            "range": "stddev: 0.015844990971596713",
            "extra": "mean: 775.027339799999 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07056999660336255,
            "unit": "iter/sec",
            "range": "stddev: 0.5656704698286922",
            "extra": "mean: 14.17032801660007 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.08210656504861366,
            "unit": "iter/sec",
            "range": "stddev: 0.07794886833656797",
            "extra": "mean: 12.179294060199936 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.010274592613629044,
            "unit": "iter/sec",
            "range": "stddev: 0.3270724649183269",
            "extra": "mean: 97.32745984239996 sec\nrounds: 5"
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
          "id": "cc0d2b287407dd91f914074ec94f4e75da12779c",
          "message": "Add a \"Location\" to support S3 and local paths (#284)\n\nThis lets us:\n* Add structured info for locations (eg s3 endpoint URLs)\n* Still use raw strings most of the time\n* Get some type safety to know if we've correctly resolved these against\nthe root before opening.\n* (In a future PR) detect the need for \"spawn\" multiprocessing (required\nfor S3 but not for local files) which has a performance cost.\n\nAlso removes the \"gantry\" boolean because I think we can use\n`data_root:/` if we want that",
          "timestamp": "2025-07-09T09:55:11-04:00",
          "tree_id": "50a1e982f6219a0b40fd1a418f47ee6ecafb894e",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/cc0d2b287407dd91f914074ec94f4e75da12779c"
        },
        "date": 1752071463927,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2851737642848073,
            "unit": "iter/sec",
            "range": "stddev: 0.01901377494977276",
            "extra": "mean: 778.1048973999987 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07022847778350386,
            "unit": "iter/sec",
            "range": "stddev: 0.456007958909278",
            "extra": "mean: 14.23923786420005 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.08174504605871333,
            "unit": "iter/sec",
            "range": "stddev: 0.07874828309224062",
            "extra": "mean: 12.233157215199935 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.009889412569393985,
            "unit": "iter/sec",
            "range": "stddev: 3.867143679983108",
            "extra": "mean: 101.11824064199996 sec\nrounds: 5"
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
          "id": "628310a0e41f387ea808db2027112753ff6c7820",
          "message": "Fix performance regression by moving towards a more buildy world with DataContainer (#295)\n\nFixes the performance regression from\n[here](https://github.com/LaureZanna/Ocean_Emulator/commit/32f3043d20bb43b24a1a097004f5c0f34f013075#commitcomment-160869800)\nby detecting if the data source supports forking or not.\n\nDoing this required a bit of code movement so I took the opportunity to\nmove us closer towards a world where creating the trainer is `trainer =\ncfg.build()` by moving a bunch of the stuff in `Trainer.__init__` into\n`DataConfig.build` and creating a new class which is the result of that\nmethod, `DataContainer`. This also reduces some of the duplication\nbetween train/eval/tests.\n\nLast commit is some drive-by fixes for warnings that were being printed\nin tests.",
          "timestamp": "2025-07-09T19:10:04-04:00",
          "tree_id": "9ef480f8a8af2a3a899f1af138ef0f5623910c5b",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/628310a0e41f387ea808db2027112753ff6c7820"
        },
        "date": 1752103961778,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2552960712153403,
            "unit": "iter/sec",
            "range": "stddev: 0.02708861350047886",
            "extra": "mean: 796.6248145999771 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2507511946010124,
            "unit": "iter/sec",
            "range": "stddev: 0.02424740351059275",
            "extra": "mean: 799.5195242000136 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19360106348304268,
            "unit": "iter/sec",
            "range": "stddev: 0.0657015089858014",
            "extra": "mean: 5.165260882400003 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.015305987302328244,
            "unit": "iter/sec",
            "range": "stddev: 0.4193031114150969",
            "extra": "mean: 65.33391020439998 sec\nrounds: 5"
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
          "id": "1946c397256e79c51032d671dae1c58466a27af3",
          "message": "Fix bug which ignored loader version in tests (#297)\n\nOur benchmarks showed a surprising jump in one of them after #295:\n<img width=\"1028\" height=\"514\" alt=\"Screenshot 2025-07-10 at 10 26\n37 AM\"\nsrc=\"https://github.com/user-attachments/assets/3322d95a-817e-4113-be9a-704ed2b3638c\"\n/>\n\nTurns out we were always building a torch loader, not an eager loader.\nRunning the benchmarks locally after this change again shows a big\nperformance gap between eager and torch loaders.",
          "timestamp": "2025-07-11T12:44:20-04:00",
          "tree_id": "bfd6a865745f3919d0465de8fe384d3b749a6cbb",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/1946c397256e79c51032d671dae1c58466a27af3"
        },
        "date": 1752253801802,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2449692206832164,
            "unit": "iter/sec",
            "range": "stddev: 0.022060454312354327",
            "extra": "mean: 803.232709200006 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07213688588125138,
            "unit": "iter/sec",
            "range": "stddev: 0.3628930060857919",
            "extra": "mean: 13.862533540000005 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19771669042294865,
            "unit": "iter/sec",
            "range": "stddev: 0.10332725531166359",
            "extra": "mean: 5.057741953199979 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.015596531062254447,
            "unit": "iter/sec",
            "range": "stddev: 0.13693051921621888",
            "extra": "mean: 64.11682161940003 sec\nrounds: 5"
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
          "id": "3683713b49d46850d0f38310c278c7e1b78de68a",
          "message": "Improvements from trying to understand the model (#296)\n\n* Makes the model summary include grid sizes by delaying calling it\nuntil we have data to run a forward pass (this will probably slow down\nbenchmarks but we can ignore)\n* This required making TrainData a typing.Mapping to satisfy the summary\nfunction (though this still isn't perfect, it somehow can't figure out\nhow much memory it uses)\n* When debug = true, include all the layers not just top ones\n* Rename and add type annotations to the core samudra model.",
          "timestamp": "2025-07-14T10:15:12-04:00",
          "tree_id": "aa79754e00d8aefa067e3de159cde356a2d9ddb0",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/3683713b49d46850d0f38310c278c7e1b78de68a"
        },
        "date": 1752504046110,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2925501522377736,
            "unit": "iter/sec",
            "range": "stddev: 0.038409179874388646",
            "extra": "mean: 773.6643706000223 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07198145383388076,
            "unit": "iter/sec",
            "range": "stddev: 0.38653558868383014",
            "extra": "mean: 13.892467389000036 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1964246867691096,
            "unit": "iter/sec",
            "range": "stddev: 0.07375063724142698",
            "extra": "mean: 5.091009772999996 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.015678897487895763,
            "unit": "iter/sec",
            "range": "stddev: 0.12133223801546983",
            "extra": "mean: 63.779994784200106 sec\nrounds: 5"
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
          "id": "2c6ea98cc5807206b0b2ee84249ab3eb075197b3",
          "message": "The default chunks of `clone_data` should be time=1. (#306)\n\nI just verified this and it looks like in the OSN pod, chunks are\n`time=1`, but by default, we write `time=10` chunks. I want to remedy\nthis so we don't make any accidents when automating are testing setup.",
          "timestamp": "2025-07-14T15:16:21-07:00",
          "tree_id": "e7090b1bcca421369346568cca8089ad091ff662",
          "url": "https://github.com/LaureZanna/Ocean_Emulator/commit/2c6ea98cc5807206b0b2ee84249ab3eb075197b3"
        },
        "date": 1752532901888,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.305629682296309,
            "unit": "iter/sec",
            "range": "stddev: 0.020805392852399004",
            "extra": "mean: 765.9139598000138 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07231820174649041,
            "unit": "iter/sec",
            "range": "stddev: 0.2941018876137041",
            "extra": "mean: 13.827777459200025 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1944319741490366,
            "unit": "iter/sec",
            "range": "stddev: 0.07283936466991443",
            "extra": "mean: 5.143186990599998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.015654257624394097,
            "unit": "iter/sec",
            "range": "stddev: 0.11208174875792931",
            "extra": "mean: 63.8803847486 sec\nrounds: 5"
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
          "id": "227c5a7be10702ef2dca2b285166f53f889d4c94",
          "message": "The default chunks of `clone_data` should be time=1. (#306)\n\nI just verified this and it looks like in the OSN pod, chunks are\n`time=1`, but by default, we write `time=10` chunks. I want to remedy\nthis so we don't make any accidents when automating are testing setup.",
          "timestamp": "2025-07-14T15:16:21-07:00",
          "tree_id": "e7090b1bcca421369346568cca8089ad091ff662",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/227c5a7be10702ef2dca2b285166f53f889d4c94"
        },
        "date": 1752812757201,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.214978163318985,
            "unit": "iter/sec",
            "range": "stddev: 0.0363692470871584",
            "extra": "mean: 823.0600599999889 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07150401334669337,
            "unit": "iter/sec",
            "range": "stddev: 0.5802694464383088",
            "extra": "mean: 13.985228984999958 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19308048713261342,
            "unit": "iter/sec",
            "range": "stddev: 0.09915794453518381",
            "extra": "mean: 5.1791872646000225 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.01495642547188826,
            "unit": "iter/sec",
            "range": "stddev: 0.3182934174701547",
            "extra": "mean: 66.86089546459989 sec\nrounds: 5"
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
          "id": "a8eca0d2e569a74291d1df4d6853506871c7904f",
          "message": "Avoid LocalLocation relative footgun (#309)\n\nPreviously, we permitted you to make *relative* local locations as in:\n\n```yaml\ndata_location:\n  type: local\n  path: foo/bar\n```\n\nBut the rule is that if you give a ResolvedLocation (type: local is a\nResolvedLocation) then it's *not* resolved relative to the data_root. So\nthis is actually relative to the cwd rather than data_root. This is just\nconfusing so we are now forbidding relative local locations; instead you\nwant:\n\n```yaml\ndata_location: \"foo/bar\"\n```\n\nThis means we can no longer express \"resolved paths which are relative\nto cwd\" but this seems preferable to having this footgun.\n\n---------\n\nCo-authored-by: Alex Merose <alex@openathena.ai>",
          "timestamp": "2025-07-22T17:39:29Z",
          "tree_id": "fa6e4a01bd0246e61bd7566ccbbf968152db6951",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/a8eca0d2e569a74291d1df4d6853506871c7904f"
        },
        "date": 1753207534652,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.3101628974084267,
            "unit": "iter/sec",
            "range": "stddev: 0.0341758149943635",
            "extra": "mean: 763.2638674000418 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06880527276576291,
            "unit": "iter/sec",
            "range": "stddev: 0.36155197200921063",
            "extra": "mean: 14.533769866800004 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1925986317637977,
            "unit": "iter/sec",
            "range": "stddev: 0.10159475431924021",
            "extra": "mean: 5.192144880999967 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.01517253637841196,
            "unit": "iter/sec",
            "range": "stddev: 0.13791328735783917",
            "extra": "mean: 65.90855840180002 sec\nrounds: 5"
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
          "id": "8d93b1e51592819c9c85a1b0d5342545bdb0006f",
          "message": "First pass at setting up one-command training with skypilot. (#298)\n\nWrote entrypoint for performing training runs with skypilot. The `sky`\ncommand is documented within the `train.sky.yaml` file and contributing\nguide.\n\nI did not successfully do a full training run, but that is because I hit\ndata engineering errors (no static_data!).\n\n<details>\n<summary>(Latest error trace related to data engineering)</summary\n\nUltimately, hitting this error seems like a good sign because the error\nis in our code, not in the provisioning system.\n```\nAssertionError: : Static data variable sea_surface_fraction not found in dataStatic data variable sea_surface_fraction not found in data\n```\n\n</details<",
          "timestamp": "2025-07-22T20:53:08Z",
          "tree_id": "6327f17fe130707f862aaa437aea12b22d259e56",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/8d93b1e51592819c9c85a1b0d5342545bdb0006f"
        },
        "date": 1753219200497,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2347392403587827,
            "unit": "iter/sec",
            "range": "stddev: 0.039897025404308804",
            "extra": "mean: 809.8876000000018 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07132509578472099,
            "unit": "iter/sec",
            "range": "stddev: 0.3691452300507837",
            "extra": "mean: 14.020310649400017 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19390613977632035,
            "unit": "iter/sec",
            "range": "stddev: 0.09558615830788372",
            "extra": "mean: 5.15713427720002 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.015125552815623498,
            "unit": "iter/sec",
            "range": "stddev: 0.7962984456221619",
            "extra": "mean: 66.11328605239996 sec\nrounds: 5"
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
          "id": "19c72a4b92ea5a55405704f61a9ec41edc804339",
          "message": "Skypilot 0.10.0 just came out!  (#314)\n\nPinning to that stable version instead of depending on nightly.",
          "timestamp": "2025-07-23T14:41:46Z",
          "tree_id": "d5efb931789a4189eac83520a16db5ad34576e0e",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/19c72a4b92ea5a55405704f61a9ec41edc804339"
        },
        "date": 1753283398263,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2376706342498445,
            "unit": "iter/sec",
            "range": "stddev: 0.02873660410470452",
            "extra": "mean: 807.9694001999997 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06851940047570126,
            "unit": "iter/sec",
            "range": "stddev: 0.358883019576516",
            "extra": "mean: 14.594406738200018 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.18482250974040018,
            "unit": "iter/sec",
            "range": "stddev: 0.09529631341313574",
            "extra": "mean: 5.410596368400093 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.01371473105351525,
            "unit": "iter/sec",
            "range": "stddev: 0.184090545079249",
            "extra": "mean: 72.91429894599996 sec\nrounds: 5"
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
          "id": "5a290621e470ab5f4abdecd9f2d1bbf0c8b19e79",
          "message": "AGENTS.md (#316)\n\nI keep wanting to put things in here so made a first cut at it so we can\ngrow it over time.\n\n---------\n\nCo-authored-by: Alex Merose <alex@openathena.ai>",
          "timestamp": "2025-07-26T13:03:11Z",
          "tree_id": "93de5b5c63179f0e0e9007484dbe1043bbe90816",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/5a290621e470ab5f4abdecd9f2d1bbf0c8b19e79"
        },
        "date": 1753536515488,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.308723016181537,
            "unit": "iter/sec",
            "range": "stddev: 0.010323689391921806",
            "extra": "mean: 764.1036243999906 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.0692609983376631,
            "unit": "iter/sec",
            "range": "stddev: 0.17046953433775328",
            "extra": "mean: 14.438140136600008 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1952525062191809,
            "unit": "iter/sec",
            "range": "stddev: 0.08334556129824325",
            "extra": "mean: 5.121573184199997 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.015633314407082616,
            "unit": "iter/sec",
            "range": "stddev: 0.1829330532247559",
            "extra": "mean: 63.965962300799994 sec\nrounds: 5"
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
          "id": "22812c426eebfcc158fcd32d4529ce25a3092742",
          "message": "Hopefully fix data race in cache reading/writing (#318)\n\nBreaks out the test change from #313 and fixes the bug (see eg [this\nactions\nrun](https://github.com/Open-Athena/Ocean_Emulator/actions/runs/16573115391/job/46880287790))\nthat exposed.\n\nWe now ensure that only one process (of possibly-many pytest processes)\nwrite to a given data source cache at once. We also ensure that\nprocesses that would like to `open` a given cache wait for any writers\nof that cache to finish. (They also wait for other readers, though this\nis unnecessary.) This locking is also hoisted upwards such that only one\nprocess will try to create the remote/compact/mock data in the case when\nit is not cached, which should hopefully reduce the chance of that\nfailing.\n\nFixes #257",
          "timestamp": "2025-07-29T14:41:42Z",
          "tree_id": "7c74980a4e88603a4e838fe3bdd4b64fe75ea52b",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/22812c426eebfcc158fcd32d4529ce25a3092742"
        },
        "date": 1753801742926,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.257248751914828,
            "unit": "iter/sec",
            "range": "stddev: 0.06104922276895987",
            "extra": "mean: 795.3875464000021 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06906807598442025,
            "unit": "iter/sec",
            "range": "stddev: 0.4744532867331867",
            "extra": "mean: 14.478469042999995 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1906041315716836,
            "unit": "iter/sec",
            "range": "stddev: 0.09790941054696249",
            "extra": "mean: 5.2464759905999925 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.014172591973173401,
            "unit": "iter/sec",
            "range": "stddev: 0.19023555202479867",
            "extra": "mean: 70.558723619 sec\nrounds: 5"
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
          "id": "fca69bcf55ee435edceb52595536b733242cfebc",
          "message": "Fix EMA state restore when resuming training from a checkpoint (#313)\n\nThis fixes a silly bug which overwrote all EMA state after resuming from\na checkpoint. It also now declares the two kinds of checkpoints to be\nfor different uses:\n\n* EMA checkpoints: these bake in the averaged weights into the model\nitself, and so are useful for inference that wants to start from that\naveraged model.\n* Non-EMA checkpoints: these now save the averaged model weights\nseparately from the main model weights so we can resume training without\nstarting that averaging over again.\n\nThese maybe could be renamed \"inference checkpoints\" and \"training\ncheckpoints\" or something if this turns out to be a useful distinction.\n\nI ended up not adding a test that training continues exactly as it would\nhave without checkpointing/restoring because I think there are other\nbugs that are not critical to solve right now like rand seeds not being\nsaved. (I started a [discussion\nhere](https://openathena.slack.com/archives/C0884476QSC/p1753229734048809))\n\nFixes #282",
          "timestamp": "2025-07-29T15:00:44Z",
          "tree_id": "284b778022da90fdfb58e7ec80b48727fb5e176c",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/fca69bcf55ee435edceb52595536b733242cfebc"
        },
        "date": 1753802811118,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2025657375929018,
            "unit": "iter/sec",
            "range": "stddev: 0.03727358835674056",
            "extra": "mean: 831.5553725999507 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07032378068391941,
            "unit": "iter/sec",
            "range": "stddev: 0.5707032375818843",
            "extra": "mean: 14.219940826199991 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1937617335287812,
            "unit": "iter/sec",
            "range": "stddev: 0.08881723658146731",
            "extra": "mean: 5.1609777730000586 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.015140074807560123,
            "unit": "iter/sec",
            "range": "stddev: 0.2054239991477934",
            "extra": "mean: 66.04987179460004 sec\nrounds: 5"
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
          "id": "ac92b9abf9c99815dae6b9ae91027ea6ce7ec05f",
          "message": "Fix UnresolvedLocation JSON schema to permit string (#325)\n\nThis was previously causing red squigglies in VS Code. It's annoying\nthat this annotation and the two `@model_*` annotations on this class\nare all required -- I spent some time looking for a more direct way to\nexpress \"this type serializes to/from a string\" in the pydantic docs but\neverything I tried was broken or more complicated than this.",
          "timestamp": "2025-08-04T12:04:10-04:00",
          "tree_id": "27307cf6f78e7b02e3b4a73430ebd435dbfd791b",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/ac92b9abf9c99815dae6b9ae91027ea6ce7ec05f"
        },
        "date": 1754325025724,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.272412534916527,
            "unit": "iter/sec",
            "range": "stddev: 0.025362461238486067",
            "extra": "mean: 785.9086361999744 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06892028345242499,
            "unit": "iter/sec",
            "range": "stddev: 0.2077078230826053",
            "extra": "mean: 14.50951664600002 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19561777703766528,
            "unit": "iter/sec",
            "range": "stddev: 0.10990864714594804",
            "extra": "mean: 5.112009834399942 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.015237023270356915,
            "unit": "iter/sec",
            "range": "stddev: 0.15449887949279195",
            "extra": "mean: 65.62961690460001 sec\nrounds: 5"
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
          "id": "97147185c31a4de7192cbd74e1de485391eb43c2",
          "message": "MSE Dynamic Loss (#310)\n\nAdds a new loss function to dynamically re-weight loss scaling to make\neach channel contribute equally to the loss.",
          "timestamp": "2025-08-04T19:22:39Z",
          "tree_id": "37dcb9e3e141071f2da6809e342144d2d2897d3c",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/97147185c31a4de7192cbd74e1de485391eb43c2"
        },
        "date": 1754336956212,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.3032773818159626,
            "unit": "iter/sec",
            "range": "stddev: 0.03296609410294671",
            "extra": "mean: 767.2963667999966 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07143451981573333,
            "unit": "iter/sec",
            "range": "stddev: 0.23848597425655643",
            "extra": "mean: 13.998834213199984 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19246492035068932,
            "unit": "iter/sec",
            "range": "stddev: 0.08808535006299062",
            "extra": "mean: 5.195752026800028 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.015042709341925158,
            "unit": "iter/sec",
            "range": "stddev: 0.41800356395116245",
            "extra": "mean: 66.47738630520003 sec\nrounds: 5"
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
          "id": "624a32a1a10d4e7fd3330079f6d8d99aff013a03",
          "message": "Sky Train: Cloning data faster via setup section instead of file mount (#319)\n\nSince `file_mount` options don't let you pass arguments to rclone, this\nPR opts to manually `rclone` data during the `setup` stage of the job.\nThis allows us to make faster file transfers with the `--transfers`\nflag.\n\nIt also has required that I map the `data/` dir to home and not root. \nWe also set the default number of workers to 4.",
          "timestamp": "2025-08-04T20:58:11Z",
          "tree_id": "6729e53b5adf4b38607cec411fd3be8a77c44535",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/624a32a1a10d4e7fd3330079f6d8d99aff013a03"
        },
        "date": 1754342608405,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2828312750438569,
            "unit": "iter/sec",
            "range": "stddev: 0.025142393325363884",
            "extra": "mean: 779.5257408000225 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06937405187453684,
            "unit": "iter/sec",
            "range": "stddev: 0.5989782048169544",
            "extra": "mean: 14.414611414199976 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1929518161048604,
            "unit": "iter/sec",
            "range": "stddev: 0.08914574576902971",
            "extra": "mean: 5.182641035399979 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.015528920569283422,
            "unit": "iter/sec",
            "range": "stddev: 0.04680770934217432",
            "extra": "mean: 64.39597623920004 sec\nrounds: 5"
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
          "id": "43b541756668c9062e25026177d197b7c9941114",
          "message": "Fix initial ema creation in distributed mode (#326)",
          "timestamp": "2025-08-04T23:28:10Z",
          "tree_id": "fe07c806709f894fe23173b0a224c3375672af53",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/43b541756668c9062e25026177d197b7c9941114"
        },
        "date": 1754351616139,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2921929106680938,
            "unit": "iter/sec",
            "range": "stddev: 0.031581478231401304",
            "extra": "mean: 773.8782590000255 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07164088522409555,
            "unit": "iter/sec",
            "range": "stddev: 0.1627372572568147",
            "extra": "mean: 13.958509821200005 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1954109914982962,
            "unit": "iter/sec",
            "range": "stddev: 0.06766687358696127",
            "extra": "mean: 5.1174194058000015 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.015469681086497194,
            "unit": "iter/sec",
            "range": "stddev: 0.2197670780185584",
            "extra": "mean: 64.64257371620002 sec\nrounds: 5"
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
          "id": "3904f8c8e5eb19e0cc0c9db7f3cf4dc568c0391f",
          "message": "Standard deviations should be stds, not means. 😬 (#327)",
          "timestamp": "2025-08-05T14:56:15Z",
          "tree_id": "11793635557474a6b941fe4b5717a5f6ab4e589a",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/3904f8c8e5eb19e0cc0c9db7f3cf4dc568c0391f"
        },
        "date": 1754407411851,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2922671969957815,
            "unit": "iter/sec",
            "range": "stddev: 0.04203403745060484",
            "extra": "mean: 773.8337723999848 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06891197410793636,
            "unit": "iter/sec",
            "range": "stddev: 0.3779521925294794",
            "extra": "mean: 14.511266190600008 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19010576521396627,
            "unit": "iter/sec",
            "range": "stddev: 0.11211138932641501",
            "extra": "mean: 5.260229740399973 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.014532266275585976,
            "unit": "iter/sec",
            "range": "stddev: 0.07268402340678563",
            "extra": "mean: 68.81239175199998 sec\nrounds: 5"
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
          "id": "f4adaeacda884121f7b1fa9ca7c54dad16743346",
          "message": "Add .git to .gitignore for the sake of skypilot (#328)\n\nWithout this, it seems that skypilot attempts to upload the whole .git\nfolder. Adding this makes it skip doing so. To me, this is a bug in\nskypilot; I'll file an issue over there and use this as a workaround for\nthe time being.",
          "timestamp": "2025-08-05T08:28:55-07:00",
          "tree_id": "a620d42ea51ca5eaf5f8306da3c022ea0cc09662",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/f4adaeacda884121f7b1fa9ca7c54dad16743346"
        },
        "date": 1754409299467,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2962735483758623,
            "unit": "iter/sec",
            "range": "stddev: 0.018471366172256",
            "extra": "mean: 771.4421090000087 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07159006428740924,
            "unit": "iter/sec",
            "range": "stddev: 0.33722447511763926",
            "extra": "mean: 13.9684188016 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.193611379048595,
            "unit": "iter/sec",
            "range": "stddev: 0.0864233172533147",
            "extra": "mean: 5.164985678600056 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.015422082771636966,
            "unit": "iter/sec",
            "range": "stddev: 0.23462798063935392",
            "extra": "mean: 64.84208487320002 sec\nrounds: 5"
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
          "id": "3b96a83248b45b68304764e0a2122b54535f86d3",
          "message": "Stop complaining about long lines (#333)\n\nThis rule seems low-value to me, and we already enforce `ruff format`\nwhich will wrap lines when reasonable. If ruff format is OK with the\nline length I don't think we should be wrapping things by hand.",
          "timestamp": "2025-08-06T12:57:24-04:00",
          "tree_id": "062a5871b5796fbe6a941a083e3934e5caf5fd67",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/3b96a83248b45b68304764e0a2122b54535f86d3"
        },
        "date": 1754501036166,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.3270880072726208,
            "unit": "iter/sec",
            "range": "stddev: 0.014590078156623198",
            "extra": "mean: 753.5295281999879 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06861328927605471,
            "unit": "iter/sec",
            "range": "stddev: 0.6857639217565965",
            "extra": "mean: 14.57443609759996 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.18959465686552573,
            "unit": "iter/sec",
            "range": "stddev: 0.10057675580168797",
            "extra": "mean: 5.274410241999976 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.014785003130308394,
            "unit": "iter/sec",
            "range": "stddev: 0.46626679847645386",
            "extra": "mean: 67.63610336679999 sec\nrounds: 5"
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
          "id": "c9b025a34d85f91d65aca4724a6c0a8aa41a1d65",
          "message": "By default, our training runs should use A100 machines with high GPU memory (#332)",
          "timestamp": "2025-08-06T17:14:28Z",
          "tree_id": "05b48b967dca12d7516baea586ef0aaaffe36035",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/c9b025a34d85f91d65aca4724a6c0a8aa41a1d65"
        },
        "date": 1754502095596,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.3127154188974697,
            "unit": "iter/sec",
            "range": "stddev: 0.009804109754638302",
            "extra": "mean: 761.7797320000136 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07008440013616499,
            "unit": "iter/sec",
            "range": "stddev: 0.5029561332995799",
            "extra": "mean: 14.268510510999999 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19058012312490005,
            "unit": "iter/sec",
            "range": "stddev: 0.08347261161686767",
            "extra": "mean: 5.247136918600018 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.014522573862611852,
            "unit": "iter/sec",
            "range": "stddev: 0.06653110597891153",
            "extra": "mean: 68.85831736580008 sec\nrounds: 5"
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
          "id": "ba0c18f9979fcd0f7143e09d7179035e2a693744",
          "message": "Work around ema name issues (#335)\n\nI would love a more principled strategy here, but this:\n* Ensures that checkpoints which include ema params are rewritten on\nload to not have a \"module\" prefix on those params\n* Ensures that EmaTrackers always store parameters without a \"module\"\nprefix.\n* Ensures that using an EmaTracker works either with or without a\n\"module.\" prefix on the requested parameters.\n\nCloses #329",
          "timestamp": "2025-08-08T14:11:52-04:00",
          "tree_id": "67ba61c66ed69a2fe95c3d1aeb4a8fcdbbcb36dd",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/ba0c18f9979fcd0f7143e09d7179035e2a693744"
        },
        "date": 1754678266883,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2667408271638718,
            "unit": "iter/sec",
            "range": "stddev: 0.016789156367289917",
            "extra": "mean: 789.427465000017 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07018096828943475,
            "unit": "iter/sec",
            "range": "stddev: 0.5240801681589701",
            "extra": "mean: 14.248877215199991 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1938044968084834,
            "unit": "iter/sec",
            "range": "stddev: 0.08862174954693228",
            "extra": "mean: 5.1598389947999745 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.014888241905281242,
            "unit": "iter/sec",
            "range": "stddev: 0.22374753660095703",
            "extra": "mean: 67.1670977918 sec\nrounds: 5"
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
          "id": "67dcd0676a9a6bef712c6d29ee0abeb6220c98b7",
          "message": "Add a scheduler which can run a cosine schedule followed by a flat tail (#334)\n\nTo try extending the behavior seen greatly improving loss at low\nlearning rates from [this\nrun](https://openathena.slack.com/archives/C08CYM42DT3/p1753308638843249?thread_ts=1752275713.570969&cid=C08CYM42DT3).",
          "timestamp": "2025-08-08T18:19:36Z",
          "tree_id": "5aef0b8e92659742f4e1c09d7a901ae43089f905",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/67dcd0676a9a6bef712c6d29ee0abeb6220c98b7"
        },
        "date": 1754678706659,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.3223606718711887,
            "unit": "iter/sec",
            "range": "stddev: 0.02757898776680775",
            "extra": "mean: 756.2233370000058 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07027482588797575,
            "unit": "iter/sec",
            "range": "stddev: 0.8177832977259908",
            "extra": "mean: 14.22984671060001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19919401143837862,
            "unit": "iter/sec",
            "range": "stddev: 0.10961546798935871",
            "extra": "mean: 5.020231244800016 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.01559656739982536,
            "unit": "iter/sec",
            "range": "stddev: 0.05305568902163035",
            "extra": "mean: 64.11667223719992 sec\nrounds: 5"
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
          "id": "6e5ba43915107283105350114a69966f932e3645",
          "message": "Un-ignoring .git folder for skypilot syncs (#338)\n\nI think this could address #337.",
          "timestamp": "2025-08-08T18:34:59Z",
          "tree_id": "83455ec4edd235c794aa4ce54c82240c15c974c4",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/6e5ba43915107283105350114a69966f932e3645"
        },
        "date": 1754679639773,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.3176444271558383,
            "unit": "iter/sec",
            "range": "stddev: 0.013924774756838856",
            "extra": "mean: 758.9300872000194 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06953904518863728,
            "unit": "iter/sec",
            "range": "stddev: 0.3653400606165202",
            "extra": "mean: 14.3804102758 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1946027649401368,
            "unit": "iter/sec",
            "range": "stddev: 0.08511305498608263",
            "extra": "mean: 5.13867313400001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.01537576592990394,
            "unit": "iter/sec",
            "range": "stddev: 0.2667918273720684",
            "extra": "mean: 65.0374104652 sec\nrounds: 5"
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
          "id": "dd3d28bffaf1a5b4a46e3d5b6288583afc0207fb",
          "message": "Avoid chunking lat/lon (#342)",
          "timestamp": "2025-08-11T17:54:37-04:00",
          "tree_id": "260baac65f6e988c4442f12409b08abb08b08c64",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/dd3d28bffaf1a5b4a46e3d5b6288583afc0207fb"
        },
        "date": 1754950847049,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.31452882739347,
            "unit": "iter/sec",
            "range": "stddev: 0.018363193129171895",
            "extra": "mean: 760.7288475999894 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.07062339826303135,
            "unit": "iter/sec",
            "range": "stddev: 0.33360118692404034",
            "extra": "mean: 14.159613167799966 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19493306050573803,
            "unit": "iter/sec",
            "range": "stddev: 0.0990079104818576",
            "extra": "mean: 5.129966140200031 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.015457816955573866,
            "unit": "iter/sec",
            "range": "stddev: 0.4170202101556956",
            "extra": "mean: 64.69218796380005 sec\nrounds: 5"
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
          "id": "ea4fb9343b5b9c2fb5da5fd9789fefc7c818c995",
          "message": "Vizualization code (#339)\n\nThis is an import and reorganization of the code from [this\nnotebook](https://github.com/Open-Athena/Ocean_Emulator/blob/fixed-branch/Notebooks/notebooks/2025-04-08-Samudra_Eval_OM4.ipynb).\nThe goal was to get it to a state we could check it in and iterate from\nthere. This PR:\n\n* Moves that code from a notebook to python code in\nsrc/ocean_emulators/viz/core.py\n* Creates a new TopLevelConfig for visualization in\nsrc/ocean_emulators/viz/config.py (see CONTRIBUTING.md for an example\ninvocation)\n* Does a minimal amount of deduplication, cleanup, and typing of the\ncode there to get it passing CI and to allow the config to configure all\nthe main knobs. Also added some TODOs about clear things we might want\nto do next.\n* Includes a tool to compare old and new viz output -- this is how I\nverified that the previous notebook and this code produce nearly\nidentical results (modulo some FP error due to slight changes in\nordering, I think).\n* Includes pointers to the basin data, now in the emulator pod, which\nwas created by the new `notebooks/regrid_basins.py` code, which I\nincluded for posterity but don't think we really need to maintain per\nse.\n\nI would not try to review core.py too deeply but skimming the structure\nis good; same with notebooks/regrid_basins.py. Everything else is fair\ngame :)",
          "timestamp": "2025-08-12T13:09:11-04:00",
          "tree_id": "bbf84611af74016865ffa2229aea8022e132b186",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/ea4fb9343b5b9c2fb5da5fd9789fefc7c818c995"
        },
        "date": 1755020179171,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2600145455222236,
            "unit": "iter/sec",
            "range": "stddev: 0.02572824976400869",
            "extra": "mean: 793.6416318000056 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06721630406866494,
            "unit": "iter/sec",
            "range": "stddev: 0.5518593591118539",
            "extra": "mean: 14.877342839000017 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19440563397113855,
            "unit": "iter/sec",
            "range": "stddev: 0.11458262473824349",
            "extra": "mean: 5.14388384519998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.014501434345380546,
            "unit": "iter/sec",
            "range": "stddev: 1.0506186953968848",
            "extra": "mean: 68.95869582159999 sec\nrounds: 5"
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
          "id": "34e217b5b958a2eedad46369abe00a8c643092ff",
          "message": "New baseline model: Wide Samudra (training and eval config) (#341)\n\nSince I'm about to fork off experiments from these configurations, I\nthought it should be merged into main.\n\n---------\n\nCo-authored-by: Jesse Rusak <jesse@openathena.ai>",
          "timestamp": "2025-08-12T17:45:25Z",
          "tree_id": "bae84318a9d1acd821c6ffb35a2a7e2829d09028",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/34e217b5b958a2eedad46369abe00a8c643092ff"
        },
        "date": 1755022319562,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.244606398293361,
            "unit": "iter/sec",
            "range": "stddev: 0.05627953427440814",
            "extra": "mean: 803.4668641999815 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06980320538664524,
            "unit": "iter/sec",
            "range": "stddev: 0.325390397019778",
            "extra": "mean: 14.325989679999998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19487589238507327,
            "unit": "iter/sec",
            "range": "stddev: 0.08326312470727584",
            "extra": "mean: 5.131471049400034 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.01501357798165184,
            "unit": "iter/sec",
            "range": "stddev: 0.11522161120091716",
            "extra": "mean: 66.60637465780005 sec\nrounds: 5"
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
          "id": "e4e5756144a42c22b2cd0043854da28cad0bd44d",
          "message": "minor tweaks to skypilot setup (#348)\n\n* Fixes numba bug with print being defined inside a function rather than\na module.\n* Fixes sqlalchemy required by skypilot\n* Some docs and convenience updates",
          "timestamp": "2025-08-14T14:19:31-07:00",
          "tree_id": "90bf8f2e0400125696c85234419ae9622f042201",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/e4e5756144a42c22b2cd0043854da28cad0bd44d"
        },
        "date": 1755207956682,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2669796825620196,
            "unit": "iter/sec",
            "range": "stddev: 0.024573617654159798",
            "extra": "mean: 789.2786394000041 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06781283463522375,
            "unit": "iter/sec",
            "range": "stddev: 0.509250022224479",
            "extra": "mean: 14.74647100919999 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.1963532289407739,
            "unit": "iter/sec",
            "range": "stddev: 0.0994255258944995",
            "extra": "mean: 5.092862518199945 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.014954221684960516,
            "unit": "iter/sec",
            "range": "stddev: 0.11591925564546583",
            "extra": "mean: 66.87074867999995 sec\nrounds: 5"
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
          "id": "4a73c3e4b7a5792158ecb87170efc3bfdc6d5f5a",
          "message": "Tweak AGENTS.md for some common annoyances (#349)",
          "timestamp": "2025-08-21T13:42:57-04:00",
          "tree_id": "ccdb95535f057b4c7dc696cb521358936914481a",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/4a73c3e4b7a5792158ecb87170efc3bfdc6d5f5a"
        },
        "date": 1755799805711,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2947179620209768,
            "unit": "iter/sec",
            "range": "stddev: 0.017697049215557883",
            "extra": "mean: 772.368986399988 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06856927086872036,
            "unit": "iter/sec",
            "range": "stddev: 0.4880806618628233",
            "extra": "mean: 14.583792234199995 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.19302028893105272,
            "unit": "iter/sec",
            "range": "stddev: 0.1005085446128249",
            "extra": "mean: 5.180802523600005 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.014690572707488098,
            "unit": "iter/sec",
            "range": "stddev: 0.1444156743664937",
            "extra": "mean: 68.07086557559997 sec\nrounds: 5"
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
          "id": "70b29b97f87db3a89dc0ec107edbfaec772474c4",
          "message": "Eval + Viz Skypilot (#352)",
          "timestamp": "2025-08-26T19:55:18Z",
          "tree_id": "0ab45c6cbe943e2753d4f33103262119ce0c2f7e",
          "url": "https://github.com/Open-Athena/Ocean_Emulator/commit/70b29b97f87db3a89dc0ec107edbfaec772474c4"
        },
        "date": 1756239782271,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 1.2905610779975523,
            "unit": "iter/sec",
            "range": "stddev: 0.030642796003654924",
            "extra": "mean: 774.8567790000379 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.06942793502438338,
            "unit": "iter/sec",
            "range": "stddev: 0.42194323869175515",
            "extra": "mean: 14.40342420740003 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.18841184378582257,
            "unit": "iter/sec",
            "range": "stddev: 0.12661527550668766",
            "extra": "mean: 5.307521968399987 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-extra_config_args0-mock-train_default.test.yaml]",
            "value": 0.013913667549041636,
            "unit": "iter/sec",
            "range": "stddev: 0.5441723810107689",
            "extra": "mean: 71.87177618519995 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}