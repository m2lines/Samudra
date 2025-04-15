window.BENCHMARK_DATA = {
  "lastUpdate": 1744736562571,
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
          "id": "97f2bb5015b51f43ffffd5435c924eb8b1438cd4",
          "message": "Ensuring uniqueness property of interpretable data. (#98)\n\nAlso, added a minor equality fix which tolerates noisier data. Luckily,\nthe uniqueness property is still preserved.",
          "timestamp": "2025-03-06T16:49:08-08:00",
          "tree_id": "4b7d7f3d7897fb96084d8ff265693f58ba2628e9",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/97f2bb5015b51f43ffffd5435c924eb8b1438cd4"
        },
        "date": 1741310516183,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cpu-train]",
            "value": 0.08404096600870015,
            "unit": "iter/sec",
            "range": "stddev: 0.0850279938238179",
            "extra": "mean: 11.898958894600014 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cpu-val]",
            "value": 0.23398952639640067,
            "unit": "iter/sec",
            "range": "stddev: 0.06960598422412712",
            "extra": "mean: 4.2736955597999895 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu]",
            "value": 0.2120294014564163,
            "unit": "iter/sec",
            "range": "stddev: 0.08449484078531101",
            "extra": "mean: 4.7163270430000015 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu]",
            "value": 0.00914679521936642,
            "unit": "iter/sec",
            "range": "stddev: 0.21895924952592646",
            "extra": "mean: 109.3279095046 sec\nrounds: 5"
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
          "id": "9bf9391bb918a926b0d2c530fde14fae0d78cfe2",
          "message": "Added some typing definitions (array and otherwise). (#97)\n\nA pre-requisite step in refactoring the data loader (for me, anyway) is\njust understanding what's going on. To that end, I've added\nbeartype-enforced jaxtyping and standard typing definitions to parts of\nthe codebase that the data loader touches.\n\nThis does not aim for complete typing coverage of the codebase. Rather,\nthese annotations have been added so I can feel confident in pulling\narray parsing routines up and down. Besides serving as useful\ndocumentation, these perform runtime type checks to ensure correctness.\n\n---------\n\nCo-authored-by: Jesse Rusak <jesse@openathena.ai>",
          "timestamp": "2025-03-07T12:55:01-08:00",
          "tree_id": "6b4029a4aae29a5d17c745b3d85bfc41f1296dfb",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/9bf9391bb918a926b0d2c530fde14fae0d78cfe2"
        },
        "date": 1741382894958,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cpu-train]",
            "value": 0.08394506414824304,
            "unit": "iter/sec",
            "range": "stddev: 0.09293984653646348",
            "extra": "mean: 11.912552693200007 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cpu-val]",
            "value": 0.23394144173227752,
            "unit": "iter/sec",
            "range": "stddev: 0.06995500765078581",
            "extra": "mean: 4.27457398140001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu]",
            "value": 0.21061980932866559,
            "unit": "iter/sec",
            "range": "stddev: 0.09174227095230708",
            "extra": "mean: 4.747891488399989 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu]",
            "value": 0.008990542264410005,
            "unit": "iter/sec",
            "range": "stddev: 1.9132895272672001",
            "extra": "mean: 111.22799610860002 sec\nrounds: 5"
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
          "id": "0aceb9715af89c643c99944c2c06358d4e48faeb",
          "message": "fix training when run by main script (#100)\n\nTurns out the handle_logging requires the directory to be there first,\nso create them first (but also try again to recreate them in the\nconstructor)",
          "timestamp": "2025-03-10T12:54:38-04:00",
          "tree_id": "8fed0ad94d9d50ba2de540cb6b793a7f92a27590",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/0aceb9715af89c643c99944c2c06358d4e48faeb"
        },
        "date": 1741627720461,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cpu-train]",
            "value": 0.08308876848398251,
            "unit": "iter/sec",
            "range": "stddev: 0.12454989519473743",
            "extra": "mean: 12.035320997600001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cpu-val]",
            "value": 0.23073466076764376,
            "unit": "iter/sec",
            "range": "stddev: 0.07502369872988854",
            "extra": "mean: 4.333982578399992 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu]",
            "value": 0.20852099929602827,
            "unit": "iter/sec",
            "range": "stddev: 0.08751779490152026",
            "extra": "mean: 4.795680067600017 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu]",
            "value": 0.008834935520368131,
            "unit": "iter/sec",
            "range": "stddev: 0.14006964569640215",
            "extra": "mean: 113.18701734659999 sec\nrounds: 5"
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
          "id": "a97aedbdefcb457c38975dcf92760f06ae6bd8fe",
          "message": "Simple utility script to clone Samudra data locally. (#107)\n\nClones OM4 in about 600s using ~100MB chunks. Saves means and stds as\nNetCDF. Is packaged as a `uv` script (it self-describes dependencies).\n\nThis change is in service of setting up remote GPU runs and chunk\ntuning.\n\n---------\n\nCo-authored-by: Surya Dheeshjith <41594351+suryadheeshjith@users.noreply.github.com>",
          "timestamp": "2025-03-12T14:58:58-07:00",
          "tree_id": "cbaadc70bab29f387c6851c16096922d6698e51d",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/a97aedbdefcb457c38975dcf92760f06ae6bd8fe"
        },
        "date": 1741818769656,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cpu-train]",
            "value": 0.08425397039594708,
            "unit": "iter/sec",
            "range": "stddev: 0.10199128897160649",
            "extra": "mean: 11.868876864799995 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cpu-val]",
            "value": 0.2342176528625997,
            "unit": "iter/sec",
            "range": "stddev: 0.09519466891075043",
            "extra": "mean: 4.269533008200005 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu]",
            "value": 0.2098116748303666,
            "unit": "iter/sec",
            "range": "stddev: 0.09907749668723016",
            "extra": "mean: 4.766179006999982 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu]",
            "value": 0.008902273599793342,
            "unit": "iter/sec",
            "range": "stddev: 0.759238922851277",
            "extra": "mean: 112.33085444859996 sec\nrounds: 5"
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
          "id": "01174fc821dbbb67f970d39642a228e9678e6794",
          "message": "Upgrade to python 3.11, zarr 3, and various other updates (#108)\n\nI originally did this for performance reasons (which [didn't pan out\nyet](https://github.com/suryadheeshjith/Ocean_Emulator/issues/106#issuecomment-2717976095))\nbut seems like it's healthy to do this anyway.",
          "timestamp": "2025-03-12T20:25:45-04:00",
          "tree_id": "b1ded0cbe7f52ca0b73bf55a3adc2ad821c76d2e",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/01174fc821dbbb67f970d39642a228e9678e6794"
        },
        "date": 1741828298022,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cpu-train]",
            "value": 0.0411522478633058,
            "unit": "iter/sec",
            "range": "stddev: 0.17979805702965126",
            "extra": "mean: 24.30000915920001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cpu-val]",
            "value": 0.11465575245289567,
            "unit": "iter/sec",
            "range": "stddev: 0.12203664141318428",
            "extra": "mean: 8.721760388000007 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu]",
            "value": 0.11195474981538805,
            "unit": "iter/sec",
            "range": "stddev: 0.10314289492338011",
            "extra": "mean: 8.932180203600002 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu]",
            "value": 0.0061999992218993525,
            "unit": "iter/sec",
            "range": "stddev: 2.4311620349088625",
            "extra": "mean: 161.29034282260002 sec\nrounds: 5"
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
          "id": "3d56ca15da553046e13312b57b682617f96021e3",
          "message": "Fix: Opening with consolidated metadata. (#117)\n\nI expect that this will fix the performance regression from switching\nbetween Zarr v2 --> Zarr v3.",
          "timestamp": "2025-03-14T14:59:55-07:00",
          "tree_id": "b13231b0e0470131ad33bec96a8355d4398e15da",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/3d56ca15da553046e13312b57b682617f96021e3"
        },
        "date": 1741992449913,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cpu-train]",
            "value": 0.040724808720175136,
            "unit": "iter/sec",
            "range": "stddev: 0.14400221591182144",
            "extra": "mean: 24.555057013800003 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cpu-val]",
            "value": 0.11405716334819677,
            "unit": "iter/sec",
            "range": "stddev: 0.2164257567748326",
            "extra": "mean: 8.767533494999986 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu]",
            "value": 0.11190599314736037,
            "unit": "iter/sec",
            "range": "stddev: 0.1561394754054805",
            "extra": "mean: 8.936071892800033 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu]",
            "value": 0.006005123903776435,
            "unit": "iter/sec",
            "range": "stddev: 0.8931699472865272",
            "extra": "mean: 166.5244574506 sec\nrounds: 5"
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
          "id": "060cd1355ab2faa143c31b2a08b2ab2f5a45f33d",
          "message": "Rework round-tripping/interpretable data with bit fields (#118)\n\nWhen I went to make another change, this test started failing again, I\nthink because it generated new test cases. Try to resolve the\ninstability by avoiding complexity of rounding and numeric error by\npacking data as smaller bitfields into the float64.",
          "timestamp": "2025-03-19T15:22:57-04:00",
          "tree_id": "6c9bbf536c4b5ca330bd2e226e4037795309748b",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/060cd1355ab2faa143c31b2a08b2ab2f5a45f33d"
        },
        "date": 1742415149940,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cpu-train]",
            "value": 0.03718484375862242,
            "unit": "iter/sec",
            "range": "stddev: 0.17812617612684606",
            "extra": "mean: 26.892677201799994 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cpu-val]",
            "value": 0.10297281333046318,
            "unit": "iter/sec",
            "range": "stddev: 0.15248712441960743",
            "extra": "mean: 9.711301145000016 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu]",
            "value": 0.10333347877382679,
            "unit": "iter/sec",
            "range": "stddev: 0.09718891536038349",
            "extra": "mean: 9.677405734000013 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu]",
            "value": 0.005849237628025825,
            "unit": "iter/sec",
            "range": "stddev: 0.6666653090527466",
            "extra": "mean: 170.96245076600005 sec\nrounds: 5"
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
          "id": "dd81ee6f1de5f290236b2f5afc6561e498ee1a58",
          "message": "Downgrading to Zarr v2. (#130)\n\nThis should be the same performance as before (from my tests, it should\neven be a little bit faster). Further, Zarr v2 is much more stable.",
          "timestamp": "2025-03-20T12:32:37-07:00",
          "tree_id": "ed8fea33a3ecc9a2eebd79494c18992bec151afd",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/dd81ee6f1de5f290236b2f5afc6561e498ee1a58"
        },
        "date": 1742501336268,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cpu-train]",
            "value": 0.06926286555390386,
            "unit": "iter/sec",
            "range": "stddev: 0.09716930112700886",
            "extra": "mean: 14.437750907400005 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cpu-val]",
            "value": 0.1901690516718983,
            "unit": "iter/sec",
            "range": "stddev: 0.10144801979801497",
            "extra": "mean: 5.25847918579999 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu]",
            "value": 0.1728603983281346,
            "unit": "iter/sec",
            "range": "stddev: 0.12070860950709696",
            "extra": "mean: 5.785015016 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu]",
            "value": 0.008142442819663262,
            "unit": "iter/sec",
            "range": "stddev: 2.1802006749613585",
            "extra": "mean: 122.8132665034 sec\nrounds: 5"
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
          "id": "464c1d08f911f0ccdf1f50b74ec8c9f4245226b3",
          "message": "Move everything into src/ocean_emulators, get rid of ocean_emulators_main (#110)\n\nFollow-up to https://github.com/suryadheeshjith/Ocean_Emulator/pull/108\nwhich moves everything into a named module and fixes #75.",
          "timestamp": "2025-03-20T15:37:54-04:00",
          "tree_id": "ec51f3a9c93c18a3be45d1bbc3719dd212d05df7",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/464c1d08f911f0ccdf1f50b74ec8c9f4245226b3"
        },
        "date": 1742501580746,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cpu-train]",
            "value": 0.0677815428196307,
            "unit": "iter/sec",
            "range": "stddev: 0.04984107001387765",
            "extra": "mean: 14.753278819000013 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cpu-val]",
            "value": 0.19113526718203994,
            "unit": "iter/sec",
            "range": "stddev: 0.08103844217343963",
            "extra": "mean: 5.231896838 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu]",
            "value": 0.17371901280342741,
            "unit": "iter/sec",
            "range": "stddev: 0.09700559237551103",
            "extra": "mean: 5.756422304400007 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu]",
            "value": 0.008384324064819051,
            "unit": "iter/sec",
            "range": "stddev: 0.3398654410476008",
            "extra": "mean: 119.27019903679995 sec\nrounds: 5"
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
          "id": "d84192f5b04080375647ef2929d9c9c5313a504a",
          "message": "Restore missing function (#131)",
          "timestamp": "2025-03-20T21:59:35-04:00",
          "tree_id": "8db37413cbea31929537d2bb59230a31e6956af0",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/d84192f5b04080375647ef2929d9c9c5313a504a"
        },
        "date": 1742524621749,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cpu-train]",
            "value": 0.06501608976774323,
            "unit": "iter/sec",
            "range": "stddev: 0.13250071066471172",
            "extra": "mean: 15.380808098 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cpu-val]",
            "value": 0.17984825253694336,
            "unit": "iter/sec",
            "range": "stddev: 0.08954489225997446",
            "extra": "mean: 5.560243071000014 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu]",
            "value": 0.16611422123234867,
            "unit": "iter/sec",
            "range": "stddev: 0.0939620964614289",
            "extra": "mean: 6.0199541771999865 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu]",
            "value": 0.007744701107258786,
            "unit": "iter/sec",
            "range": "stddev: 0.14222873978908288",
            "extra": "mean: 129.1205414064 sec\nrounds: 5"
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
          "id": "151275c56c72857b639e3312eaa0abe8e97f3a46",
          "message": "fix build, not sure how this was working in CI before (#133)\n\nCurrently, `uv run python` doesn't let you run `import ocean_emulators`.\nThis fixes that.",
          "timestamp": "2025-03-21T10:44:15-04:00",
          "tree_id": "b97592be8143c1817d59f2b91723d921274fcb14",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/151275c56c72857b639e3312eaa0abe8e97f3a46"
        },
        "date": 1742570485799,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cpu-train]",
            "value": 0.06795110596832138,
            "unit": "iter/sec",
            "range": "stddev: 0.034761535747241765",
            "extra": "mean: 14.716463930200007 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cpu-val]",
            "value": 0.18685259939420026,
            "unit": "iter/sec",
            "range": "stddev: 0.08762517505957977",
            "extra": "mean: 5.351812087400049 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu]",
            "value": 0.1704194952420998,
            "unit": "iter/sec",
            "range": "stddev: 0.12162227048780766",
            "extra": "mean: 5.867873265200023 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu]",
            "value": 0.008249294397808906,
            "unit": "iter/sec",
            "range": "stddev: 0.11374542200399147",
            "extra": "mean: 121.2224890732 sec\nrounds: 5"
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
          "id": "e37711d4d1f172bf499d0f6d2fd39029fb23f4bd",
          "message": "Restore benchmarks to old behavior (#135)\n\nThis was a silly bug but after looking at this we probably want to leave\nthem separate so the benchmark web format doesn't change.",
          "timestamp": "2025-03-21T09:24:29-07:00",
          "tree_id": "4c5e2fa60fb5dda47a5c12e2d6803b748ab7676f",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/e37711d4d1f172bf499d0f6d2fd39029fb23f4bd"
        },
        "date": 1742576455293,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cpu-train]",
            "value": 0.06968540876965415,
            "unit": "iter/sec",
            "range": "stddev: 0.03573396522040372",
            "extra": "mean: 14.350206415600008 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cpu-val]",
            "value": 0.19239538866398495,
            "unit": "iter/sec",
            "range": "stddev: 0.08894272661080597",
            "extra": "mean: 5.197629771399988 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_cm4.test.yaml-cpu]",
            "value": 0.17363497899943983,
            "unit": "iter/sec",
            "range": "stddev: 0.08641981433751587",
            "extra": "mean: 5.759208229600017 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_cm4.test.yaml-cpu]",
            "value": 0.008483200656788419,
            "unit": "iter/sec",
            "range": "stddev: 0.18481124039439273",
            "extra": "mean: 117.88003613939993 sec\nrounds: 5"
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
          "id": "763edcb8ac6f77dae876b94f33cfa928c0c370bf",
          "message": "Speed up tests (#136)\n\nCuts test time on my machine from 1:45 to :40. This does change the\nbenchmark test run, I could break out a separate config for that if\ndesired to keep it consistent. WDYT @alxmrs?\n\nBefore:\n```\n==================================================================================== slowest durations =====================================================================================\n67.17s call     tests/test_trainer.py::test_trainer__mini_2step[train_cm4_2step.test.yaml-cpu]\n6.48s call     tests/test_datasets.py::test__data_is_not_zeros[train_cm4.test.yaml-cpu-train]\n6.42s call     tests/test_datasets.py::test_train__data_shape[train_cm4.test.yaml-cpu]\n6.26s call     tests/test_datasets.py::test_train__loads_correct_number_of_samples[train_cm4.test.yaml-cpu]\n2.57s call     tests/test_datasets.py::test_inference__data_is_not_zero[train_cm4.test.yaml-cpu]\n2.48s call     tests/test_datasets.py::test__data_is_not_zeros[train_cm4.test.yaml-cpu-val]\n2.41s call     tests/test_datasets.py::test_inference__data_shape[train_cm4.test.yaml-cpu]\n2.37s call     tests/test_datasets.py::test_val__data_shape[train_cm4.test.yaml-cpu]\n2.23s call     tests/test_datasets.py::test_val__loads_correct_number_of_samples[train_cm4.test.yaml-cpu]\n1.86s setup    tests/test_datasets.py::test_test_util__data_source_roundtrip[train_cm4.test.yaml-cpu]\n1.61s setup    tests/test_datasets.py::test_test_util__data_source_roundtrip[train_cm4_2step.test.yaml-cpu]\n0.79s setup    tests/test_datasets.py::test_train__loads_correct_number_of_samples[train_cm4.test.yaml-cpu]\n0.36s call     tests/test_datasets.py::test_test_util__data_source_roundtrip[train_cm4.test.yaml-cpu]\n0.28s setup    tests/test_datasets.py::test_train__loads_correct_number_of_samples[train_cm4_2step.test.yaml-cpu]\n0.14s teardown tests/test_trainer.py::test_trainer__mini_2step[train_cm4_2step.test.yaml-cpu]\n\n(65 durations < 0.005s hidden.  Use -vv to show these durations.)\n========================================================== 16 passed, 16 skipped, 52 deselected, 17 warnings in 104.09s (0:01:44) ==========================================================\npytest -m 'not cuda and not manual' --durations=0  326.39s user 636.78s system 914% cpu 1:45.30 total'\n```\n\nAfter:\n```\n==================================================================================== slowest durations =====================================================================================\n18.48s call     tests/test_trainer.py::test_trainer__mini_2step[train_cm4_2step.test.yaml-cpu]\n2.46s call     tests/test_datasets.py::test_train__data_shape[train_cm4.test.yaml-cpu]\n2.40s call     tests/test_datasets.py::test__data_is_not_zeros[train_cm4.test.yaml-cpu-train]\n2.27s call     tests/test_datasets.py::test_train__loads_correct_number_of_samples[train_cm4.test.yaml-cpu]\n1.80s call     tests/test_datasets.py::test_val__loads_correct_number_of_samples[train_cm4.test.yaml-cpu]\n1.78s call     tests/test_datasets.py::test__data_is_not_zeros[train_cm4.test.yaml-cpu-val]\n1.76s call     tests/test_datasets.py::test_inference__data_is_not_zero[train_cm4.test.yaml-cpu]\n1.75s call     tests/test_datasets.py::test_val__data_shape[train_cm4.test.yaml-cpu]\n1.65s call     tests/test_datasets.py::test_inference__data_shape[train_cm4.test.yaml-cpu]\n1.26s setup    tests/test_datasets.py::test_test_util__data_source_roundtrip[train_cm4.test.yaml-cpu]\n0.92s setup    tests/test_datasets.py::test_test_util__data_source_roundtrip[train_cm4_2step.test.yaml-cpu]\n0.86s setup    tests/test_datasets.py::test_train__loads_correct_number_of_samples[train_cm4.test.yaml-cpu]\n0.66s call     tests/test_datasets.py::test_test_util__data_source_roundtrip[train_cm4.test.yaml-cpu]\n0.28s setup    tests/test_datasets.py::test_train__loads_correct_number_of_samples[train_cm4_2step.test.yaml-cpu]\n0.08s teardown tests/test_trainer.py::test_trainer__mini_2step[train_cm4_2step.test.yaml-cpu]\n\n(65 durations < 0.005s hidden.  Use -vv to show these durations.)\n=============================================================== 16 passed, 16 skipped, 52 deselected, 15 warnings in 39.08s ================================================================\npytest -m 'not cuda and not manual' --durations=0  138.88s user 226.06s system 906% cpu 40.267 total\n```",
          "timestamp": "2025-03-21T15:20:21-04:00",
          "tree_id": "e460bec79b4d8cf9088e7c58c454adada17e6b62",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/763edcb8ac6f77dae876b94f33cfa928c0c370bf"
        },
        "date": 1742586951730,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cpu-train]",
            "value": 0.06895847041618705,
            "unit": "iter/sec",
            "range": "stddev: 0.10562074025887465",
            "extra": "mean: 14.501481746399985 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cpu-val]",
            "value": 0.19087705689517837,
            "unit": "iter/sec",
            "range": "stddev: 0.09720028754913404",
            "extra": "mean: 5.238974323399998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_cm4.test.yaml-cpu]",
            "value": 0.16990271529585044,
            "unit": "iter/sec",
            "range": "stddev: 0.1201586284574341",
            "extra": "mean: 5.885721121400013 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_cm4.test.yaml-cpu]",
            "value": 0.008225420629993558,
            "unit": "iter/sec",
            "range": "stddev: 0.1301129745222928",
            "extra": "mean: 121.57432974960007 sec\nrounds: 5"
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
          "id": "3e60b05c5c81f03626e4cfac64f593f57ddcba0d",
          "message": "Update ruff to avoid fighting with VS Code version (#137)\n\nThese keep flipping back and forth for me since the VS Code extension\nversion is more recent.",
          "timestamp": "2025-03-21T16:20:22-04:00",
          "tree_id": "93de6f752a78e8021c83006d784af175ab86c07a",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/3e60b05c5c81f03626e4cfac64f593f57ddcba0d"
        },
        "date": 1742590536951,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cpu-train]",
            "value": 0.0689943936133953,
            "unit": "iter/sec",
            "range": "stddev: 0.04855285735382052",
            "extra": "mean: 14.493931283799986 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cpu-val]",
            "value": 0.19114971290495925,
            "unit": "iter/sec",
            "range": "stddev: 0.08944643846432378",
            "extra": "mean: 5.2315014488000084 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_cm4.test.yaml-cpu]",
            "value": 0.17357419613457148,
            "unit": "iter/sec",
            "range": "stddev: 0.0959993187788843",
            "extra": "mean: 5.761225010800013 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_cm4.test.yaml-cpu]",
            "value": 0.008339578588728018,
            "unit": "iter/sec",
            "range": "stddev: 0.46817362021820524",
            "extra": "mean: 119.9101356694 sec\nrounds: 5"
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
          "id": "0c4abfac895fd2788eac49061755d0c5aa406aa0",
          "message": "Resolve CFTime warning (#139)\n\nThe underlying [warning\n](https://github.com/Unidata/cftime/blob/eab0212df73decd5a420adaae7938b942d131c95/src/cftime/_cftime.pyx#L1151)\nwas due to us passing a date with year < 0 for a calendar that doesn't\nsupport that.\n\nThis was for 2 reasons: first, we had no minimum in the possible\ngenerated dates and second because the fromordinal function we were\nusing takes the number of days since year -4713 (yes, negative four\nthousand years BCE) but the python datetime toordinal produces the\nnumber of days since year 1. So, we have a minimum now and we are using\na conversion function which is a little less silly.\n\nFixes #112",
          "timestamp": "2025-03-24T09:08:43-04:00",
          "tree_id": "5da7886abe8dc19aaef0095065bae2d814b441f4",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/0c4abfac895fd2788eac49061755d0c5aa406aa0"
        },
        "date": 1742824204965,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cpu-train]",
            "value": 0.06205709284086216,
            "unit": "iter/sec",
            "range": "stddev: 0.12453614106018021",
            "extra": "mean: 16.114193466400014 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cpu-val]",
            "value": 0.1707943356537342,
            "unit": "iter/sec",
            "range": "stddev: 0.10951997899555739",
            "extra": "mean: 5.854995109600031 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_cm4.test.yaml-cpu]",
            "value": 0.1579723372091907,
            "unit": "iter/sec",
            "range": "stddev: 0.11583030001867972",
            "extra": "mean: 6.330222225399984 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_cm4.test.yaml-cpu]",
            "value": 0.007375407708434811,
            "unit": "iter/sec",
            "range": "stddev: 1.6514650812859053",
            "extra": "mean: 135.58572482120005 sec\nrounds: 5"
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
          "id": "eb98afdfdc4bf7a30a52d911c8ea63e0256e15f2",
          "message": "March refactor - renaming (#141)\n\n* Bunch of renaming and removal of unused experiment configs in\nconstants. The most important rename here is inputs -> prognostic, extra\n-> boundary and removal of output variables names/keys since they are\nthe same as input variables. This functionality is okay for now since\nthe codebase depends on this fact. I don't see a near future where this\nis going to change.\n* The scripts for training on gantry, empire ai etc. may not work and I\nam planning to fix that in a later PR\n\nFollowing Draft PR #134 \n\nCloses issue #80",
          "timestamp": "2025-03-24T12:48:06-04:00",
          "tree_id": "59391f69030ea4c0c6afe39efe1c8505d88b171d",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/eb98afdfdc4bf7a30a52d911c8ea63e0256e15f2"
        },
        "date": 1742837086487,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cpu-train]",
            "value": 0.06763295617984809,
            "unit": "iter/sec",
            "range": "stddev: 0.0963179671467952",
            "extra": "mean: 14.785691125799996 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cpu-val]",
            "value": 0.18655951594202988,
            "unit": "iter/sec",
            "range": "stddev: 0.09631010720272",
            "extra": "mean: 5.360219739799993 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_cm4.test.yaml-cpu]",
            "value": 0.1706063409341989,
            "unit": "iter/sec",
            "range": "stddev: 0.0850553650681752",
            "extra": "mean: 5.861446852000006 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_cm4.test.yaml-cpu]",
            "value": 0.008113011443881997,
            "unit": "iter/sec",
            "range": "stddev: 0.12853033699078015",
            "extra": "mean: 123.25879322579999 sec\nrounds: 5"
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
          "id": "9799852f5388729098c91ceba1ac5b3551e6af9e",
          "message": "Wrote an Xarray-based OM4 Dataloader. (#123)\n\nFixes #119. This new loader is behind a feature flag. After I manually\nprofile the new loader, I will make a follow-up PR to make this one the\ndefault.",
          "timestamp": "2025-03-24T10:52:58-07:00",
          "tree_id": "e4c10fbc9947ac98469f38c568fbe31e8bd30dbc",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/9799852f5388729098c91ceba1ac5b3551e6af9e"
        },
        "date": 1742841054274,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cpu-train]",
            "value": 0.06638999034847015,
            "unit": "iter/sec",
            "range": "stddev: 0.12375218936370307",
            "extra": "mean: 15.062511603799976 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cpu-val]",
            "value": 0.1882575890750037,
            "unit": "iter/sec",
            "range": "stddev: 0.10893222612356636",
            "extra": "mean: 5.311870851599986 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_cm4.test.yaml-cpu]",
            "value": 0.17137026453511092,
            "unit": "iter/sec",
            "range": "stddev: 0.10718736856648822",
            "extra": "mean: 5.835318062400006 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_cm4.test.yaml-cpu]",
            "value": 0.007689739785678892,
            "unit": "iter/sec",
            "range": "stddev: 4.689125988061758",
            "extra": "mean: 130.04341211419998 sec\nrounds: 5"
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
          "id": "e7a3cfffeef70b46b06c94bbcca028da122d9162",
          "message": "uv script to prototype Zarr opening + update to Zarr cloning script. (#111)\n\n# Experiments\n\nSo far, it looks like we _should_ be using Zarr v2 + time_chunks=700, as\nthis is the fastest way to open this particular dataset!\n\n## Opening Remote OM4 data\n```\n# Zarr v2 (I think this uses consolidated=True by default!)\nuv run scripts/open_zarr_tuning.py                        \n3.11.11 (main, Jan 14 2025, 23:36:41) [Clang 19.1.6 ]\nzarr-version=2.18.4,xarray-version=2025.1.2\nELAPSED: 7.5452s. Config: Namespace(target=None, n_iters=8, zarr_concurrency=None, time_chunks=None)\n\n# Zarr v2 + time=10\nuv run scripts/open_zarr_tuning.py --time_chunks=10\n3.11.11 (main, Jan 14 2025, 23:36:41) [Clang 19.1.6 ]\nzarr-version=2.18.4,xarray-version=2025.1.2\nELAPSED: 7.1839s. Config: Namespace(target=None, n_iters=8, zarr_concurrency=None, time_chunks=10)\n\n# Zarr v2 + time=100\nuv run scripts/open_zarr_tuning.py --time_chunks=100\n3.11.11 (main, Jan 14 2025, 23:36:41) [Clang 19.1.6 ]\nzarr-version=2.18.4,xarray-version=2025.1.2\nELAPSED: 7.1952s. Config: Namespace(target=None, n_iters=8, zarr_concurrency=None, time_chunks=100)\n\n# Zarr v2 + time=700\nuv run scripts/open_zarr_tuning.py --time_chunks=700\n3.11.11 (main, Jan 14 2025, 23:36:41) [Clang 19.1.6 ]\nzarr-version=2.18.4,xarray-version=2025.1.2\nELAPSED: 7.0575s. Config: Namespace(target=None, n_iters=8, zarr_concurrency=None, time_chunks=700)\n\n\n\n# Zarr v3 (defaults)\nuv run scripts/open_zarr_tuning.py \n3.11.11 (main, Jan 14 2025, 23:36:41) [Clang 19.1.6 ]\nzarr-version=3.0.5,xarray-version=2025.1.2\nELAPSED: 11.2039s. Config: Namespace(target=None, n_iters=8, zarr_concurrency=10, time_chunks=None)\n\n# Zarr v3 + manually setting `consolidated=True`\n uv run scripts/open_zarr_tuning.py \n3.11.11 (main, Jan 14 2025, 23:36:41) [Clang 19.1.6 ]\nzarr-version=3.0.5,xarray-version=2025.1.2\nELAPSED: 11.2342s. Config: Namespace(target=None, n_iters=8, zarr_concurrency=10, time_chunks=None)\n\n# Zarr v3 + concurrency=50\nuv run scripts/open_zarr_tuning.py --zarr_concurrency=50\n3.11.11 (main, Jan 14 2025, 23:36:41) [Clang 19.1.6 ]\nzarr-version=3.0.5,xarray-version=2025.1.2\nELAPSED: 10.9889s. Config: Namespace(target=None, n_iters=8, zarr_concurrency=50, time_chunks=None)\n\n# Zarr v3 + concurrency=100\nuv run scripts/open_zarr_tuning.py --zarr_concurrency=100\n3.11.11 (main, Jan 14 2025, 23:36:41) [Clang 19.1.6 ]\nzarr-version=3.0.5,xarray-version=2025.1.2\nELAPSED: 11.0583s. Config: Namespace(target=None, n_iters=8, zarr_concurrency=100, time_chunks=None)\n\n# Zarr v3 + time=10\nuv run scripts/open_zarr_tuning.py --time_chunks=10   \n3.11.11 (main, Jan 14 2025, 23:36:41) [Clang 19.1.6 ]\nzarr-version=3.0.5,xarray-version=2025.1.2\nELAPSED: 10.9641s. Config: Namespace(target=None, n_iters=8, zarr_concurrency=10, time_chunks=10)\n\n# Zarr v3 + time=100\nuv run scripts/open_zarr_tuning.py --time_chunks=100\n3.11.11 (main, Jan 14 2025, 23:36:41) [Clang 19.1.6 ]\nzarr-version=3.0.5,xarray-version=2025.1.2\nELAPSED: 10.7012s. Config: Namespace(target=None, n_iters=8, zarr_concurrency=10, time_chunks=100)\n\n# Zarr v3 + time=700\nuv run scripts/open_zarr_tuning.py --time_chunks=700\n3.11.11 (main, Jan 14 2025, 23:36:41) [Clang 19.1.6 ]\nzarr-version=3.0.5,xarray-version=2025.1.2\nELAPSED: 10.7541s. Config: Namespace(target=None, n_iters=8, zarr_concurrency=10, time_chunks=700)\n\n# Zarr v3 + time=700 + concurrency=100\nuv run scripts/open_zarr_tuning.py --time_chunks=700 --zarr_concurrency=100\n3.11.11 (main, Jan 14 2025, 23:36:41) [Clang 19.1.6 ]\nzarr-version=3.0.5,xarray-version=2025.1.2\nELAPSED: 10.7544s. Config: Namespace(target=None, n_iters=8, zarr_concurrency=100, time_chunks=700)\n```\n\n**TBD: Opening locally cloned OM4 data...**",
          "timestamp": "2025-03-24T14:36:33-07:00",
          "tree_id": "56a3d2bdbe750aa8daa38384211e433ea10c0a3a",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/e7a3cfffeef70b46b06c94bbcca028da122d9162"
        },
        "date": 1742854459328,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cpu-train]",
            "value": 0.06876749219873958,
            "unit": "iter/sec",
            "range": "stddev: 0.09713369215957661",
            "extra": "mean: 14.541754658000002 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cpu-val]",
            "value": 0.191268375757793,
            "unit": "iter/sec",
            "range": "stddev: 0.08011467813971818",
            "extra": "mean: 5.2282558266000025 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_cm4.test.yaml-cpu]",
            "value": 0.17293424888685172,
            "unit": "iter/sec",
            "range": "stddev: 0.0813211960834321",
            "extra": "mean: 5.782544559199982 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_cm4.test.yaml-cpu]",
            "value": 0.008194592413365854,
            "unit": "iter/sec",
            "range": "stddev: 0.8788788767632002",
            "extra": "mean: 122.03169475139998 sec\nrounds: 5"
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
          "id": "06969612418c78fecfc8dc05ba9961002ba600da",
          "message": "March refactors - functionality (#142)\n\n* Ability to train on OM4 or CM4\n* Added metadata\n* Reduced checkpointing\n* Removed unnecessary zos check in wandblogger\n* Logger fix\n* Logging in eval fix\n* Train/val/inference time range extraction fix\n\nFollowing Draft PR\nhttps://github.com/suryadheeshjith/Ocean_Emulator/pull/134\n\nCloses Issues #116 , #74 , #80 , #101 , #69 , #59\n\n---------\n\nCo-authored-by: Alex Merose <alex@openathena.ai>",
          "timestamp": "2025-03-25T15:06:28-04:00",
          "tree_id": "a27cae05f25dc0e2e6be4ca3871e69b9b9e23994",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/06969612418c78fecfc8dc05ba9961002ba600da"
        },
        "date": 1742931913645,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cpu-train]",
            "value": 0.06482969318918327,
            "unit": "iter/sec",
            "range": "stddev: 0.13672058912403892",
            "extra": "mean: 15.42503058100001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cpu-val]",
            "value": 0.17849006001453624,
            "unit": "iter/sec",
            "range": "stddev: 0.13974848701115505",
            "extra": "mean: 5.602552881199995 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_cm4.test.yaml-cpu]",
            "value": 0.16069923339358813,
            "unit": "iter/sec",
            "range": "stddev: 0.0824668727926253",
            "extra": "mean: 6.2228050432000375 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_cm4.test.yaml-cpu]",
            "value": 0.007492071814673759,
            "unit": "iter/sec",
            "range": "stddev: 1.0706218304741464",
            "extra": "mean: 133.47442800019994 sec\nrounds: 5"
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
          "id": "8ff6f801a7256089e93d1070d14fa90f0480978a",
          "message": "Jaxtyping minimum version is 0.3.0 (#145)",
          "timestamp": "2025-03-25T12:22:10-07:00",
          "tree_id": "05230b73a3ca241651392c77b448e702b9de684b",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/8ff6f801a7256089e93d1070d14fa90f0480978a"
        },
        "date": 1742932709008,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cpu-train]",
            "value": 0.06883328075983636,
            "unit": "iter/sec",
            "range": "stddev: 0.11195811569919031",
            "extra": "mean: 14.527856132400007 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cpu-val]",
            "value": 0.1899313435765186,
            "unit": "iter/sec",
            "range": "stddev: 0.12549927526884",
            "extra": "mean: 5.265060422200008 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_cm4.test.yaml-cpu]",
            "value": 0.17314716542765096,
            "unit": "iter/sec",
            "range": "stddev: 0.11929487428906987",
            "extra": "mean: 5.775433848600005 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_cm4.test.yaml-cpu]",
            "value": 0.008012504219794258,
            "unit": "iter/sec",
            "range": "stddev: 0.5388581939941012",
            "extra": "mean: 124.80492647100004 sec\nrounds: 5"
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
          "id": "37788f5c24aed9807dfdf0686896146d4265f6e4",
          "message": "Omitting filtering by guaranteeing inputs are unique. (#143)\n\nFixing #140 by making omitting slow step (`itertools.product`), instead\nguaranteeing that all inputs are unique.",
          "timestamp": "2025-03-25T13:35:00-07:00",
          "tree_id": "a326a105eab36c3d8e73baf0d530fc9affde4aa5",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/37788f5c24aed9807dfdf0686896146d4265f6e4"
        },
        "date": 1742937067189,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cpu-train]",
            "value": 0.06885437482683969,
            "unit": "iter/sec",
            "range": "stddev: 0.0847332161354219",
            "extra": "mean: 14.523405411999999 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cpu-val]",
            "value": 0.18988259495163673,
            "unit": "iter/sec",
            "range": "stddev: 0.10284596631609669",
            "extra": "mean: 5.266412123000009 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_cm4.test.yaml-cpu]",
            "value": 0.17246449418571344,
            "unit": "iter/sec",
            "range": "stddev: 0.08907192237461768",
            "extra": "mean: 5.7982949170000095 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_cm4.test.yaml-cpu]",
            "value": 0.00805028901506674,
            "unit": "iter/sec",
            "range": "stddev: 0.2577031009398641",
            "extra": "mean: 124.21914270760001 sec\nrounds: 5"
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
          "id": "327bc7c42465a7ccd6821b255cd1bd0b1bf1cd45",
          "message": "Fixes to the lazy data loader given more realistic data. (#148)\n\nExtracted some pre-conditions from the `validate_data` function into\nsmaller parts and applying some of them to all input from the current\ndata loader.\n\nIn addition, I've caught a minor issue in this loader's collate_fn as\nwell as my new data loader test.\n\nThis fixes were found during #128.\n\n---------\n\nCo-authored-by: Jesse Rusak <jesse@openathena.ai>",
          "timestamp": "2025-03-26T11:08:24-07:00",
          "tree_id": "7be8460bf295353bf8588e337751cb3c8f65bde1",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/327bc7c42465a7ccd6821b255cd1bd0b1bf1cd45"
        },
        "date": 1743014899125,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cpu-train]",
            "value": 0.06061484922652249,
            "unit": "iter/sec",
            "range": "stddev: 0.222510901749197",
            "extra": "mean: 16.497607645000006 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cpu-val]",
            "value": 0.17060904545627875,
            "unit": "iter/sec",
            "range": "stddev: 0.10910546080145367",
            "extra": "mean: 5.861353935400018 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_cm4.test.yaml-cpu]",
            "value": 0.15469816710082626,
            "unit": "iter/sec",
            "range": "stddev: 0.11614681502382321",
            "extra": "mean: 6.464200699600008 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_cm4.test.yaml-cpu]",
            "value": 0.007276253430666181,
            "unit": "iter/sec",
            "range": "stddev: 2.091037949685462",
            "extra": "mean: 137.43336588379995 sec\nrounds: 5"
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
          "id": "53dae35e99b6fd0a2f1e5dd6173b0806544dd52a",
          "message": "Fix parsing of loader_version config parameter. (#149)",
          "timestamp": "2025-03-27T18:26:22-07:00",
          "tree_id": "076bc5392b6347a05971318063e112b0a7b62512",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/53dae35e99b6fd0a2f1e5dd6173b0806544dd52a"
        },
        "date": 1743127549935,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cpu-train]",
            "value": 0.062092479881305236,
            "unit": "iter/sec",
            "range": "stddev: 0.749451776676905",
            "extra": "mean: 16.105009848400005 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cpu-val]",
            "value": 0.1772498771628218,
            "unit": "iter/sec",
            "range": "stddev: 0.33776207944805897",
            "extra": "mean: 5.6417528520000015 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_cm4.test.yaml-cpu]",
            "value": 0.15106705380623353,
            "unit": "iter/sec",
            "range": "stddev: 0.1227987465445997",
            "extra": "mean: 6.619577034200006 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_cm4.test.yaml-cpu]",
            "value": 0.007145199523872609,
            "unit": "iter/sec",
            "range": "stddev: 1.9720871657693768",
            "extra": "mean: 139.9541043828 sec\nrounds: 5"
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
          "id": "35bb5cb6d97abb0e8bd9df99bc4af91c01d511f3",
          "message": "Adding remote data to tests + caching test data (#128)\n\nThis PR addressed part of #146. It adds ~30 days worth of OM4 to the\ntest data sources. To make this more performant, we add a local data\ncaching mechanism.",
          "timestamp": "2025-03-27T19:50:56-07:00",
          "tree_id": "8ff73fc05cbb1d2c6577ae342ab1a27d86f553e2",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/35bb5cb6d97abb0e8bd9df99bc4af91c01d511f3"
        },
        "date": 1743133808661,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-cpu-train]",
            "value": 0.06805100074389493,
            "unit": "iter/sec",
            "range": "stddev: 0.19215561630474054",
            "extra": "mean: 14.694861046400012 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-cpu-val]",
            "value": 0.18759282828272913,
            "unit": "iter/sec",
            "range": "stddev: 0.11261888546933578",
            "extra": "mean: 5.330694190999975 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-cpu]",
            "value": 0.17140512915221195,
            "unit": "iter/sec",
            "range": "stddev: 0.09309060590109618",
            "extra": "mean: 5.834131131000026 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-cpu]",
            "value": 0.00789744144958315,
            "unit": "iter/sec",
            "range": "stddev: 0.14038154000028777",
            "extra": "mean: 126.62328760320001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-cpu-train]",
            "value": 0.2951618389086693,
            "unit": "iter/sec",
            "range": "stddev: 0.08509378358749765",
            "extra": "mean: 3.3879718451998997 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-cpu-val]",
            "value": 0.6455542309365162,
            "unit": "iter/sec",
            "range": "stddev: 0.10568591807117317",
            "extra": "mean: 1.5490565348000018 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-cpu]",
            "value": 0.41771067633595405,
            "unit": "iter/sec",
            "range": "stddev: 0.10534094180730513",
            "extra": "mean: 2.394001534200015 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-cpu]",
            "value": 0.011437565351366717,
            "unit": "iter/sec",
            "range": "stddev: 0.31712113113160606",
            "extra": "mean: 87.43119442640004 sec\nrounds: 5"
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
          "id": "2ad42a381a36e40aca79d4469f684f623c13e041",
          "message": "Eval checkpoint load fix (#153)\n\nNeeded to remove the \"module.\" component in the keys of the checkpoint",
          "timestamp": "2025-03-27T23:47:47-04:00",
          "tree_id": "865556090eff2ac4bc7c3aa24b17056c9b7db012",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/2ad42a381a36e40aca79d4469f684f623c13e041"
        },
        "date": 1743137132280,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-cpu-train]",
            "value": 0.06794325076375798,
            "unit": "iter/sec",
            "range": "stddev: 0.16028859771807827",
            "extra": "mean: 14.718165362399997 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-cpu-val]",
            "value": 0.18711830662166187,
            "unit": "iter/sec",
            "range": "stddev: 0.10656089705694066",
            "extra": "mean: 5.344212536199995 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-cpu]",
            "value": 0.1675220435464086,
            "unit": "iter/sec",
            "range": "stddev: 0.12278085469542117",
            "extra": "mean: 5.9693636659999925 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-cpu]",
            "value": 0.007795256660479932,
            "unit": "iter/sec",
            "range": "stddev: 1.4429334330298083",
            "extra": "mean: 128.28313980600004 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-cpu-train]",
            "value": 0.29776166262333736,
            "unit": "iter/sec",
            "range": "stddev: 0.10557283858820266",
            "extra": "mean: 3.3583907047999673 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-cpu-val]",
            "value": 0.6461851442525958,
            "unit": "iter/sec",
            "range": "stddev: 0.10749916164905914",
            "extra": "mean: 1.547544088399991 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-cpu]",
            "value": 0.41901897276526695,
            "unit": "iter/sec",
            "range": "stddev: 0.10860597012621588",
            "extra": "mean: 2.3865267803999815 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-cpu]",
            "value": 0.011334912164607002,
            "unit": "iter/sec",
            "range": "stddev: 1.00216817341911",
            "extra": "mean: 88.22300389080002 sec\nrounds: 5"
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
          "id": "0758666ad441442670f5e33a10e8ef92af72bbab",
          "message": "Compact data during clone. (#144)\n\nAdded an option to data cloning script to compact variables. Here is\nwhat a sample output looks like:\n\n```\n<xarray.Dataset> Size: 74GB\nDimensions:         (y: 180, x: 360, lev: 19, time: 3504, y_b: 181, x_b: 361)\nCoordinates:\n    areacello       (y, x) float64 518kB dask.array<chunksize=(90, 360), meta=np.ndarray>\n    dz              (lev) int64 152B dask.array<chunksize=(19,), meta=np.ndarray>\n    lat             (y, x) float64 518kB dask.array<chunksize=(90, 360), meta=np.ndarray>\n    lat_b           (y_b, x_b) float64 523kB dask.array<chunksize=(91, 361), meta=np.ndarray>\n  * lev             (lev) float64 152B 2.5 10.0 22.5 40.0 ... 4e+03 5e+03 6e+03\n    lon             (y, x) float64 518kB dask.array<chunksize=(90, 360), meta=np.ndarray>\n    lon_b           (y_b, x_b) float64 523kB dask.array<chunksize=(91, 361), meta=np.ndarray>\n    ocean_fraction  (lev, y, x) float64 10MB dask.array<chunksize=(19, 180, 360), meta=np.ndarray>\n  * time            (time) object 28kB 1975-01-03 12:00:00 ... 2022-12-29 12:...\n    wetmask         (lev, y, x) bool 1MB dask.array<chunksize=(19, 90, 360), meta=np.ndarray>\n  * x               (x) float64 3kB 0.5 1.5 2.5 3.5 ... 356.5 357.5 358.5 359.5\n  * y               (y) float64 1kB -89.24 -88.25 -87.25 ... 87.25 88.25 89.24\nDimensions without coordinates: y_b, x_b\nData variables:\n    hfds            (time, y, x) float32 908MB dask.array<chunksize=(50, 180, 360), meta=np.ndarray>\n    hfds_anomalies  (time, y, x) float32 908MB dask.array<chunksize=(50, 180, 360), meta=np.ndarray>\n    tauuo           (time, y, x) float32 908MB dask.array<chunksize=(50, 180, 360), meta=np.ndarray>\n    tauvo           (time, y, x) float32 908MB dask.array<chunksize=(50, 180, 360), meta=np.ndarray>\n    zos             (time, y, x) float32 908MB dask.array<chunksize=(50, 180, 360), meta=np.ndarray>\n    so              (lev, time, y, x) float32 17GB dask.array<chunksize=(19, 50, 180, 360), meta=np.ndarray>\n    thetao          (lev, time, y, x) float32 17GB dask.array<chunksize=(19, 50, 180, 360), meta=np.ndarray>\n    uo              (lev, time, y, x) float32 17GB dask.array<chunksize=(19, 50, 180, 360), meta=np.ndarray>\n    vo              (lev, time, y, x) float32 17GB dask.array<chunksize=(19, 50, 180, 360), meta=np.ndarray\n```\n\nThis script defaults to reading all levels as one chunk. This makes the\nrecommended chunking heuristic ~50-60 time chunks.",
          "timestamp": "2025-03-28T16:06:31-07:00",
          "tree_id": "ac7374803b7284f28d81ea158e8bc0ca2ebc066e",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/0758666ad441442670f5e33a10e8ef92af72bbab"
        },
        "date": 1743207054236,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-cpu-train]",
            "value": 0.06192602190338845,
            "unit": "iter/sec",
            "range": "stddev: 0.7589353782252718",
            "extra": "mean: 16.148300330999984 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-cpu-val]",
            "value": 0.1681211673664407,
            "unit": "iter/sec",
            "range": "stddev: 0.13651723575993333",
            "extra": "mean: 5.948090984999988 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-cpu]",
            "value": 0.1494440737945972,
            "unit": "iter/sec",
            "range": "stddev: 0.18540199846759348",
            "extra": "mean: 6.691466410200019 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-cpu]",
            "value": 0.0069497983045991776,
            "unit": "iter/sec",
            "range": "stddev: 0.2284333043143579",
            "extra": "mean: 143.8890678796 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-cpu-train]",
            "value": 0.26727826736650434,
            "unit": "iter/sec",
            "range": "stddev: 0.12045102909618449",
            "extra": "mean: 3.7414190455999687 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-cpu-val]",
            "value": 0.5736334994156672,
            "unit": "iter/sec",
            "range": "stddev: 0.11864015642253276",
            "extra": "mean: 1.7432733636000193 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-cpu]",
            "value": 0.36233614588532514,
            "unit": "iter/sec",
            "range": "stddev: 0.13414870199404172",
            "extra": "mean: 2.759868181399952 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-cpu]",
            "value": 0.01009885693706528,
            "unit": "iter/sec",
            "range": "stddev: 0.38875234504013834",
            "extra": "mean: 99.02110765920001 sec\nrounds: 5"
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
          "id": "a773fdcc8ee50509195476702df0d0cd9a7e40e7",
          "message": "Fixed anomalies statistics for anomalies use (#156)\n\nNeed to compute statistics when you compute anomalies before training",
          "timestamp": "2025-03-31T12:20:49-04:00",
          "tree_id": "9229b9d38ddf3fb9bf752c7e6e5c4abc50f59abc",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/a773fdcc8ee50509195476702df0d0cd9a7e40e7"
        },
        "date": 1743441487683,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-cpu-train]",
            "value": 0.06817990405859362,
            "unit": "iter/sec",
            "range": "stddev: 0.11065029184188023",
            "extra": "mean: 14.667078427399996 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-cpu-val]",
            "value": 0.18902782651898106,
            "unit": "iter/sec",
            "range": "stddev: 0.09642165211006563",
            "extra": "mean: 5.290226409599995 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-cpu]",
            "value": 0.16902180452987633,
            "unit": "iter/sec",
            "range": "stddev: 0.08930393089289385",
            "extra": "mean: 5.916396424599998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-cpu]",
            "value": 0.007864308933688738,
            "unit": "iter/sec",
            "range": "stddev: 1.7597971493237983",
            "extra": "mean: 127.15675444999997 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-cpu-train]",
            "value": 0.2995966069621478,
            "unit": "iter/sec",
            "range": "stddev: 0.08342444491013071",
            "extra": "mean: 3.3378215132000606 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-cpu-val]",
            "value": 0.6484949473819137,
            "unit": "iter/sec",
            "range": "stddev: 0.10532292057038158",
            "extra": "mean: 1.5420320605999678 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-cpu]",
            "value": 0.4211383869158773,
            "unit": "iter/sec",
            "range": "stddev: 0.10195608208425748",
            "extra": "mean: 2.3745163848000175 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-cpu]",
            "value": 0.011482888407761247,
            "unit": "iter/sec",
            "range": "stddev: 0.16637770562591725",
            "extra": "mean: 87.08610277220001 sec\nrounds: 5"
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
          "id": "00206f926d2761313ac4f35088f132cf16163ea6",
          "message": "Torch arrays should be floats, not doubles. (#170)",
          "timestamp": "2025-04-02T17:49:17-07:00",
          "tree_id": "c45653554e4f644b81a3baa7be8cefa01458c26d",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/00206f926d2761313ac4f35088f132cf16163ea6"
        },
        "date": 1743662125261,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist0-cpu-train]",
            "value": 0.06074502745598928,
            "unit": "iter/sec",
            "range": "stddev: 0.040284731695570834",
            "extra": "mean: 16.462252827600015 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist0-cpu-val]",
            "value": 0.1548307348660979,
            "unit": "iter/sec",
            "range": "stddev: 0.08516367832341513",
            "extra": "mean: 6.458665980400008 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-eager-hist0-cpu]",
            "value": 0.13759029886756707,
            "unit": "iter/sec",
            "range": "stddev: 0.11998447659071373",
            "extra": "mean: 7.267954268799985 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-eager-hist0-cpu]",
            "value": 0.007149750744956825,
            "unit": "iter/sec",
            "range": "stddev: 0.5894295979676637",
            "extra": "mean: 139.8650156728 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist2-cpu-train]",
            "value": 0.12516418024586495,
            "unit": "iter/sec",
            "range": "stddev: 0.09828489265893638",
            "extra": "mean: 7.9895062471999605 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist2-cpu-val]",
            "value": 0.40747706448271664,
            "unit": "iter/sec",
            "range": "stddev: 0.09464355239737145",
            "extra": "mean: 2.4541258568000104 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist2-cpu]",
            "value": 1.2115882070063386,
            "unit": "iter/sec",
            "range": "stddev: 0.096636709499376",
            "extra": "mean: 825.3629361999629 msec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-lazy-hist2-cpu]",
            "value": 0.008769811815709027,
            "unit": "iter/sec",
            "range": "stddev: 0.47963487496660423",
            "extra": "mean: 114.02753229079994 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist2-cpu-train]",
            "value": 0.20351377612963692,
            "unit": "iter/sec",
            "range": "stddev: 0.026116094674358055",
            "extra": "mean: 4.913672278200011 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist2-cpu-val]",
            "value": 0.6831207289751369,
            "unit": "iter/sec",
            "range": "stddev: 0.11515152081194631",
            "extra": "mean: 1.4638700856000468 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist2-cpu]",
            "value": 1.16173922452327,
            "unit": "iter/sec",
            "range": "stddev: 0.11487523384666752",
            "extra": "mean: 860.7783733998986 msec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-eager-hist2-cpu]",
            "value": 0.009730351633311745,
            "unit": "iter/sec",
            "range": "stddev: 0.7300303500292272",
            "extra": "mean: 102.77120886120001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist2-cpu-train]",
            "value": 0.04219578344005863,
            "unit": "iter/sec",
            "range": "stddev: 0.1297276922031493",
            "extra": "mean: 23.69905043760009 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist2-cpu-val]",
            "value": 0.28447990762781,
            "unit": "iter/sec",
            "range": "stddev: 0.1381942712807739",
            "extra": "mean: 3.515186743199865 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-lazy-hist2-cpu]",
            "value": 0.44047485211014503,
            "unit": "iter/sec",
            "range": "stddev: 0.033627427601037554",
            "extra": "mean: 2.2702771683999345 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-lazy-hist2-cpu]",
            "value": 0.006552310691400773,
            "unit": "iter/sec",
            "range": "stddev: 1.3113275469489583",
            "extra": "mean: 152.61791558699989 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist2-cpu-train]",
            "value": 0.06007179304022524,
            "unit": "iter/sec",
            "range": "stddev: 0.18063241933363083",
            "extra": "mean: 16.646747989199866 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist2-cpu-val]",
            "value": 0.4238636555079993,
            "unit": "iter/sec",
            "range": "stddev: 0.02556612596953584",
            "extra": "mean: 2.359249223200095 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-eager-hist2-cpu]",
            "value": 0.4586371429966374,
            "unit": "iter/sec",
            "range": "stddev: 0.10719536917695707",
            "extra": "mean: 2.1803729054001453 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-eager-hist2-cpu]",
            "value": 0.007383635356746451,
            "unit": "iter/sec",
            "range": "stddev: 1.0617533698707229",
            "extra": "mean: 135.4346404832 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist1-cpu-train]",
            "value": 0.1386767456949162,
            "unit": "iter/sec",
            "range": "stddev: 0.13448853360345994",
            "extra": "mean: 7.211014326799704 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist1-cpu-val]",
            "value": 0.32942922582836165,
            "unit": "iter/sec",
            "range": "stddev: 0.10100437699880313",
            "extra": "mean: 3.0355533802001444 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist1-cpu]",
            "value": 0.6875137100202103,
            "unit": "iter/sec",
            "range": "stddev: 0.11657585621510595",
            "extra": "mean: 1.4545164488001319 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-lazy-hist1-cpu]",
            "value": 0.009370652178295096,
            "unit": "iter/sec",
            "range": "stddev: 1.5968357471852612",
            "extra": "mean: 106.71615816839986 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist0-cpu-train]",
            "value": 0.17568541254441564,
            "unit": "iter/sec",
            "range": "stddev: 0.123804587241767",
            "extra": "mean: 5.691992212200239 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist0-cpu-val]",
            "value": 0.35387955147489497,
            "unit": "iter/sec",
            "range": "stddev: 0.11863801032257239",
            "extra": "mean: 2.8258202426000936 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist0-cpu]",
            "value": 0.3200552661396521,
            "unit": "iter/sec",
            "range": "stddev: 0.11120716972589453",
            "extra": "mean: 3.1244603848001136 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-lazy-hist0-cpu]",
            "value": 0.009693067808275618,
            "unit": "iter/sec",
            "range": "stddev: 1.189445719938358",
            "extra": "mean: 103.16651237560036 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist1-cpu-train]",
            "value": 0.21711472611781626,
            "unit": "iter/sec",
            "range": "stddev: 0.11228083178194881",
            "extra": "mean: 4.6058598505997 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist1-cpu-val]",
            "value": 0.524315594799618,
            "unit": "iter/sec",
            "range": "stddev: 0.10443702496210487",
            "extra": "mean: 1.907248248799806 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist1-cpu]",
            "value": 0.689605753363213,
            "unit": "iter/sec",
            "range": "stddev: 0.12079770925496719",
            "extra": "mean: 1.4501039109998601 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-eager-hist1-cpu]",
            "value": 0.010271168235615616,
            "unit": "iter/sec",
            "range": "stddev: 0.5397130482521897",
            "extra": "mean: 97.35990853819985 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist0-cpu-train]",
            "value": 0.2752022406267279,
            "unit": "iter/sec",
            "range": "stddev: 0.09531745992572642",
            "extra": "mean: 3.6336913453998934 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist0-cpu-val]",
            "value": 0.5540221787683595,
            "unit": "iter/sec",
            "range": "stddev: 0.10830689977529308",
            "extra": "mean: 1.8049818911998954 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist0-cpu]",
            "value": 0.3332773543802844,
            "unit": "iter/sec",
            "range": "stddev: 0.10946225055266062",
            "extra": "mean: 3.000503895200018 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-eager-hist0-cpu]",
            "value": 0.01050622429666201,
            "unit": "iter/sec",
            "range": "stddev: 0.559105235681004",
            "extra": "mean: 95.18167247940019 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist1-cpu-train]",
            "value": 0.0418204637417189,
            "unit": "iter/sec",
            "range": "stddev: 0.23888148958844077",
            "extra": "mean: 23.911738668799806 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist1-cpu-val]",
            "value": 0.14897358908825686,
            "unit": "iter/sec",
            "range": "stddev: 0.07465183002259805",
            "extra": "mean: 6.712599233999572 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-lazy-hist1-cpu]",
            "value": 0.3039523521433321,
            "unit": "iter/sec",
            "range": "stddev: 0.08498809215653877",
            "extra": "mean: 3.289989345199865 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-lazy-hist1-cpu]",
            "value": 0.006534759043404375,
            "unit": "iter/sec",
            "range": "stddev: 1.9592885294126579",
            "extra": "mean: 153.0278306143995 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist0-cpu-train]",
            "value": 0.04783979166011926,
            "unit": "iter/sec",
            "range": "stddev: 0.10024130974495925",
            "extra": "mean: 20.903101064999646 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist0-cpu-val]",
            "value": 0.12369954096146796,
            "unit": "iter/sec",
            "range": "stddev: 0.11299187311765839",
            "extra": "mean: 8.084104372800358 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-lazy-hist0-cpu]",
            "value": 0.1354609066878415,
            "unit": "iter/sec",
            "range": "stddev: 0.07896800581495084",
            "extra": "mean: 7.3822036515997755 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-lazy-hist0-cpu]",
            "value": 0.0064659430514909955,
            "unit": "iter/sec",
            "range": "stddev: 0.3339016782605396",
            "extra": "mean: 154.6564812026001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist1-cpu-train]",
            "value": 0.05972636798051713,
            "unit": "iter/sec",
            "range": "stddev: 0.3409247319394133",
            "extra": "mean: 16.743023790199366 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist1-cpu-val]",
            "value": 0.21890209737466595,
            "unit": "iter/sec",
            "range": "stddev: 0.14203307110148922",
            "extra": "mean: 4.568252255200787 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-eager-hist1-cpu]",
            "value": 0.3011549194852535,
            "unit": "iter/sec",
            "range": "stddev: 0.14412740449121958",
            "extra": "mean: 3.3205501066004217 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-eager-hist1-cpu]",
            "value": 0.0074264007237753804,
            "unit": "iter/sec",
            "range": "stddev: 0.4301585739606997",
            "extra": "mean: 134.65473210980016 sec\nrounds: 5"
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
          "id": "648cd211c95f934b7a03d4ebc68440bfb7ed3499",
          "message": "Turn off dask by default (#168)\n\nAccording to [results\nhere](https://github.com/suryadheeshjith/Ocean_Emulator/issues/105#issuecomment-2773940880),\nturning off dask is 20x faster for the eager data loader and 10x faster\nfor the lazy data loader. This disables dask by default but you can\ntoggle it back on.",
          "timestamp": "2025-04-03T11:56:34-04:00",
          "tree_id": "738bcd88d7610dcd35c276699c47fe9c082c881e",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/648cd211c95f934b7a03d4ebc68440bfb7ed3499"
        },
        "date": 1743709289200,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist0-cpu-train]",
            "value": 2.4178005945941874,
            "unit": "iter/sec",
            "range": "stddev: 0.006192203629206583",
            "extra": "mean: 413.59903800000666 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist0-cpu-val]",
            "value": 5.626683783364358,
            "unit": "iter/sec",
            "range": "stddev: 0.0036692974119999174",
            "extra": "mean: 177.72457783331674 msec\nrounds: 6"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-eager-hist0-cpu]",
            "value": 0.4227288462166558,
            "unit": "iter/sec",
            "range": "stddev: 0.028187746749596905",
            "extra": "mean: 2.365582592599992 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-eager-hist0-cpu]",
            "value": 0.013275453386354594,
            "unit": "iter/sec",
            "range": "stddev: 1.3400605572407605",
            "extra": "mean: 75.32699418220002 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist2-cpu-train]",
            "value": 1.7923596640097128,
            "unit": "iter/sec",
            "range": "stddev: 0.0027997079198316946",
            "extra": "mean: 557.9237359999979 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist2-cpu-val]",
            "value": 5.959860281406464,
            "unit": "iter/sec",
            "range": "stddev: 0.001677125288731333",
            "extra": "mean: 167.78916833332383 msec\nrounds: 6"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist2-cpu]",
            "value": 1.0370392359903824,
            "unit": "iter/sec",
            "range": "stddev: 0.0048455601493076665",
            "extra": "mean: 964.2836696000131 msec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-lazy-hist2-cpu]",
            "value": 0.0118221476683305,
            "unit": "iter/sec",
            "range": "stddev: 0.36686111674819993",
            "extra": "mean: 84.58699959219999 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist2-cpu-train]",
            "value": 2.2181123705353003,
            "unit": "iter/sec",
            "range": "stddev: 0.0029008290497086328",
            "extra": "mean: 450.8337870000105 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist2-cpu-val]",
            "value": 7.443675586365369,
            "unit": "iter/sec",
            "range": "stddev: 0.0012141495924978421",
            "extra": "mean: 134.34223300001236 msec\nrounds: 8"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist2-cpu]",
            "value": 1.0227926565609948,
            "unit": "iter/sec",
            "range": "stddev: 0.002123104171829306",
            "extra": "mean: 977.7152715999819 msec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-eager-hist2-cpu]",
            "value": 0.011684671626446217,
            "unit": "iter/sec",
            "range": "stddev: 0.8340541333330646",
            "extra": "mean: 85.58220821000005 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist2-cpu-train]",
            "value": 0.9110546409886817,
            "unit": "iter/sec",
            "range": "stddev: 0.02988270354832167",
            "extra": "mean: 1.0976290059998974 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist2-cpu-val]",
            "value": 4.43279106229086,
            "unit": "iter/sec",
            "range": "stddev: 0.025586813873866694",
            "extra": "mean: 225.5915033999372 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-lazy-hist2-cpu]",
            "value": 0.7856626746577899,
            "unit": "iter/sec",
            "range": "stddev: 0.003684026171548324",
            "extra": "mean: 1.2728108795999105 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-lazy-hist2-cpu]",
            "value": 0.011361152358934918,
            "unit": "iter/sec",
            "range": "stddev: 1.3389515683144062",
            "extra": "mean: 88.01924033819996 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist2-cpu-train]",
            "value": 1.1834050892270966,
            "unit": "iter/sec",
            "range": "stddev: 0.029720403064178724",
            "extra": "mean: 845.019181600037 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist2-cpu-val]",
            "value": 5.563875599969852,
            "unit": "iter/sec",
            "range": "stddev: 0.009298825811315104",
            "extra": "mean: 179.73083366662954 msec\nrounds: 6"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-eager-hist2-cpu]",
            "value": 0.7775215835521149,
            "unit": "iter/sec",
            "range": "stddev: 0.00853676795963598",
            "extra": "mean: 1.2861379299999498 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-eager-hist2-cpu]",
            "value": 0.011515436151766324,
            "unit": "iter/sec",
            "range": "stddev: 0.5387665981041194",
            "extra": "mean: 86.83995871460002 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist1-cpu-train]",
            "value": 1.9430289721265945,
            "unit": "iter/sec",
            "range": "stddev: 0.003616336608205787",
            "extra": "mean: 514.6603649998724 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist1-cpu-val]",
            "value": 4.619076600675307,
            "unit": "iter/sec",
            "range": "stddev: 0.0011035143908862935",
            "extra": "mean: 216.49348699993425 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist1-cpu]",
            "value": 0.7485027275539519,
            "unit": "iter/sec",
            "range": "stddev: 0.003787863473581628",
            "extra": "mean: 1.3360004756000308 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-lazy-hist1-cpu]",
            "value": 0.01293975014779372,
            "unit": "iter/sec",
            "range": "stddev: 0.7041681046349249",
            "extra": "mean: 77.281244891 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist0-cpu-train]",
            "value": 2.699697777095013,
            "unit": "iter/sec",
            "range": "stddev: 0.006180421425795055",
            "extra": "mean: 370.41183219998857 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist0-cpu-val]",
            "value": 5.368764583484435,
            "unit": "iter/sec",
            "range": "stddev: 0.002706993984942888",
            "extra": "mean: 186.2625906668048 msec\nrounds: 6"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist0-cpu]",
            "value": 0.5437527117763324,
            "unit": "iter/sec",
            "range": "stddev: 0.008179079696007785",
            "extra": "mean: 1.8390712880000137 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-lazy-hist0-cpu]",
            "value": 0.013456029148314346,
            "unit": "iter/sec",
            "range": "stddev: 0.10804432552457968",
            "extra": "mean: 74.31612914760008 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist1-cpu-train]",
            "value": 2.6169734477652633,
            "unit": "iter/sec",
            "range": "stddev: 0.005091221331392522",
            "extra": "mean: 382.120804799888 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist1-cpu-val]",
            "value": 6.291359651120104,
            "unit": "iter/sec",
            "range": "stddev: 0.0012126723280475862",
            "extra": "mean: 158.94815357153544 msec\nrounds: 7"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist1-cpu]",
            "value": 0.7492789101238887,
            "unit": "iter/sec",
            "range": "stddev: 0.00590104196302721",
            "extra": "mean: 1.334616504599944 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-eager-hist1-cpu]",
            "value": 0.013025611010289219,
            "unit": "iter/sec",
            "range": "stddev: 0.231953228682371",
            "extra": "mean: 76.7718304508002 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist0-cpu-train]",
            "value": 4.10520738117755,
            "unit": "iter/sec",
            "range": "stddev: 0.004259187589725024",
            "extra": "mean: 243.59305319994746 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist0-cpu-val]",
            "value": 8.255513560307017,
            "unit": "iter/sec",
            "range": "stddev: 0.0018785266777371658",
            "extra": "mean: 121.1311680000209 msec\nrounds: 9"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist0-cpu]",
            "value": 0.5412261763927526,
            "unit": "iter/sec",
            "range": "stddev: 0.014655701490718384",
            "extra": "mean: 1.8476563840000382 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-eager-hist0-cpu]",
            "value": 0.01355103421005687,
            "unit": "iter/sec",
            "range": "stddev: 0.2864486390277148",
            "extra": "mean: 73.79510556160002 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist1-cpu-train]",
            "value": 1.0779646305625614,
            "unit": "iter/sec",
            "range": "stddev: 0.006180413399337768",
            "extra": "mean: 927.6742219994048 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist1-cpu-val]",
            "value": 3.2338291394035257,
            "unit": "iter/sec",
            "range": "stddev: 0.007133298558926722",
            "extra": "mean: 309.23093239998707 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-lazy-hist1-cpu]",
            "value": 0.6037084029878418,
            "unit": "iter/sec",
            "range": "stddev: 0.014486005915359488",
            "extra": "mean: 1.656428823999886 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-lazy-hist1-cpu]",
            "value": 0.01272096598874521,
            "unit": "iter/sec",
            "range": "stddev: 0.36288089238452975",
            "extra": "mean: 78.61038233140025 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist0-cpu-train]",
            "value": 1.9248285311053146,
            "unit": "iter/sec",
            "range": "stddev: 0.00376883186412506",
            "extra": "mean: 519.5267962002617 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist0-cpu-val]",
            "value": 4.560332135699614,
            "unit": "iter/sec",
            "range": "stddev: 0.001928556736517997",
            "extra": "mean: 219.28227380012686 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-lazy-hist0-cpu]",
            "value": 0.4263716100175196,
            "unit": "iter/sec",
            "range": "stddev: 0.006986970917744214",
            "extra": "mean: 2.345371916199838 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-lazy-hist0-cpu]",
            "value": 0.013396358837702581,
            "unit": "iter/sec",
            "range": "stddev: 0.21159940449804526",
            "extra": "mean: 74.64714943180006 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist1-cpu-train]",
            "value": 1.426262632520138,
            "unit": "iter/sec",
            "range": "stddev: 0.007587005138145919",
            "extra": "mean: 701.133141399805 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist1-cpu-val]",
            "value": 4.3180070827596,
            "unit": "iter/sec",
            "range": "stddev: 0.0012180736577468963",
            "extra": "mean: 231.58831860018836 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-eager-hist1-cpu]",
            "value": 0.6037765370585522,
            "unit": "iter/sec",
            "range": "stddev: 0.0029525037297728003",
            "extra": "mean: 1.6562419017998764 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-eager-hist1-cpu]",
            "value": 0.012540359580706059,
            "unit": "iter/sec",
            "range": "stddev: 0.7011178396201121",
            "extra": "mean: 79.74252999400014 sec\nrounds: 5"
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
          "id": "a2e57c158589e0447799a389b24b96a2d207fa61",
          "message": "Datasets test for TrainData + Test fix (#172)\n\n- Just a test for traindata ensuring underlying data is not manipulated\nby TrainData. Added a test just in case (even with #169 )\n- Removed lat/lon coords in tests data statistics",
          "timestamp": "2025-04-03T15:49:40-04:00",
          "tree_id": "df76883e78482679bd1224e1c34d6eae0c7291e5",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/a2e57c158589e0447799a389b24b96a2d207fa61"
        },
        "date": 1743723096481,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist0-cpu-train]",
            "value": 2.3046649487454043,
            "unit": "iter/sec",
            "range": "stddev: 0.008319316368729634",
            "extra": "mean: 433.9025508000077 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist0-cpu-val]",
            "value": 5.159718315085358,
            "unit": "iter/sec",
            "range": "stddev: 0.004061035584811636",
            "extra": "mean: 193.8090296666625 msec\nrounds: 6"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-eager-hist0-cpu]",
            "value": 0.4199549174146207,
            "unit": "iter/sec",
            "range": "stddev: 0.015241762992303603",
            "extra": "mean: 2.3812079786000027 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-eager-hist0-cpu]",
            "value": 0.013780429625430437,
            "unit": "iter/sec",
            "range": "stddev: 0.9719831568910111",
            "extra": "mean: 72.5666780486 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist2-cpu-train]",
            "value": 1.816238086126111,
            "unit": "iter/sec",
            "range": "stddev: 0.003011183620581127",
            "extra": "mean: 550.5886081999961 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist2-cpu-val]",
            "value": 6.033005114442061,
            "unit": "iter/sec",
            "range": "stddev: 0.001970787726316305",
            "extra": "mean: 165.7548735714077 msec\nrounds: 7"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist2-cpu]",
            "value": 1.0346638584046415,
            "unit": "iter/sec",
            "range": "stddev: 0.006259813809238659",
            "extra": "mean: 966.497468600005 msec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-lazy-hist2-cpu]",
            "value": 0.011934752481395264,
            "unit": "iter/sec",
            "range": "stddev: 0.35322470675388595",
            "extra": "mean: 83.78891825020006 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist2-cpu-train]",
            "value": 2.278989819694713,
            "unit": "iter/sec",
            "range": "stddev: 0.0020666559687070895",
            "extra": "mean: 438.7909025999761 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist2-cpu-val]",
            "value": 7.568635811984987,
            "unit": "iter/sec",
            "range": "stddev: 0.0009767489050589273",
            "extra": "mean: 132.12420637501054 msec\nrounds: 8"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist2-cpu]",
            "value": 1.0404597459335234,
            "unit": "iter/sec",
            "range": "stddev: 0.0012444452357588628",
            "extra": "mean: 961.113588400076 msec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-eager-hist2-cpu]",
            "value": 0.012035969151689797,
            "unit": "iter/sec",
            "range": "stddev: 0.48442475962795206",
            "extra": "mean: 83.08429403540008 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist2-cpu-train]",
            "value": 1.0076111865743025,
            "unit": "iter/sec",
            "range": "stddev: 0.08972411233033846",
            "extra": "mean: 992.4463060000562 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist2-cpu-val]",
            "value": 5.294150221656049,
            "unit": "iter/sec",
            "range": "stddev: 0.0010852472834202238",
            "extra": "mean: 188.88772666658346 msec\nrounds: 6"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-lazy-hist2-cpu]",
            "value": 0.7909479580026444,
            "unit": "iter/sec",
            "range": "stddev: 0.0024690967830099503",
            "extra": "mean: 1.2643056851999064 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-lazy-hist2-cpu]",
            "value": 0.011818212795849133,
            "unit": "iter/sec",
            "range": "stddev: 2.048503422962624",
            "extra": "mean: 84.6151628232 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist2-cpu-train]",
            "value": 1.2010578579393771,
            "unit": "iter/sec",
            "range": "stddev: 0.012429173183269323",
            "extra": "mean: 832.599356800074 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist2-cpu-val]",
            "value": 6.10221478258071,
            "unit": "iter/sec",
            "range": "stddev: 0.006467198007637629",
            "extra": "mean: 163.87492666672188 msec\nrounds: 6"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-eager-hist2-cpu]",
            "value": 0.7865706618705802,
            "unit": "iter/sec",
            "range": "stddev: 0.002385470197239957",
            "extra": "mean: 1.2713415951999196 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-eager-hist2-cpu]",
            "value": 0.011676579917162193,
            "unit": "iter/sec",
            "range": "stddev: 1.393219263315863",
            "extra": "mean: 85.64151550320003 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist1-cpu-train]",
            "value": 1.9671838017805516,
            "unit": "iter/sec",
            "range": "stddev: 0.0035737511115565756",
            "extra": "mean: 508.340907999991 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist1-cpu-val]",
            "value": 4.590548770516635,
            "unit": "iter/sec",
            "range": "stddev: 0.00315457338821973",
            "extra": "mean: 217.83887939991473 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist1-cpu]",
            "value": 0.7497772599954103,
            "unit": "iter/sec",
            "range": "stddev: 0.002333916938316887",
            "extra": "mean: 1.3337294332000966 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-lazy-hist1-cpu]",
            "value": 0.012930228790527316,
            "unit": "iter/sec",
            "range": "stddev: 0.37414876772283717",
            "extra": "mean: 77.33815203119994 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist0-cpu-train]",
            "value": 2.7089800479139563,
            "unit": "iter/sec",
            "range": "stddev: 0.002332214537380244",
            "extra": "mean: 369.1426228000637 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist0-cpu-val]",
            "value": 5.43799945135778,
            "unit": "iter/sec",
            "range": "stddev: 0.0018404544299682184",
            "extra": "mean: 183.89115500008302 msec\nrounds: 6"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist0-cpu]",
            "value": 0.5438318365074576,
            "unit": "iter/sec",
            "range": "stddev: 0.006458956604591762",
            "extra": "mean: 1.8388037126000198 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-lazy-hist0-cpu]",
            "value": 0.013507006544359362,
            "unit": "iter/sec",
            "range": "stddev: 0.47230464594834287",
            "extra": "mean: 74.03564932880026 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist1-cpu-train]",
            "value": 2.6109110919674663,
            "unit": "iter/sec",
            "range": "stddev: 0.00225578182290232",
            "extra": "mean: 383.0080630001248 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist1-cpu-val]",
            "value": 6.35665940985046,
            "unit": "iter/sec",
            "range": "stddev: 0.0017725668806457102",
            "extra": "mean: 157.3153342855481 msec\nrounds: 7"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist1-cpu]",
            "value": 0.7495609858783625,
            "unit": "iter/sec",
            "range": "stddev: 0.001820322700812295",
            "extra": "mean: 1.334114260000024 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-eager-hist1-cpu]",
            "value": 0.013034197097660686,
            "unit": "iter/sec",
            "range": "stddev: 0.4767015081967856",
            "extra": "mean: 76.72125812640006 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist0-cpu-train]",
            "value": 3.9848401728646734,
            "unit": "iter/sec",
            "range": "stddev: 0.006143164049840425",
            "extra": "mean: 250.95109380035862 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist0-cpu-val]",
            "value": 8.042685540118809,
            "unit": "iter/sec",
            "range": "stddev: 0.0016031198687020578",
            "extra": "mean: 124.33657825010869 msec\nrounds: 8"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist0-cpu]",
            "value": 0.5434210835798051,
            "unit": "iter/sec",
            "range": "stddev: 0.0030086294072002692",
            "extra": "mean: 1.8401935998001135 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-eager-hist0-cpu]",
            "value": 0.013649835757811795,
            "unit": "iter/sec",
            "range": "stddev: 0.23384093796145294",
            "extra": "mean: 73.26095476480005 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist1-cpu-train]",
            "value": 1.0715966458513633,
            "unit": "iter/sec",
            "range": "stddev: 0.002588183340748011",
            "extra": "mean: 933.1869448000361 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist1-cpu-val]",
            "value": 3.211186668677036,
            "unit": "iter/sec",
            "range": "stddev: 0.001279191854367144",
            "extra": "mean: 311.41135760008183 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-lazy-hist1-cpu]",
            "value": 0.6110474331927593,
            "unit": "iter/sec",
            "range": "stddev: 0.004150308452839695",
            "extra": "mean: 1.6365341635999358 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-lazy-hist1-cpu]",
            "value": 0.01286855910700248,
            "unit": "iter/sec",
            "range": "stddev: 0.5537441318679758",
            "extra": "mean: 77.70877778040013 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist0-cpu-train]",
            "value": 1.9562578307633718,
            "unit": "iter/sec",
            "range": "stddev: 0.002514582703351867",
            "extra": "mean: 511.1800623999443 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist0-cpu-val]",
            "value": 4.544014349024898,
            "unit": "iter/sec",
            "range": "stddev: 0.0020510345897050775",
            "extra": "mean: 220.06972759991186 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-lazy-hist0-cpu]",
            "value": 0.42700484105530356,
            "unit": "iter/sec",
            "range": "stddev: 0.006159300583448119",
            "extra": "mean: 2.341893823799728 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-lazy-hist0-cpu]",
            "value": 0.013529717037333404,
            "unit": "iter/sec",
            "range": "stddev: 0.1314941739605824",
            "extra": "mean: 73.91137576939983 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist1-cpu-train]",
            "value": 1.4273471839199796,
            "unit": "iter/sec",
            "range": "stddev: 0.005795321796077921",
            "extra": "mean: 700.6003943999531 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist1-cpu-val]",
            "value": 4.266042811485909,
            "unit": "iter/sec",
            "range": "stddev: 0.001678600622943959",
            "extra": "mean: 234.40927440005908 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-eager-hist1-cpu]",
            "value": 0.6118428207794882,
            "unit": "iter/sec",
            "range": "stddev: 0.0021569591269245667",
            "extra": "mean: 1.634406690800097 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-eager-hist1-cpu]",
            "value": 0.012948517162228184,
            "unit": "iter/sec",
            "range": "stddev: 0.505580160478684",
            "extra": "mean: 77.22892030579969 sec\nrounds: 5"
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
          "id": "374ff6d371214fc1b499da69c87fa108e6bbc41b",
          "message": "Re-enable dask by default (#173)\n\nSetting use_dask = false was effectively just loading the whole dataset\ninto memory at the start (during normalization) which is why it was so\nfast. If we disable normalization (for diagnosis) or delay normalization\nuntil data loading time (ie in `__getitem__`) then `use_dask: false` is\n2x slower than `use_dask: true`.",
          "timestamp": "2025-04-03T17:11:07-04:00",
          "tree_id": "c3e47985598f8c1c2c6d7754c191b3ff9ee94f25",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/374ff6d371214fc1b499da69c87fa108e6bbc41b"
        },
        "date": 1743734755736,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist0-cpu-train]",
            "value": 0.060805043553744095,
            "unit": "iter/sec",
            "range": "stddev: 0.14241772924688462",
            "extra": "mean: 16.4460041726 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist0-cpu-val]",
            "value": 0.1559705406513736,
            "unit": "iter/sec",
            "range": "stddev: 0.0849661005329425",
            "extra": "mean: 6.411467164400017 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-eager-hist0-cpu]",
            "value": 0.13834451751575091,
            "unit": "iter/sec",
            "range": "stddev: 0.10186953073540833",
            "extra": "mean: 7.228331255599971 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-eager-hist0-cpu]",
            "value": 0.007404165564847118,
            "unit": "iter/sec",
            "range": "stddev: 0.3566473590245129",
            "extra": "mean: 135.05910844940001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist2-cpu-train]",
            "value": 0.12618320792173177,
            "unit": "iter/sec",
            "range": "stddev: 0.08275364332812284",
            "extra": "mean: 7.924984761999985 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist2-cpu-val]",
            "value": 0.4234902010919711,
            "unit": "iter/sec",
            "range": "stddev: 0.0875197748693395",
            "extra": "mean: 2.3613297247999983 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist2-cpu]",
            "value": 1.1979737669132817,
            "unit": "iter/sec",
            "range": "stddev: 0.08880062832505",
            "extra": "mean: 834.7428195999782 msec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-lazy-hist2-cpu]",
            "value": 0.009062612210353019,
            "unit": "iter/sec",
            "range": "stddev: 1.029095312358875",
            "extra": "mean: 110.34346133200006 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist2-cpu-train]",
            "value": 0.20641899412098988,
            "unit": "iter/sec",
            "range": "stddev: 0.02532561789077251",
            "extra": "mean: 4.84451541999988 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist2-cpu-val]",
            "value": 0.6873910924619028,
            "unit": "iter/sec",
            "range": "stddev: 0.10682910380878226",
            "extra": "mean: 1.454775907000021 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist2-cpu]",
            "value": 1.1610812310276912,
            "unit": "iter/sec",
            "range": "stddev: 0.11345398247624286",
            "extra": "mean: 861.2661829998615 msec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-eager-hist2-cpu]",
            "value": 0.010055141019921535,
            "unit": "iter/sec",
            "range": "stddev: 0.37274310799252874",
            "extra": "mean: 99.45161365900003 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist2-cpu-train]",
            "value": 0.043457522192127895,
            "unit": "iter/sec",
            "range": "stddev: 0.0968887085799093",
            "extra": "mean: 23.01097599579998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist2-cpu-val]",
            "value": 0.29104446964088965,
            "unit": "iter/sec",
            "range": "stddev: 0.10494717826153646",
            "extra": "mean: 3.435901053999987 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-lazy-hist2-cpu]",
            "value": 0.44702686348157633,
            "unit": "iter/sec",
            "range": "stddev: 0.01284698631758786",
            "extra": "mean: 2.2370020276001013 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-lazy-hist2-cpu]",
            "value": 0.006823459796947797,
            "unit": "iter/sec",
            "range": "stddev: 0.2746699001372919",
            "extra": "mean: 146.55321929899992 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist2-cpu-train]",
            "value": 0.06115580375598475,
            "unit": "iter/sec",
            "range": "stddev: 0.13422578597441834",
            "extra": "mean: 16.351677822599775 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist2-cpu-val]",
            "value": 0.43818675697501286,
            "unit": "iter/sec",
            "range": "stddev: 0.013371375043428956",
            "extra": "mean: 2.282131954200122 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-eager-hist2-cpu]",
            "value": 0.4560804124117092,
            "unit": "iter/sec",
            "range": "stddev: 0.09476440597576855",
            "extra": "mean: 2.1925958072000866 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-eager-hist2-cpu]",
            "value": 0.0076340372030320024,
            "unit": "iter/sec",
            "range": "stddev: 0.48211486437325896",
            "extra": "mean: 130.99228801280023 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist1-cpu-train]",
            "value": 0.13960549899260943,
            "unit": "iter/sec",
            "range": "stddev: 0.14745188484387795",
            "extra": "mean: 7.163041622400124 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist1-cpu-val]",
            "value": 0.3345900874283854,
            "unit": "iter/sec",
            "range": "stddev: 0.12245460585009425",
            "extra": "mean: 2.9887316976000875 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist1-cpu]",
            "value": 0.6775722636204935,
            "unit": "iter/sec",
            "range": "stddev: 0.12288918417834098",
            "extra": "mean: 1.4758573420001995 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-lazy-hist1-cpu]",
            "value": 0.00973275507597791,
            "unit": "iter/sec",
            "range": "stddev: 0.572091077342985",
            "extra": "mean: 102.74583015740009 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist0-cpu-train]",
            "value": 0.18117412067923416,
            "unit": "iter/sec",
            "range": "stddev: 0.12057805947670949",
            "extra": "mean: 5.519552109600045 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist0-cpu-val]",
            "value": 0.3601335078642674,
            "unit": "iter/sec",
            "range": "stddev: 0.09032501409046836",
            "extra": "mean: 2.7767480064001573 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist0-cpu]",
            "value": 0.33191318537086567,
            "unit": "iter/sec",
            "range": "stddev: 0.10397912988516865",
            "extra": "mean: 3.0128360188000443 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-lazy-hist0-cpu]",
            "value": 0.010123742892989875,
            "unit": "iter/sec",
            "range": "stddev: 0.20032201077250078",
            "extra": "mean: 98.77769621080006 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist1-cpu-train]",
            "value": 0.225048345983916,
            "unit": "iter/sec",
            "range": "stddev: 0.08697403001564942",
            "extra": "mean: 4.443489667200083 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist1-cpu-val]",
            "value": 0.536423063404662,
            "unit": "iter/sec",
            "range": "stddev: 0.09669735701477969",
            "extra": "mean: 1.864200233399788 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist1-cpu]",
            "value": 0.6972614773642859,
            "unit": "iter/sec",
            "range": "stddev: 0.1094576415632517",
            "extra": "mean: 1.4341822006001166 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-eager-hist1-cpu]",
            "value": 0.010748415205594103,
            "unit": "iter/sec",
            "range": "stddev: 0.3672526614175037",
            "extra": "mean: 93.03697157879996 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist0-cpu-train]",
            "value": 0.2767627537150511,
            "unit": "iter/sec",
            "range": "stddev: 0.08797421495877683",
            "extra": "mean: 3.613202956600071 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist0-cpu-val]",
            "value": 0.5579564457372509,
            "unit": "iter/sec",
            "range": "stddev: 0.10668545610226234",
            "extra": "mean: 1.792254588400101 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist0-cpu]",
            "value": 0.3332276577348689,
            "unit": "iter/sec",
            "range": "stddev: 0.10550011158686455",
            "extra": "mean: 3.0009513819997666 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-eager-hist0-cpu]",
            "value": 0.010921312148360467,
            "unit": "iter/sec",
            "range": "stddev: 0.19493118023148043",
            "extra": "mean: 91.56408922439986 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist1-cpu-train]",
            "value": 0.04210773893054011,
            "unit": "iter/sec",
            "range": "stddev: 0.11612145813541046",
            "extra": "mean: 23.748603591600478 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist1-cpu-val]",
            "value": 0.1528058421424727,
            "unit": "iter/sec",
            "range": "stddev: 0.012616153671993353",
            "extra": "mean: 6.544252405399675 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-lazy-hist1-cpu]",
            "value": 0.30542822500424777,
            "unit": "iter/sec",
            "range": "stddev: 0.09954056698405148",
            "extra": "mean: 3.2740916461996674 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-lazy-hist1-cpu]",
            "value": 0.006717654975274208,
            "unit": "iter/sec",
            "range": "stddev: 0.35399365600350596",
            "extra": "mean: 148.86147080800038 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist0-cpu-train]",
            "value": 0.04874349006499937,
            "unit": "iter/sec",
            "range": "stddev: 0.1092715966084478",
            "extra": "mean: 20.515560101800293 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist0-cpu-val]",
            "value": 0.12460685332826747,
            "unit": "iter/sec",
            "range": "stddev: 0.12888988327254786",
            "extra": "mean: 8.025240773599943 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-lazy-hist0-cpu]",
            "value": 0.13694599848892858,
            "unit": "iter/sec",
            "range": "stddev: 0.10408206283364756",
            "extra": "mean: 7.3021483725999135 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-lazy-hist0-cpu]",
            "value": 0.006646091171619335,
            "unit": "iter/sec",
            "range": "stddev: 0.9285780747833672",
            "extra": "mean: 150.46438187159984 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist1-cpu-train]",
            "value": 0.06007962515489605,
            "unit": "iter/sec",
            "range": "stddev: 0.13569177314040307",
            "extra": "mean: 16.644577881799705 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist1-cpu-val]",
            "value": 0.22470386141482898,
            "unit": "iter/sec",
            "range": "stddev: 0.09735803296099181",
            "extra": "mean: 4.450301804800256 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-eager-hist1-cpu]",
            "value": 0.3071739609335182,
            "unit": "iter/sec",
            "range": "stddev: 0.09053829651045078",
            "extra": "mean: 3.255484276599964 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-eager-hist1-cpu]",
            "value": 0.007648289293352617,
            "unit": "iter/sec",
            "range": "stddev: 1.1806109022030544",
            "extra": "mean: 130.74819239239986 sec\nrounds: 5"
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
          "id": "e66d0ec12f83862c56dbec7a14241671fb8a3846",
          "message": "Update base image to speed up CI, give instructions on how to do this (#184)\n\nWe could consider just making this always be the latest image but that\nwould also be a bit brittle.\n\nYou can see the first \"install nvidia drivers\" step takes about 2\nminutes extra here\nhttps://github.com/suryadheeshjith/Ocean_Emulator/actions/runs/14336583301/job/40185301143\nbecause it's waiting for automated upgrades to complete; this should\nreduce that.",
          "timestamp": "2025-04-08T12:08:32-04:00",
          "tree_id": "e526de0cff987379e6476caf723380fe57e4d866",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/e66d0ec12f83862c56dbec7a14241671fb8a3846"
        },
        "date": 1744131305958,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_LAZY-cpu-mock-train_default.test.yaml]",
            "value": 0.04434858472390428,
            "unit": "iter/sec",
            "range": "stddev: 0.15236646969956175",
            "extra": "mean: 22.548633879200008 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-mock-train_default.test.yaml]",
            "value": 0.05298105789341611,
            "unit": "iter/sec",
            "range": "stddev: 0.21048712472365072",
            "extra": "mean: 18.874670302200002 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-mock-train_default.test.yaml]",
            "value": 0.13514516917815456,
            "unit": "iter/sec",
            "range": "stddev: 0.10577361700082187",
            "extra": "mean: 7.399450576599998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-mock-train_default.test.yaml]",
            "value": 0.007131401828677064,
            "unit": "iter/sec",
            "range": "stddev: 2.0187477121762014",
            "extra": "mean: 140.22488481559992 sec\nrounds: 5"
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
          "id": "15f367623238390035d81215398ef9e75100fa6a",
          "message": "Torch-first data loader without dask (#178)\n\nI did some profiling and continue to see a huge amount of time spent in\nxarray indexing. I built a dataloader which goes as quickly as possible\ninto (CPU) torch and does not use dask. On my laptop this is 3x faster\nthan the eager data loader, and on my test GPU machine it's about 5x\nfaster in the data loader benchmarks. Also seems to eliminate any\nwaiting for data on my GPU machine with 4 workers when running with\nbatch size == 2.",
          "timestamp": "2025-04-08T13:09:10-04:00",
          "tree_id": "528cef2823966e8e35a9ade9ef153d4a66ead93d",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/15f367623238390035d81215398ef9e75100fa6a"
        },
        "date": 1744134654401,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-mock-train_default.test.yaml]",
            "value": 0.1964656921019934,
            "unit": "iter/sec",
            "range": "stddev: 0.6569722111700642",
            "extra": "mean: 5.089947202999997 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_LAZY-cpu-mock-train_default.test.yaml]",
            "value": 0.5285101690921249,
            "unit": "iter/sec",
            "range": "stddev: 0.08281959057327175",
            "extra": "mean: 1.8921111805999886 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-mock-train_default.test.yaml]",
            "value": 0.05483215258736427,
            "unit": "iter/sec",
            "range": "stddev: 1.3622184130786024",
            "extra": "mean: 18.237474781000003 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-mock-train_default.test.yaml]",
            "value": 0.1386892461082167,
            "unit": "iter/sec",
            "range": "stddev: 0.09775715987708272",
            "extra": "mean: 7.210364379799989 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-mock-train_default.test.yaml]",
            "value": 0.007324764178857834,
            "unit": "iter/sec",
            "range": "stddev: 0.37396371121412264",
            "extra": "mean: 136.52316655960004 sec\nrounds: 5"
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
          "id": "975c149cf969ae1284f5efbf187330a4ca5cf7f8",
          "message": "Using local logger in train.py (#189)\n\nStarts to address #102 (incrementally).",
          "timestamp": "2025-04-09T15:40:49-07:00",
          "tree_id": "c5ff4cb3f26f0dd308b139c3d1e0d39f09d2c947",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/975c149cf969ae1284f5efbf187330a4ca5cf7f8"
        },
        "date": 1744241128887,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-mock-train_default.test.yaml]",
            "value": 0.17939521514102835,
            "unit": "iter/sec",
            "range": "stddev: 0.3083663912689785",
            "extra": "mean: 5.574284683199982 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_LAZY-cpu-mock-train_default.test.yaml]",
            "value": 0.5084435144055348,
            "unit": "iter/sec",
            "range": "stddev: 0.08602240615223306",
            "extra": "mean: 1.9667868143999954 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-mock-train_default.test.yaml]",
            "value": 0.05295516452764494,
            "unit": "iter/sec",
            "range": "stddev: 0.27613393410732273",
            "extra": "mean: 18.883899406600005 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-mock-train_default.test.yaml]",
            "value": 0.1356229148649359,
            "unit": "iter/sec",
            "range": "stddev: 0.10030427155859396",
            "extra": "mean: 7.373385249800004 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-mock-train_default.test.yaml]",
            "value": 0.007256988417581169,
            "unit": "iter/sec",
            "range": "stddev: 0.08982559766039906",
            "extra": "mean: 137.79820808000002 sec\nrounds: 5"
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
          "id": "00a492c73f87cd7c5a23ca0b3011f24f57705014",
          "message": "Inline inference refactors (#193)\n\nComponents — target extraction, aggregator recording, wandb logging, and\nwriting — are now handled in batches rather than singular timesteps.\nThis allowed me to remove the record_every parameter as well.\nAdditionally, some other minor optimizations were made.\n\nHere is a comparison of the time taken in minutes, before and after\noptimization:\n\n| Dataset (Period) | Old Inference (With Write) | Old Inference (Without\nWrite) | Optimized Inference (With Write) | Optimized Inference (Without\nWrite) |\n| ------------- | ------------- | ------------- | ------------- |\n------------- |\n| OM4 (~8 years) |\n[14:08](https://wandb.ai/m2lines/ocean-emulators/runs/pjzsf71m) |\n[13:14](https://wandb.ai/m2lines/ocean-emulators/runs/025m3u3s) |\n[03:52](https://wandb.ai/m2lines/ocean-emulators/runs/rubapmx5) |\n[02:54](https://wandb.ai/m2lines/ocean-emulators/runs/i9kkjb85) |\n| CM4 (20 years) |\n[24:52](https://wandb.ai/m2lines/ocean-emulators/runs/zcbmfvme) |\n[22:45](https://wandb.ai/m2lines/ocean-emulators/runs/v8zazvdi) | [05:27\n](https://wandb.ai/m2lines/ocean-emulators/runs/1ziu9lrt) |\n[03:17](https://wandb.ai/m2lines/ocean-emulators/runs/mxk43iwq) |\n\nI made a comparison between writing the actual zarr and skipping write\nbecause during train we don't actually save the data produced by the\ninference rollout but in standalone inference, we do.\n \nTraining should now be cut down from ~4 days to 2.5 days and Inference\nshould now go from 1152 SYPD (~100x) to 4800 SYPD (400x).\n\nAdd tests\n- [x] Ensure correct data fed to model (Use time indexed values for\ndata)\n- [x] ~~Check if batched output metrics are different from single output\nmetrics~~ This test is not needed since Alex caught my bug\n- [x] Fix time returned so we save appropriate time in zarr\n\n\nNOTE: The above times with the optimization actually had a minor bug in\nchoosing the next prognostic but there is no change in performance\nbecause I just had to change an index. See this\n[comment](https://github.com/suryadheeshjith/Ocean_Emulator/pull/193#discussion_r2037893244).\nA fixed run with OM4 with write is logged\n[here](https://wandb.ai/m2lines/ocean-emulators/runs/b3qznqb0) and time\ntaken was: 03:46\n\nCloses #182",
          "timestamp": "2025-04-11T13:35:46-04:00",
          "tree_id": "c025ae360c18694549e1c5e02bd06d92c2d3c75c",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/00a492c73f87cd7c5a23ca0b3011f24f57705014"
        },
        "date": 1744395347259,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-mock-train_default.test.yaml]",
            "value": 0.19850492436771733,
            "unit": "iter/sec",
            "range": "stddev: 0.39117295539868013",
            "extra": "mean: 5.037658401599981 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_LAZY-cpu-mock-train_default.test.yaml]",
            "value": 0.5209399744128496,
            "unit": "iter/sec",
            "range": "stddev: 0.0342746450752714",
            "extra": "mean: 1.9196069588 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-mock-train_default.test.yaml]",
            "value": 0.05600950856866077,
            "unit": "iter/sec",
            "range": "stddev: 0.8744278711658546",
            "extra": "mean: 17.854111302800003 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-mock-train_default.test.yaml]",
            "value": 0.13698751387074268,
            "unit": "iter/sec",
            "range": "stddev: 0.10305196056268658",
            "extra": "mean: 7.299935386400034 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-mock-train_default.test.yaml]",
            "value": 0.0077829152542061375,
            "unit": "iter/sec",
            "range": "stddev: 1.9181893011254751",
            "extra": "mean: 128.48655899979997 sec\nrounds: 5"
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
          "id": "da87f40dd31ea461e48420355907cdb32dbab072",
          "message": "Document how to use multitons (#209)\n\nFixes #161",
          "timestamp": "2025-04-15T11:05:24-04:00",
          "tree_id": "10ffb70f5dbdc62a15191b8394f03405ac07496f",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/da87f40dd31ea461e48420355907cdb32dbab072"
        },
        "date": 1744732051687,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-mock-train_default.test.yaml]",
            "value": 0.18924417927076706,
            "unit": "iter/sec",
            "range": "stddev: 0.3488069135644665",
            "extra": "mean: 5.284178376599994 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_LAZY-cpu-mock-train_default.test.yaml]",
            "value": 0.5075206643613437,
            "unit": "iter/sec",
            "range": "stddev: 0.08844116455209254",
            "extra": "mean: 1.9703631205999954 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-mock-train_default.test.yaml]",
            "value": 0.05269758966668105,
            "unit": "iter/sec",
            "range": "stddev: 0.7238168812415677",
            "extra": "mean: 18.976199980399997 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-mock-train_default.test.yaml]",
            "value": 0.13533655603402706,
            "unit": "iter/sec",
            "range": "stddev: 0.07250294091469198",
            "extra": "mean: 7.388986607200013 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-mock-train_default.test.yaml]",
            "value": 0.007497670774905414,
            "unit": "iter/sec",
            "range": "stddev: 1.0481466602953116",
            "extra": "mean: 133.37475464339997 sec\nrounds: 5"
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
          "id": "8b2e5b23c4ec05300e6ec0fc9fdf83059da78cd1",
          "message": "Inline Inference - Clarity Refactor (#198)\n\n- Added a comment to provide clarity on what we are doing during\ninference and validation.",
          "timestamp": "2025-04-15T11:11:20-04:00",
          "tree_id": "46679b918f1c5e268e3ab0c4bc9bb344c1b961d5",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/8b2e5b23c4ec05300e6ec0fc9fdf83059da78cd1"
        },
        "date": 1744732543190,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-mock-train_default.test.yaml]",
            "value": 0.18522912258955643,
            "unit": "iter/sec",
            "range": "stddev: 0.15039049787711062",
            "extra": "mean: 5.398719089199972 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_LAZY-cpu-mock-train_default.test.yaml]",
            "value": 0.5139901960924032,
            "unit": "iter/sec",
            "range": "stddev: 0.08395858792927487",
            "extra": "mean: 1.9455624010000065 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-mock-train_default.test.yaml]",
            "value": 0.05403827763089332,
            "unit": "iter/sec",
            "range": "stddev: 2.0364164586837523",
            "extra": "mean: 18.505401057199993 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-mock-train_default.test.yaml]",
            "value": 0.1373598214032791,
            "unit": "iter/sec",
            "range": "stddev: 0.11643881377529476",
            "extra": "mean: 7.280149244399991 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-mock-train_default.test.yaml]",
            "value": 0.007649288108670579,
            "unit": "iter/sec",
            "range": "stddev: 0.4319537832999786",
            "extra": "mean: 130.7311197844 sec\nrounds: 5"
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
          "id": "49404e57add8a29a90a65115a8c3cb323238808b",
          "message": "Better colors for wandb map plots (#201)\n\nBetter coloring of land cells in the plots",
          "timestamp": "2025-04-15T11:57:53-04:00",
          "tree_id": "7211baffd337a8ada8e00e813c1035f14ed63174",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/49404e57add8a29a90a65115a8c3cb323238808b"
        },
        "date": 1744735160779,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-mock-train_default.test.yaml]",
            "value": 0.20367659192179657,
            "unit": "iter/sec",
            "range": "stddev: 0.38032993443959745",
            "extra": "mean: 4.909744367599979 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_LAZY-cpu-mock-train_default.test.yaml]",
            "value": 0.5154965877331783,
            "unit": "iter/sec",
            "range": "stddev: 0.04200177036315707",
            "extra": "mean: 1.9398770501999933 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-mock-train_default.test.yaml]",
            "value": 0.05226099043953667,
            "unit": "iter/sec",
            "range": "stddev: 1.7609252884375242",
            "extra": "mean: 19.13473111760003 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-mock-train_default.test.yaml]",
            "value": 0.13883513202603812,
            "unit": "iter/sec",
            "range": "stddev: 0.10275785496986953",
            "extra": "mean: 7.202787834800006 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-mock-train_default.test.yaml]",
            "value": 0.007689187274762291,
            "unit": "iter/sec",
            "range": "stddev: 0.5153105580971139",
            "extra": "mean: 130.05275645740008 sec\nrounds: 5"
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
          "id": "8352f9e3600ea516fa0660e35b24e5de5307bbfa",
          "message": "Minor stepper refactors and norm metrics fix  (#202)\n\n- Fix the calculation of metrics for normalized data. We were still\nconsidering 0s at land.\n- Renamed output types and moved them to output.py\n\n---------\n\nCo-authored-by: Jesse Rusak <jesse@openathena.ai>",
          "timestamp": "2025-04-15T12:19:34-04:00",
          "tree_id": "44494de794a6c2ce8f13cb30f9f5b06d9303fd3b",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/8352f9e3600ea516fa0660e35b24e5de5307bbfa"
        },
        "date": 1744736561431,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cpu-mock-train_default.test.yaml]",
            "value": 0.19432120320392143,
            "unit": "iter/sec",
            "range": "stddev: 0.13286844129840109",
            "extra": "mean: 5.146118815199986 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_LAZY-cpu-mock-train_default.test.yaml]",
            "value": 0.49867078360142997,
            "unit": "iter/sec",
            "range": "stddev: 0.09795502878856381",
            "extra": "mean: 2.0053310378000107 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cpu-mock-train_default.test.yaml]",
            "value": 0.04671760337355208,
            "unit": "iter/sec",
            "range": "stddev: 0.9814194832053866",
            "extra": "mean: 21.405207625999992 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cpu-mock-train_default.test.yaml]",
            "value": 0.13302856395324836,
            "unit": "iter/sec",
            "range": "stddev: 0.14136123399801237",
            "extra": "mean: 7.517182552999975 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cpu-mock-train_default.test.yaml]",
            "value": 0.007376641350621573,
            "unit": "iter/sec",
            "range": "stddev: 0.3245401010240525",
            "extra": "mean: 135.56304996659998 sec\nrounds: 5"
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
          "id": "97f2bb5015b51f43ffffd5435c924eb8b1438cd4",
          "message": "Ensuring uniqueness property of interpretable data. (#98)\n\nAlso, added a minor equality fix which tolerates noisier data. Luckily,\nthe uniqueness property is still preserved.",
          "timestamp": "2025-03-06T16:49:08-08:00",
          "tree_id": "4b7d7f3d7897fb96084d8ff265693f58ba2628e9",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/97f2bb5015b51f43ffffd5435c924eb8b1438cd4"
        },
        "date": 1741310518219,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cuda-train]",
            "value": 0.22309867030627117,
            "unit": "iter/sec",
            "range": "stddev: 0.21069167186643556",
            "extra": "mean: 4.482321649999949 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cuda-val]",
            "value": 0.4720718166140207,
            "unit": "iter/sec",
            "range": "stddev: 0.009172992028600273",
            "extra": "mean: 2.1183217570000124 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda]",
            "value": 0.1919366322234432,
            "unit": "iter/sec",
            "range": "stddev: 0.06809184254506163",
            "extra": "mean: 5.210052861799977 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda]",
            "value": 0.011739087073800239,
            "unit": "iter/sec",
            "range": "stddev: 0.6091465259295263",
            "extra": "mean: 85.185499836 sec\nrounds: 5"
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
          "id": "9bf9391bb918a926b0d2c530fde14fae0d78cfe2",
          "message": "Added some typing definitions (array and otherwise). (#97)\n\nA pre-requisite step in refactoring the data loader (for me, anyway) is\njust understanding what's going on. To that end, I've added\nbeartype-enforced jaxtyping and standard typing definitions to parts of\nthe codebase that the data loader touches.\n\nThis does not aim for complete typing coverage of the codebase. Rather,\nthese annotations have been added so I can feel confident in pulling\narray parsing routines up and down. Besides serving as useful\ndocumentation, these perform runtime type checks to ensure correctness.\n\n---------\n\nCo-authored-by: Jesse Rusak <jesse@openathena.ai>",
          "timestamp": "2025-03-07T12:55:01-08:00",
          "tree_id": "6b4029a4aae29a5d17c745b3d85bfc41f1296dfb",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/9bf9391bb918a926b0d2c530fde14fae0d78cfe2"
        },
        "date": 1741382897875,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cuda-train]",
            "value": 0.22992087375214657,
            "unit": "iter/sec",
            "range": "stddev: 0.23922939074319327",
            "extra": "mean: 4.349322372000006 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cuda-val]",
            "value": 0.526724586413213,
            "unit": "iter/sec",
            "range": "stddev: 0.00403647286869211",
            "extra": "mean: 1.8985253884000486 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda]",
            "value": 0.18910201421960646,
            "unit": "iter/sec",
            "range": "stddev: 0.09447154343676127",
            "extra": "mean: 5.288150970399966 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda]",
            "value": 0.011287388306822532,
            "unit": "iter/sec",
            "range": "stddev: 0.4751405139903926",
            "extra": "mean: 88.5944536342 sec\nrounds: 5"
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
          "id": "0aceb9715af89c643c99944c2c06358d4e48faeb",
          "message": "fix training when run by main script (#100)\n\nTurns out the handle_logging requires the directory to be there first,\nso create them first (but also try again to recreate them in the\nconstructor)",
          "timestamp": "2025-03-10T12:54:38-04:00",
          "tree_id": "8fed0ad94d9d50ba2de540cb6b793a7f92a27590",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/0aceb9715af89c643c99944c2c06358d4e48faeb"
        },
        "date": 1741627722387,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cuda-train]",
            "value": 0.22426568282225387,
            "unit": "iter/sec",
            "range": "stddev: 0.4167917507396066",
            "extra": "mean: 4.458996969200006 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cuda-val]",
            "value": 0.5275285167846749,
            "unit": "iter/sec",
            "range": "stddev: 0.01090028772761698",
            "extra": "mean: 1.8956321188000858 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda]",
            "value": 0.18882281577043872,
            "unit": "iter/sec",
            "range": "stddev: 0.09797101964973079",
            "extra": "mean: 5.295970171399995 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda]",
            "value": 0.011281859647242404,
            "unit": "iter/sec",
            "range": "stddev: 0.35685856422703643",
            "extra": "mean: 88.63786922259996 sec\nrounds: 5"
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
          "id": "a97aedbdefcb457c38975dcf92760f06ae6bd8fe",
          "message": "Simple utility script to clone Samudra data locally. (#107)\n\nClones OM4 in about 600s using ~100MB chunks. Saves means and stds as\nNetCDF. Is packaged as a `uv` script (it self-describes dependencies).\n\nThis change is in service of setting up remote GPU runs and chunk\ntuning.\n\n---------\n\nCo-authored-by: Surya Dheeshjith <41594351+suryadheeshjith@users.noreply.github.com>",
          "timestamp": "2025-03-12T14:58:58-07:00",
          "tree_id": "cbaadc70bab29f387c6851c16096922d6698e51d",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/a97aedbdefcb457c38975dcf92760f06ae6bd8fe"
        },
        "date": 1741818771578,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cuda-train]",
            "value": 0.23123529478017207,
            "unit": "iter/sec",
            "range": "stddev: 0.24013835136856687",
            "extra": "mean: 4.324599326200041 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cuda-val]",
            "value": 0.5322367196088752,
            "unit": "iter/sec",
            "range": "stddev: 0.005233017962253959",
            "extra": "mean: 1.8788632260000213 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda]",
            "value": 0.18940782889244237,
            "unit": "iter/sec",
            "range": "stddev: 0.09042051518324944",
            "extra": "mean: 5.279612811400011 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda]",
            "value": 0.011096605515861198,
            "unit": "iter/sec",
            "range": "stddev: 1.4333323437858405",
            "extra": "mean: 90.1176489126 sec\nrounds: 5"
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
          "id": "3d56ca15da553046e13312b57b682617f96021e3",
          "message": "Fix: Opening with consolidated metadata. (#117)\n\nI expect that this will fix the performance regression from switching\nbetween Zarr v2 --> Zarr v3.",
          "timestamp": "2025-03-14T14:59:55-07:00",
          "tree_id": "b13231b0e0470131ad33bec96a8355d4398e15da",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/3d56ca15da553046e13312b57b682617f96021e3"
        },
        "date": 1741992451844,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cuda-train]",
            "value": 0.10245816811119454,
            "unit": "iter/sec",
            "range": "stddev: 0.7779846083122064",
            "extra": "mean: 9.760080806000087 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cuda-val]",
            "value": 0.23041606824007824,
            "unit": "iter/sec",
            "range": "stddev: 0.03554523288931647",
            "extra": "mean: 4.339975105199983 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda]",
            "value": 0.10193199337675968,
            "unit": "iter/sec",
            "range": "stddev: 0.15138358502023613",
            "extra": "mean: 9.810462514000028 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda]",
            "value": 0.00858241174217352,
            "unit": "iter/sec",
            "range": "stddev: 1.332111811423076",
            "extra": "mean: 116.51736482019996 sec\nrounds: 5"
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
          "id": "060cd1355ab2faa143c31b2a08b2ab2f5a45f33d",
          "message": "Rework round-tripping/interpretable data with bit fields (#118)\n\nWhen I went to make another change, this test started failing again, I\nthink because it generated new test cases. Try to resolve the\ninstability by avoiding complexity of rounding and numeric error by\npacking data as smaller bitfields into the float64.",
          "timestamp": "2025-03-19T15:22:57-04:00",
          "tree_id": "6c9bbf536c4b5ca330bd2e226e4037795309748b",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/060cd1355ab2faa143c31b2a08b2ab2f5a45f33d"
        },
        "date": 1742415156704,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cuda-train]",
            "value": 0.09827433913918458,
            "unit": "iter/sec",
            "range": "stddev: 0.5111526659955918",
            "extra": "mean: 10.175596282399965 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cuda-val]",
            "value": 0.23001381182152827,
            "unit": "iter/sec",
            "range": "stddev: 0.10013621893105613",
            "extra": "mean: 4.347565009600021 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda]",
            "value": 0.09905162804185527,
            "unit": "iter/sec",
            "range": "stddev: 0.11227607520478092",
            "extra": "mean: 10.09574521659997 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda]",
            "value": 0.008568044966964504,
            "unit": "iter/sec",
            "range": "stddev: 0.9181040182591051",
            "extra": "mean: 116.7127394704 sec\nrounds: 5"
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
          "id": "dd81ee6f1de5f290236b2f5afc6561e498ee1a58",
          "message": "Downgrading to Zarr v2. (#130)\n\nThis should be the same performance as before (from my tests, it should\neven be a little bit faster). Further, Zarr v2 is much more stable.",
          "timestamp": "2025-03-20T12:32:37-07:00",
          "tree_id": "ed8fea33a3ecc9a2eebd79494c18992bec151afd",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/dd81ee6f1de5f290236b2f5afc6561e498ee1a58"
        },
        "date": 1742501338289,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cuda-train]",
            "value": 0.18610022833182838,
            "unit": "iter/sec",
            "range": "stddev: 0.22287913552369473",
            "extra": "mean: 5.373448538799948 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cuda-val]",
            "value": 0.409732657480782,
            "unit": "iter/sec",
            "range": "stddev: 0.051186316318432915",
            "extra": "mean: 2.4406158057999163 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda]",
            "value": 0.15904303284458252,
            "unit": "iter/sec",
            "range": "stddev: 0.12067871823738407",
            "extra": "mean: 6.287606455400055 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda]",
            "value": 0.01105097526473756,
            "unit": "iter/sec",
            "range": "stddev: 0.26376087559888944",
            "extra": "mean: 90.48975099880003 sec\nrounds: 5"
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
          "id": "464c1d08f911f0ccdf1f50b74ec8c9f4245226b3",
          "message": "Move everything into src/ocean_emulators, get rid of ocean_emulators_main (#110)\n\nFollow-up to https://github.com/suryadheeshjith/Ocean_Emulator/pull/108\nwhich moves everything into a named module and fixes #75.",
          "timestamp": "2025-03-20T15:37:54-04:00",
          "tree_id": "ec51f3a9c93c18a3be45d1bbc3719dd212d05df7",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/464c1d08f911f0ccdf1f50b74ec8c9f4245226b3"
        },
        "date": 1742501584127,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cuda-train]",
            "value": 0.1850146335566775,
            "unit": "iter/sec",
            "range": "stddev: 0.2958826975038004",
            "extra": "mean: 5.4049778699999935 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cuda-val]",
            "value": 0.40417749575591005,
            "unit": "iter/sec",
            "range": "stddev: 0.03546283354902511",
            "extra": "mean: 2.4741605124000214 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda]",
            "value": 0.16336636997334078,
            "unit": "iter/sec",
            "range": "stddev: 0.06829334469491871",
            "extra": "mean: 6.121210872000074 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda]",
            "value": 0.011365303890399539,
            "unit": "iter/sec",
            "range": "stddev: 0.568414207761383",
            "extra": "mean: 87.9870885674 sec\nrounds: 5"
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
          "id": "d84192f5b04080375647ef2929d9c9c5313a504a",
          "message": "Restore missing function (#131)",
          "timestamp": "2025-03-20T21:59:35-04:00",
          "tree_id": "8db37413cbea31929537d2bb59230a31e6956af0",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/d84192f5b04080375647ef2929d9c9c5313a504a"
        },
        "date": 1742524626297,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cuda-train]",
            "value": 0.18078555476696706,
            "unit": "iter/sec",
            "range": "stddev: 0.3401281043368643",
            "extra": "mean: 5.53141539039998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cuda-val]",
            "value": 0.3958579906166685,
            "unit": "iter/sec",
            "range": "stddev: 0.048476780761183974",
            "extra": "mean: 2.5261584298000344 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda]",
            "value": 0.1550344972459677,
            "unit": "iter/sec",
            "range": "stddev: 0.13953052679821082",
            "extra": "mean: 6.450177333200008 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda]",
            "value": 0.010456419583935823,
            "unit": "iter/sec",
            "range": "stddev: 1.123087082054234",
            "extra": "mean: 95.63502994239998 sec\nrounds: 5"
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
          "id": "151275c56c72857b639e3312eaa0abe8e97f3a46",
          "message": "fix build, not sure how this was working in CI before (#133)\n\nCurrently, `uv run python` doesn't let you run `import ocean_emulators`.\nThis fixes that.",
          "timestamp": "2025-03-21T10:44:15-04:00",
          "tree_id": "b97592be8143c1817d59f2b91723d921274fcb14",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/151275c56c72857b639e3312eaa0abe8e97f3a46"
        },
        "date": 1742570491454,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cuda-train]",
            "value": 0.18945313695449525,
            "unit": "iter/sec",
            "range": "stddev: 0.2869304795697792",
            "extra": "mean: 5.278350182399936 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[cuda-val]",
            "value": 0.4184489726175985,
            "unit": "iter/sec",
            "range": "stddev: 0.01668241704065987",
            "extra": "mean: 2.389777644200012 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda]",
            "value": 0.16460058898513266,
            "unit": "iter/sec",
            "range": "stddev: 0.07815265749341407",
            "extra": "mean: 6.075312404199986 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda]",
            "value": 0.011175783861375745,
            "unit": "iter/sec",
            "range": "stddev: 0.8065705436120311",
            "extra": "mean: 89.47918216780003 sec\nrounds: 5"
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
          "id": "e37711d4d1f172bf499d0f6d2fd39029fb23f4bd",
          "message": "Restore benchmarks to old behavior (#135)\n\nThis was a silly bug but after looking at this we probably want to leave\nthem separate so the benchmark web format doesn't change.",
          "timestamp": "2025-03-21T09:24:29-07:00",
          "tree_id": "4c5e2fa60fb5dda47a5c12e2d6803b748ab7676f",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/e37711d4d1f172bf499d0f6d2fd39029fb23f4bd"
        },
        "date": 1742576457377,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cuda-train]",
            "value": 0.19089439244279183,
            "unit": "iter/sec",
            "range": "stddev: 0.3426076267989491",
            "extra": "mean: 5.2384985604000125 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cuda-val]",
            "value": 0.41617666726426783,
            "unit": "iter/sec",
            "range": "stddev: 0.061276922170954234",
            "extra": "mean: 2.4028257196000142 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_cm4.test.yaml-cuda]",
            "value": 0.1636092985271038,
            "unit": "iter/sec",
            "range": "stddev: 0.07870947796219925",
            "extra": "mean: 6.112122043199998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_cm4.test.yaml-cuda]",
            "value": 0.011514881760466143,
            "unit": "iter/sec",
            "range": "stddev: 0.543457370684461",
            "extra": "mean: 86.84413967959999 sec\nrounds: 5"
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
          "id": "763edcb8ac6f77dae876b94f33cfa928c0c370bf",
          "message": "Speed up tests (#136)\n\nCuts test time on my machine from 1:45 to :40. This does change the\nbenchmark test run, I could break out a separate config for that if\ndesired to keep it consistent. WDYT @alxmrs?\n\nBefore:\n```\n==================================================================================== slowest durations =====================================================================================\n67.17s call     tests/test_trainer.py::test_trainer__mini_2step[train_cm4_2step.test.yaml-cpu]\n6.48s call     tests/test_datasets.py::test__data_is_not_zeros[train_cm4.test.yaml-cpu-train]\n6.42s call     tests/test_datasets.py::test_train__data_shape[train_cm4.test.yaml-cpu]\n6.26s call     tests/test_datasets.py::test_train__loads_correct_number_of_samples[train_cm4.test.yaml-cpu]\n2.57s call     tests/test_datasets.py::test_inference__data_is_not_zero[train_cm4.test.yaml-cpu]\n2.48s call     tests/test_datasets.py::test__data_is_not_zeros[train_cm4.test.yaml-cpu-val]\n2.41s call     tests/test_datasets.py::test_inference__data_shape[train_cm4.test.yaml-cpu]\n2.37s call     tests/test_datasets.py::test_val__data_shape[train_cm4.test.yaml-cpu]\n2.23s call     tests/test_datasets.py::test_val__loads_correct_number_of_samples[train_cm4.test.yaml-cpu]\n1.86s setup    tests/test_datasets.py::test_test_util__data_source_roundtrip[train_cm4.test.yaml-cpu]\n1.61s setup    tests/test_datasets.py::test_test_util__data_source_roundtrip[train_cm4_2step.test.yaml-cpu]\n0.79s setup    tests/test_datasets.py::test_train__loads_correct_number_of_samples[train_cm4.test.yaml-cpu]\n0.36s call     tests/test_datasets.py::test_test_util__data_source_roundtrip[train_cm4.test.yaml-cpu]\n0.28s setup    tests/test_datasets.py::test_train__loads_correct_number_of_samples[train_cm4_2step.test.yaml-cpu]\n0.14s teardown tests/test_trainer.py::test_trainer__mini_2step[train_cm4_2step.test.yaml-cpu]\n\n(65 durations < 0.005s hidden.  Use -vv to show these durations.)\n========================================================== 16 passed, 16 skipped, 52 deselected, 17 warnings in 104.09s (0:01:44) ==========================================================\npytest -m 'not cuda and not manual' --durations=0  326.39s user 636.78s system 914% cpu 1:45.30 total'\n```\n\nAfter:\n```\n==================================================================================== slowest durations =====================================================================================\n18.48s call     tests/test_trainer.py::test_trainer__mini_2step[train_cm4_2step.test.yaml-cpu]\n2.46s call     tests/test_datasets.py::test_train__data_shape[train_cm4.test.yaml-cpu]\n2.40s call     tests/test_datasets.py::test__data_is_not_zeros[train_cm4.test.yaml-cpu-train]\n2.27s call     tests/test_datasets.py::test_train__loads_correct_number_of_samples[train_cm4.test.yaml-cpu]\n1.80s call     tests/test_datasets.py::test_val__loads_correct_number_of_samples[train_cm4.test.yaml-cpu]\n1.78s call     tests/test_datasets.py::test__data_is_not_zeros[train_cm4.test.yaml-cpu-val]\n1.76s call     tests/test_datasets.py::test_inference__data_is_not_zero[train_cm4.test.yaml-cpu]\n1.75s call     tests/test_datasets.py::test_val__data_shape[train_cm4.test.yaml-cpu]\n1.65s call     tests/test_datasets.py::test_inference__data_shape[train_cm4.test.yaml-cpu]\n1.26s setup    tests/test_datasets.py::test_test_util__data_source_roundtrip[train_cm4.test.yaml-cpu]\n0.92s setup    tests/test_datasets.py::test_test_util__data_source_roundtrip[train_cm4_2step.test.yaml-cpu]\n0.86s setup    tests/test_datasets.py::test_train__loads_correct_number_of_samples[train_cm4.test.yaml-cpu]\n0.66s call     tests/test_datasets.py::test_test_util__data_source_roundtrip[train_cm4.test.yaml-cpu]\n0.28s setup    tests/test_datasets.py::test_train__loads_correct_number_of_samples[train_cm4_2step.test.yaml-cpu]\n0.08s teardown tests/test_trainer.py::test_trainer__mini_2step[train_cm4_2step.test.yaml-cpu]\n\n(65 durations < 0.005s hidden.  Use -vv to show these durations.)\n=============================================================== 16 passed, 16 skipped, 52 deselected, 15 warnings in 39.08s ================================================================\npytest -m 'not cuda and not manual' --durations=0  138.88s user 226.06s system 906% cpu 40.267 total\n```",
          "timestamp": "2025-03-21T15:20:21-04:00",
          "tree_id": "e460bec79b4d8cf9088e7c58c454adada17e6b62",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/763edcb8ac6f77dae876b94f33cfa928c0c370bf"
        },
        "date": 1742586958562,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cuda-train]",
            "value": 0.18485893559868144,
            "unit": "iter/sec",
            "range": "stddev: 0.34160535541389386",
            "extra": "mean: 5.409530227799996 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cuda-val]",
            "value": 0.41992808019931815,
            "unit": "iter/sec",
            "range": "stddev: 0.014969422745414223",
            "extra": "mean: 2.3813601594000375 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_cm4.test.yaml-cuda]",
            "value": 0.16229664114498024,
            "unit": "iter/sec",
            "range": "stddev: 0.07126051537806542",
            "extra": "mean: 6.161556967199931 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_cm4.test.yaml-cuda]",
            "value": 0.011031339451521656,
            "unit": "iter/sec",
            "range": "stddev: 0.9783622752292289",
            "extra": "mean: 90.65082299340001 sec\nrounds: 5"
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
          "id": "3e60b05c5c81f03626e4cfac64f593f57ddcba0d",
          "message": "Update ruff to avoid fighting with VS Code version (#137)\n\nThese keep flipping back and forth for me since the VS Code extension\nversion is more recent.",
          "timestamp": "2025-03-21T16:20:22-04:00",
          "tree_id": "93de6f752a78e8021c83006d784af175ab86c07a",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/3e60b05c5c81f03626e4cfac64f593f57ddcba0d"
        },
        "date": 1742590538841,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cuda-train]",
            "value": 0.18522225042818355,
            "unit": "iter/sec",
            "range": "stddev: 0.3699769829993322",
            "extra": "mean: 5.398919393799997 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cuda-val]",
            "value": 0.4099613699304605,
            "unit": "iter/sec",
            "range": "stddev: 0.027624606223565273",
            "extra": "mean: 2.439254215999972 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_cm4.test.yaml-cuda]",
            "value": 0.16248715919185935,
            "unit": "iter/sec",
            "range": "stddev: 0.12679921705164654",
            "extra": "mean: 6.154332471399994 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_cm4.test.yaml-cuda]",
            "value": 0.011283811944055994,
            "unit": "iter/sec",
            "range": "stddev: 0.5983125529445893",
            "extra": "mean: 88.62253332099999 sec\nrounds: 5"
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
          "id": "0c4abfac895fd2788eac49061755d0c5aa406aa0",
          "message": "Resolve CFTime warning (#139)\n\nThe underlying [warning\n](https://github.com/Unidata/cftime/blob/eab0212df73decd5a420adaae7938b942d131c95/src/cftime/_cftime.pyx#L1151)\nwas due to us passing a date with year < 0 for a calendar that doesn't\nsupport that.\n\nThis was for 2 reasons: first, we had no minimum in the possible\ngenerated dates and second because the fromordinal function we were\nusing takes the number of days since year -4713 (yes, negative four\nthousand years BCE) but the python datetime toordinal produces the\nnumber of days since year 1. So, we have a minimum now and we are using\na conversion function which is a little less silly.\n\nFixes #112",
          "timestamp": "2025-03-24T09:08:43-04:00",
          "tree_id": "5da7886abe8dc19aaef0095065bae2d814b441f4",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/0c4abfac895fd2788eac49061755d0c5aa406aa0"
        },
        "date": 1742824207251,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cuda-train]",
            "value": 0.17135109607620952,
            "unit": "iter/sec",
            "range": "stddev: 0.4265057691279516",
            "extra": "mean: 5.835970839400079 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cuda-val]",
            "value": 0.38862911173253434,
            "unit": "iter/sec",
            "range": "stddev: 0.022200079304087607",
            "extra": "mean: 2.5731474297999286 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_cm4.test.yaml-cuda]",
            "value": 0.14804940483196855,
            "unit": "iter/sec",
            "range": "stddev: 0.13284364178961677",
            "extra": "mean: 6.754501993000031 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_cm4.test.yaml-cuda]",
            "value": 0.010017219201281952,
            "unit": "iter/sec",
            "range": "stddev: 2.482006448489079",
            "extra": "mean: 99.82810397840004 sec\nrounds: 5"
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
          "id": "eb98afdfdc4bf7a30a52d911c8ea63e0256e15f2",
          "message": "March refactor - renaming (#141)\n\n* Bunch of renaming and removal of unused experiment configs in\nconstants. The most important rename here is inputs -> prognostic, extra\n-> boundary and removal of output variables names/keys since they are\nthe same as input variables. This functionality is okay for now since\nthe codebase depends on this fact. I don't see a near future where this\nis going to change.\n* The scripts for training on gantry, empire ai etc. may not work and I\nam planning to fix that in a later PR\n\nFollowing Draft PR #134 \n\nCloses issue #80",
          "timestamp": "2025-03-24T12:48:06-04:00",
          "tree_id": "59391f69030ea4c0c6afe39efe1c8505d88b171d",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/eb98afdfdc4bf7a30a52d911c8ea63e0256e15f2"
        },
        "date": 1742837088463,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cuda-train]",
            "value": 0.18141295538540508,
            "unit": "iter/sec",
            "range": "stddev: 0.4410208124203638",
            "extra": "mean: 5.512285480800074 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cuda-val]",
            "value": 0.40290182828266446,
            "unit": "iter/sec",
            "range": "stddev: 0.07414456138204527",
            "extra": "mean: 2.481994197599988 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_cm4.test.yaml-cuda]",
            "value": 0.15999076494395098,
            "unit": "iter/sec",
            "range": "stddev: 0.09727689425106809",
            "extra": "mean: 6.250360765199957 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_cm4.test.yaml-cuda]",
            "value": 0.010980219298517023,
            "unit": "iter/sec",
            "range": "stddev: 0.5754198441006614",
            "extra": "mean: 91.07286228200005 sec\nrounds: 5"
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
          "id": "9799852f5388729098c91ceba1ac5b3551e6af9e",
          "message": "Wrote an Xarray-based OM4 Dataloader. (#123)\n\nFixes #119. This new loader is behind a feature flag. After I manually\nprofile the new loader, I will make a follow-up PR to make this one the\ndefault.",
          "timestamp": "2025-03-24T10:52:58-07:00",
          "tree_id": "e4c10fbc9947ac98469f38c568fbe31e8bd30dbc",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/9799852f5388729098c91ceba1ac5b3551e6af9e"
        },
        "date": 1742841056373,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cuda-train]",
            "value": 0.17218306580625672,
            "unit": "iter/sec",
            "range": "stddev: 0.3637772068949323",
            "extra": "mean: 5.807772066999997 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cuda-val]",
            "value": 0.39026110779863565,
            "unit": "iter/sec",
            "range": "stddev: 0.017209378356344316",
            "extra": "mean: 2.562387027599925 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_cm4.test.yaml-cuda]",
            "value": 0.148040046577813,
            "unit": "iter/sec",
            "range": "stddev: 0.09404192597159702",
            "extra": "mean: 6.754928974400036 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_cm4.test.yaml-cuda]",
            "value": 0.01015980460054561,
            "unit": "iter/sec",
            "range": "stddev: 0.5470538620160127",
            "extra": "mean: 98.42708982280006 sec\nrounds: 5"
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
          "id": "e7a3cfffeef70b46b06c94bbcca028da122d9162",
          "message": "uv script to prototype Zarr opening + update to Zarr cloning script. (#111)\n\n# Experiments\n\nSo far, it looks like we _should_ be using Zarr v2 + time_chunks=700, as\nthis is the fastest way to open this particular dataset!\n\n## Opening Remote OM4 data\n```\n# Zarr v2 (I think this uses consolidated=True by default!)\nuv run scripts/open_zarr_tuning.py                        \n3.11.11 (main, Jan 14 2025, 23:36:41) [Clang 19.1.6 ]\nzarr-version=2.18.4,xarray-version=2025.1.2\nELAPSED: 7.5452s. Config: Namespace(target=None, n_iters=8, zarr_concurrency=None, time_chunks=None)\n\n# Zarr v2 + time=10\nuv run scripts/open_zarr_tuning.py --time_chunks=10\n3.11.11 (main, Jan 14 2025, 23:36:41) [Clang 19.1.6 ]\nzarr-version=2.18.4,xarray-version=2025.1.2\nELAPSED: 7.1839s. Config: Namespace(target=None, n_iters=8, zarr_concurrency=None, time_chunks=10)\n\n# Zarr v2 + time=100\nuv run scripts/open_zarr_tuning.py --time_chunks=100\n3.11.11 (main, Jan 14 2025, 23:36:41) [Clang 19.1.6 ]\nzarr-version=2.18.4,xarray-version=2025.1.2\nELAPSED: 7.1952s. Config: Namespace(target=None, n_iters=8, zarr_concurrency=None, time_chunks=100)\n\n# Zarr v2 + time=700\nuv run scripts/open_zarr_tuning.py --time_chunks=700\n3.11.11 (main, Jan 14 2025, 23:36:41) [Clang 19.1.6 ]\nzarr-version=2.18.4,xarray-version=2025.1.2\nELAPSED: 7.0575s. Config: Namespace(target=None, n_iters=8, zarr_concurrency=None, time_chunks=700)\n\n\n\n# Zarr v3 (defaults)\nuv run scripts/open_zarr_tuning.py \n3.11.11 (main, Jan 14 2025, 23:36:41) [Clang 19.1.6 ]\nzarr-version=3.0.5,xarray-version=2025.1.2\nELAPSED: 11.2039s. Config: Namespace(target=None, n_iters=8, zarr_concurrency=10, time_chunks=None)\n\n# Zarr v3 + manually setting `consolidated=True`\n uv run scripts/open_zarr_tuning.py \n3.11.11 (main, Jan 14 2025, 23:36:41) [Clang 19.1.6 ]\nzarr-version=3.0.5,xarray-version=2025.1.2\nELAPSED: 11.2342s. Config: Namespace(target=None, n_iters=8, zarr_concurrency=10, time_chunks=None)\n\n# Zarr v3 + concurrency=50\nuv run scripts/open_zarr_tuning.py --zarr_concurrency=50\n3.11.11 (main, Jan 14 2025, 23:36:41) [Clang 19.1.6 ]\nzarr-version=3.0.5,xarray-version=2025.1.2\nELAPSED: 10.9889s. Config: Namespace(target=None, n_iters=8, zarr_concurrency=50, time_chunks=None)\n\n# Zarr v3 + concurrency=100\nuv run scripts/open_zarr_tuning.py --zarr_concurrency=100\n3.11.11 (main, Jan 14 2025, 23:36:41) [Clang 19.1.6 ]\nzarr-version=3.0.5,xarray-version=2025.1.2\nELAPSED: 11.0583s. Config: Namespace(target=None, n_iters=8, zarr_concurrency=100, time_chunks=None)\n\n# Zarr v3 + time=10\nuv run scripts/open_zarr_tuning.py --time_chunks=10   \n3.11.11 (main, Jan 14 2025, 23:36:41) [Clang 19.1.6 ]\nzarr-version=3.0.5,xarray-version=2025.1.2\nELAPSED: 10.9641s. Config: Namespace(target=None, n_iters=8, zarr_concurrency=10, time_chunks=10)\n\n# Zarr v3 + time=100\nuv run scripts/open_zarr_tuning.py --time_chunks=100\n3.11.11 (main, Jan 14 2025, 23:36:41) [Clang 19.1.6 ]\nzarr-version=3.0.5,xarray-version=2025.1.2\nELAPSED: 10.7012s. Config: Namespace(target=None, n_iters=8, zarr_concurrency=10, time_chunks=100)\n\n# Zarr v3 + time=700\nuv run scripts/open_zarr_tuning.py --time_chunks=700\n3.11.11 (main, Jan 14 2025, 23:36:41) [Clang 19.1.6 ]\nzarr-version=3.0.5,xarray-version=2025.1.2\nELAPSED: 10.7541s. Config: Namespace(target=None, n_iters=8, zarr_concurrency=10, time_chunks=700)\n\n# Zarr v3 + time=700 + concurrency=100\nuv run scripts/open_zarr_tuning.py --time_chunks=700 --zarr_concurrency=100\n3.11.11 (main, Jan 14 2025, 23:36:41) [Clang 19.1.6 ]\nzarr-version=3.0.5,xarray-version=2025.1.2\nELAPSED: 10.7544s. Config: Namespace(target=None, n_iters=8, zarr_concurrency=100, time_chunks=700)\n```\n\n**TBD: Opening locally cloned OM4 data...**",
          "timestamp": "2025-03-24T14:36:33-07:00",
          "tree_id": "56a3d2bdbe750aa8daa38384211e433ea10c0a3a",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/e7a3cfffeef70b46b06c94bbcca028da122d9162"
        },
        "date": 1742854461384,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cuda-train]",
            "value": 0.18765561408180703,
            "unit": "iter/sec",
            "range": "stddev: 0.3176951093230007",
            "extra": "mean: 5.328910647800058 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cuda-val]",
            "value": 0.41668978805379203,
            "unit": "iter/sec",
            "range": "stddev: 0.024392429426929547",
            "extra": "mean: 2.399866828200038 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_cm4.test.yaml-cuda]",
            "value": 0.16204823072863508,
            "unit": "iter/sec",
            "range": "stddev: 0.11049471356183548",
            "extra": "mean: 6.171002272000078 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_cm4.test.yaml-cuda]",
            "value": 0.011141400522572823,
            "unit": "iter/sec",
            "range": "stddev: 0.7458540900687095",
            "extra": "mean: 89.75532276879994 sec\nrounds: 5"
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
          "id": "06969612418c78fecfc8dc05ba9961002ba600da",
          "message": "March refactors - functionality (#142)\n\n* Ability to train on OM4 or CM4\n* Added metadata\n* Reduced checkpointing\n* Removed unnecessary zos check in wandblogger\n* Logger fix\n* Logging in eval fix\n* Train/val/inference time range extraction fix\n\nFollowing Draft PR\nhttps://github.com/suryadheeshjith/Ocean_Emulator/pull/134\n\nCloses Issues #116 , #74 , #80 , #101 , #69 , #59\n\n---------\n\nCo-authored-by: Alex Merose <alex@openathena.ai>",
          "timestamp": "2025-03-25T15:06:28-04:00",
          "tree_id": "a27cae05f25dc0e2e6be4ca3871e69b9b9e23994",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/06969612418c78fecfc8dc05ba9961002ba600da"
        },
        "date": 1742931915686,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cuda-train]",
            "value": 0.17571772056983503,
            "unit": "iter/sec",
            "range": "stddev: 0.32915506286354274",
            "extra": "mean: 5.690945664199944 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cuda-val]",
            "value": 0.3928828723607619,
            "unit": "iter/sec",
            "range": "stddev: 0.06008406996243891",
            "extra": "mean: 2.5452878462000172 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_cm4.test.yaml-cuda]",
            "value": 0.15337995394234802,
            "unit": "iter/sec",
            "range": "stddev: 0.08503130243647936",
            "extra": "mean: 6.519756814999937 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_cm4.test.yaml-cuda]",
            "value": 0.010034233363195574,
            "unit": "iter/sec",
            "range": "stddev: 1.0515166233986755",
            "extra": "mean: 99.65883429300001 sec\nrounds: 5"
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
          "id": "8ff6f801a7256089e93d1070d14fa90f0480978a",
          "message": "Jaxtyping minimum version is 0.3.0 (#145)",
          "timestamp": "2025-03-25T12:22:10-07:00",
          "tree_id": "05230b73a3ca241651392c77b448e702b9de684b",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/8ff6f801a7256089e93d1070d14fa90f0480978a"
        },
        "date": 1742932711021,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cuda-train]",
            "value": 0.18355986679274755,
            "unit": "iter/sec",
            "range": "stddev: 0.34770127002845364",
            "extra": "mean: 5.447813933799989 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cuda-val]",
            "value": 0.41264129645426734,
            "unit": "iter/sec",
            "range": "stddev: 0.027961544463990285",
            "extra": "mean: 2.4234123161999834 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_cm4.test.yaml-cuda]",
            "value": 0.16187829435746734,
            "unit": "iter/sec",
            "range": "stddev: 0.07574942748941199",
            "extra": "mean: 6.177480458199989 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_cm4.test.yaml-cuda]",
            "value": 0.010751534624054914,
            "unit": "iter/sec",
            "range": "stddev: 0.6891541074005118",
            "extra": "mean: 93.00997810700001 sec\nrounds: 5"
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
          "id": "37788f5c24aed9807dfdf0686896146d4265f6e4",
          "message": "Omitting filtering by guaranteeing inputs are unique. (#143)\n\nFixing #140 by making omitting slow step (`itertools.product`), instead\nguaranteeing that all inputs are unique.",
          "timestamp": "2025-03-25T13:35:00-07:00",
          "tree_id": "a326a105eab36c3d8e73baf0d530fc9affde4aa5",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/37788f5c24aed9807dfdf0686896146d4265f6e4"
        },
        "date": 1742937069163,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cuda-train]",
            "value": 0.18536194094971292,
            "unit": "iter/sec",
            "range": "stddev: 0.4009785218001505",
            "extra": "mean: 5.39485071680001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cuda-val]",
            "value": 0.41540435553890254,
            "unit": "iter/sec",
            "range": "stddev: 0.023581894720292457",
            "extra": "mean: 2.407293006600048 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_cm4.test.yaml-cuda]",
            "value": 0.1627187741038636,
            "unit": "iter/sec",
            "range": "stddev: 0.0795927567533129",
            "extra": "mean: 6.145572356399998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_cm4.test.yaml-cuda]",
            "value": 0.010851301581809048,
            "unit": "iter/sec",
            "range": "stddev: 0.3749149058532742",
            "extra": "mean: 92.1548435882 sec\nrounds: 5"
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
          "id": "327bc7c42465a7ccd6821b255cd1bd0b1bf1cd45",
          "message": "Fixes to the lazy data loader given more realistic data. (#148)\n\nExtracted some pre-conditions from the `validate_data` function into\nsmaller parts and applying some of them to all input from the current\ndata loader.\n\nIn addition, I've caught a minor issue in this loader's collate_fn as\nwell as my new data loader test.\n\nThis fixes were found during #128.\n\n---------\n\nCo-authored-by: Jesse Rusak <jesse@openathena.ai>",
          "timestamp": "2025-03-26T11:08:24-07:00",
          "tree_id": "7be8460bf295353bf8588e337751cb3c8f65bde1",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/327bc7c42465a7ccd6821b255cd1bd0b1bf1cd45"
        },
        "date": 1743014901244,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cuda-train]",
            "value": 0.16948990979419995,
            "unit": "iter/sec",
            "range": "stddev: 0.388882645935231",
            "extra": "mean: 5.900056240599997 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cuda-val]",
            "value": 0.4149490810409914,
            "unit": "iter/sec",
            "range": "stddev: 0.04121987606572914",
            "extra": "mean: 2.4099342441999854 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_cm4.test.yaml-cuda]",
            "value": 0.15975718808138695,
            "unit": "iter/sec",
            "range": "stddev: 0.26551690970996733",
            "extra": "mean: 6.259499256400022 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_cm4.test.yaml-cuda]",
            "value": 0.00954275825338341,
            "unit": "iter/sec",
            "range": "stddev: 2.255115900672051",
            "extra": "mean: 104.79150508140006 sec\nrounds: 5"
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
          "id": "53dae35e99b6fd0a2f1e5dd6173b0806544dd52a",
          "message": "Fix parsing of loader_version config parameter. (#149)",
          "timestamp": "2025-03-27T18:26:22-07:00",
          "tree_id": "076bc5392b6347a05971318063e112b0a7b62512",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/53dae35e99b6fd0a2f1e5dd6173b0806544dd52a"
        },
        "date": 1743127551913,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cuda-train]",
            "value": 0.16325439082459858,
            "unit": "iter/sec",
            "range": "stddev: 0.271401246382306",
            "extra": "mean: 6.125409521599977 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_cm4.test.yaml-cuda-val]",
            "value": 0.36446749154010255,
            "unit": "iter/sec",
            "range": "stddev: 0.039695582820277665",
            "extra": "mean: 2.743728928400105 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_cm4.test.yaml-cuda]",
            "value": 0.1400385365422095,
            "unit": "iter/sec",
            "range": "stddev: 0.11076280488447969",
            "extra": "mean: 7.140891533799959 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_cm4.test.yaml-cuda]",
            "value": 0.010468420082980573,
            "unit": "iter/sec",
            "range": "stddev: 3.9415473300042096",
            "extra": "mean: 95.52539849119998 sec\nrounds: 5"
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
          "id": "35bb5cb6d97abb0e8bd9df99bc4af91c01d511f3",
          "message": "Adding remote data to tests + caching test data (#128)\n\nThis PR addressed part of #146. It adds ~30 days worth of OM4 to the\ntest data sources. To make this more performant, we add a local data\ncaching mechanism.",
          "timestamp": "2025-03-27T19:50:56-07:00",
          "tree_id": "8ff73fc05cbb1d2c6577ae342ab1a27d86f553e2",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/35bb5cb6d97abb0e8bd9df99bc4af91c01d511f3"
        },
        "date": 1743133810569,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-cuda-train]",
            "value": 0.18280927747655967,
            "unit": "iter/sec",
            "range": "stddev: 0.26232602664626126",
            "extra": "mean: 5.470181895600035 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-cuda-val]",
            "value": 0.3988792253552795,
            "unit": "iter/sec",
            "range": "stddev: 0.039388244808652946",
            "extra": "mean: 2.507024523799919 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-cuda]",
            "value": 0.1604045193837981,
            "unit": "iter/sec",
            "range": "stddev: 0.05409982026471947",
            "extra": "mean: 6.234238310999899 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-cuda]",
            "value": 0.01059191873946944,
            "unit": "iter/sec",
            "range": "stddev: 0.6166140697946135",
            "extra": "mean: 94.41160044720009 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-cuda-train]",
            "value": 0.6075289821813089,
            "unit": "iter/sec",
            "range": "stddev: 0.016233660541422182",
            "extra": "mean: 1.6460120082000684 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-cuda-val]",
            "value": 0.8853941751183447,
            "unit": "iter/sec",
            "range": "stddev: 0.05973942124164683",
            "extra": "mean: 1.1294404549999855 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-cuda]",
            "value": 0.3601557582751241,
            "unit": "iter/sec",
            "range": "stddev: 0.15071982701821962",
            "extra": "mean: 2.776576458999989 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-cuda]",
            "value": 0.013482744247303122,
            "unit": "iter/sec",
            "range": "stddev: 0.19963597071093725",
            "extra": "mean: 74.16887702220001 sec\nrounds: 5"
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
          "id": "2ad42a381a36e40aca79d4469f684f623c13e041",
          "message": "Eval checkpoint load fix (#153)\n\nNeeded to remove the \"module.\" component in the keys of the checkpoint",
          "timestamp": "2025-03-27T23:47:47-04:00",
          "tree_id": "865556090eff2ac4bc7c3aa24b17056c9b7db012",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/2ad42a381a36e40aca79d4469f684f623c13e041"
        },
        "date": 1743137134316,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-cuda-train]",
            "value": 0.18076559193381483,
            "unit": "iter/sec",
            "range": "stddev: 0.2181746082051167",
            "extra": "mean: 5.532026251799834 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-cuda-val]",
            "value": 0.38610828047923185,
            "unit": "iter/sec",
            "range": "stddev: 0.14446431651665445",
            "extra": "mean: 2.5899470448000104 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-cuda]",
            "value": 0.15843093348523787,
            "unit": "iter/sec",
            "range": "stddev: 0.08747520631603871",
            "extra": "mean: 6.311898680399918 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-cuda]",
            "value": 0.010503081634496005,
            "unit": "iter/sec",
            "range": "stddev: 0.3570704363792467",
            "extra": "mean: 95.21015210580008 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-cuda-train]",
            "value": 0.6150284527482472,
            "unit": "iter/sec",
            "range": "stddev: 0.018123646550300625",
            "extra": "mean: 1.6259410366000338 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-cuda-val]",
            "value": 0.9229188818796475,
            "unit": "iter/sec",
            "range": "stddev: 0.004840987278236338",
            "extra": "mean: 1.083518843999991 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-cuda]",
            "value": 0.3593796738651814,
            "unit": "iter/sec",
            "range": "stddev: 0.14934936655775716",
            "extra": "mean: 2.7825725067999882 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-cuda]",
            "value": 0.013315125237101108,
            "unit": "iter/sec",
            "range": "stddev: 0.1510298951285877",
            "extra": "mean: 75.10256059880021 sec\nrounds: 5"
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
          "id": "0758666ad441442670f5e33a10e8ef92af72bbab",
          "message": "Compact data during clone. (#144)\n\nAdded an option to data cloning script to compact variables. Here is\nwhat a sample output looks like:\n\n```\n<xarray.Dataset> Size: 74GB\nDimensions:         (y: 180, x: 360, lev: 19, time: 3504, y_b: 181, x_b: 361)\nCoordinates:\n    areacello       (y, x) float64 518kB dask.array<chunksize=(90, 360), meta=np.ndarray>\n    dz              (lev) int64 152B dask.array<chunksize=(19,), meta=np.ndarray>\n    lat             (y, x) float64 518kB dask.array<chunksize=(90, 360), meta=np.ndarray>\n    lat_b           (y_b, x_b) float64 523kB dask.array<chunksize=(91, 361), meta=np.ndarray>\n  * lev             (lev) float64 152B 2.5 10.0 22.5 40.0 ... 4e+03 5e+03 6e+03\n    lon             (y, x) float64 518kB dask.array<chunksize=(90, 360), meta=np.ndarray>\n    lon_b           (y_b, x_b) float64 523kB dask.array<chunksize=(91, 361), meta=np.ndarray>\n    ocean_fraction  (lev, y, x) float64 10MB dask.array<chunksize=(19, 180, 360), meta=np.ndarray>\n  * time            (time) object 28kB 1975-01-03 12:00:00 ... 2022-12-29 12:...\n    wetmask         (lev, y, x) bool 1MB dask.array<chunksize=(19, 90, 360), meta=np.ndarray>\n  * x               (x) float64 3kB 0.5 1.5 2.5 3.5 ... 356.5 357.5 358.5 359.5\n  * y               (y) float64 1kB -89.24 -88.25 -87.25 ... 87.25 88.25 89.24\nDimensions without coordinates: y_b, x_b\nData variables:\n    hfds            (time, y, x) float32 908MB dask.array<chunksize=(50, 180, 360), meta=np.ndarray>\n    hfds_anomalies  (time, y, x) float32 908MB dask.array<chunksize=(50, 180, 360), meta=np.ndarray>\n    tauuo           (time, y, x) float32 908MB dask.array<chunksize=(50, 180, 360), meta=np.ndarray>\n    tauvo           (time, y, x) float32 908MB dask.array<chunksize=(50, 180, 360), meta=np.ndarray>\n    zos             (time, y, x) float32 908MB dask.array<chunksize=(50, 180, 360), meta=np.ndarray>\n    so              (lev, time, y, x) float32 17GB dask.array<chunksize=(19, 50, 180, 360), meta=np.ndarray>\n    thetao          (lev, time, y, x) float32 17GB dask.array<chunksize=(19, 50, 180, 360), meta=np.ndarray>\n    uo              (lev, time, y, x) float32 17GB dask.array<chunksize=(19, 50, 180, 360), meta=np.ndarray>\n    vo              (lev, time, y, x) float32 17GB dask.array<chunksize=(19, 50, 180, 360), meta=np.ndarray\n```\n\nThis script defaults to reading all levels as one chunk. This makes the\nrecommended chunking heuristic ~50-60 time chunks.",
          "timestamp": "2025-03-28T16:06:31-07:00",
          "tree_id": "ac7374803b7284f28d81ea158e8bc0ca2ebc066e",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/0758666ad441442670f5e33a10e8ef92af72bbab"
        },
        "date": 1743207056097,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-cuda-train]",
            "value": 0.1643853726301984,
            "unit": "iter/sec",
            "range": "stddev: 0.29855730009647574",
            "extra": "mean: 6.083266314999946 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-cuda-val]",
            "value": 0.3427130917398997,
            "unit": "iter/sec",
            "range": "stddev: 0.23961032782027541",
            "extra": "mean: 2.9178926165999655 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-cuda]",
            "value": 0.14109194226884647,
            "unit": "iter/sec",
            "range": "stddev: 0.08436979170828567",
            "extra": "mean: 7.087576965199969 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-cuda]",
            "value": 0.009451879845408387,
            "unit": "iter/sec",
            "range": "stddev: 1.3636957031225512",
            "extra": "mean: 105.79905969559995 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-cuda-train]",
            "value": 0.5601194224884402,
            "unit": "iter/sec",
            "range": "stddev: 0.014841616833248606",
            "extra": "mean: 1.7853335554002114 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-cuda-val]",
            "value": 0.8098071652696187,
            "unit": "iter/sec",
            "range": "stddev: 0.021835046381481565",
            "extra": "mean: 1.2348618818000432 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-cuda]",
            "value": 0.3128598993824905,
            "unit": "iter/sec",
            "range": "stddev: 0.18470304808144886",
            "extra": "mean: 3.196318869800052 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-cuda]",
            "value": 0.011868376749583647,
            "unit": "iter/sec",
            "range": "stddev: 0.213538404103135",
            "extra": "mean: 84.25752072920004 sec\nrounds: 5"
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
          "id": "a773fdcc8ee50509195476702df0d0cd9a7e40e7",
          "message": "Fixed anomalies statistics for anomalies use (#156)\n\nNeed to compute statistics when you compute anomalies before training",
          "timestamp": "2025-03-31T12:20:49-04:00",
          "tree_id": "9229b9d38ddf3fb9bf752c7e6e5c4abc50f59abc",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/a773fdcc8ee50509195476702df0d0cd9a7e40e7"
        },
        "date": 1743441489718,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-cuda-train]",
            "value": 0.18393634568741343,
            "unit": "iter/sec",
            "range": "stddev: 0.377816839153439",
            "extra": "mean: 5.436663408000004 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-cuda-val]",
            "value": 0.40472848183673604,
            "unit": "iter/sec",
            "range": "stddev: 0.035738735535776556",
            "extra": "mean: 2.4707922592000613 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-cuda]",
            "value": 0.16047418519551168,
            "unit": "iter/sec",
            "range": "stddev: 0.08253076355662128",
            "extra": "mean: 6.231531873999939 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-cuda]",
            "value": 0.010708562401713272,
            "unit": "iter/sec",
            "range": "stddev: 0.6844408452845111",
            "extra": "mean: 93.38321639139994 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-cuda-train]",
            "value": 0.7517593907274467,
            "unit": "iter/sec",
            "range": "stddev: 0.10720141882926944",
            "extra": "mean: 1.3302128478000668 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-cuda-val]",
            "value": 1.3433228617322615,
            "unit": "iter/sec",
            "range": "stddev: 0.0067342571639314635",
            "extra": "mean: 744.4226763999723 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-cuda]",
            "value": 0.36411352972993577,
            "unit": "iter/sec",
            "range": "stddev: 0.14329487267451257",
            "extra": "mean: 2.746396160400036 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-cuda]",
            "value": 0.01361312948794102,
            "unit": "iter/sec",
            "range": "stddev: 0.207279111604258",
            "extra": "mean: 73.45849467499993 sec\nrounds: 5"
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
          "id": "00206f926d2761313ac4f35088f132cf16163ea6",
          "message": "Torch arrays should be floats, not doubles. (#170)",
          "timestamp": "2025-04-02T17:49:17-07:00",
          "tree_id": "c45653554e4f644b81a3baa7be8cefa01458c26d",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/00206f926d2761313ac4f35088f132cf16163ea6"
        },
        "date": 1743662127312,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist0-cuda-train]",
            "value": 0.17413263970931167,
            "unit": "iter/sec",
            "range": "stddev: 0.5246337575012436",
            "extra": "mean: 5.742748755600042 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist0-cuda-val]",
            "value": 0.3796718399987546,
            "unit": "iter/sec",
            "range": "stddev: 0.03967574064874967",
            "extra": "mean: 2.63385348779957 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-eager-hist0-cuda]",
            "value": 0.12893040420639168,
            "unit": "iter/sec",
            "range": "stddev: 0.15748596624135613",
            "extra": "mean: 7.756122430200412 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-eager-hist0-cuda]",
            "value": 0.009930546644874373,
            "unit": "iter/sec",
            "range": "stddev: 1.014206275358031",
            "extra": "mean: 100.69939105680024 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist1-cuda-train]",
            "value": 0.18136821522344135,
            "unit": "iter/sec",
            "range": "stddev: 0.28038996314038117",
            "extra": "mean: 5.513645258999896 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist1-cuda-val]",
            "value": 0.5840922217527749,
            "unit": "iter/sec",
            "range": "stddev: 0.020401451806453544",
            "extra": "mean: 1.7120584091997444 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-eager-hist1-cuda]",
            "value": 0.26655128461918426,
            "unit": "iter/sec",
            "range": "stddev: 0.10662052805883611",
            "extra": "mean: 3.751623262400244 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-eager-hist1-cuda]",
            "value": 0.010925507943653337,
            "unit": "iter/sec",
            "range": "stddev: 1.3926299468096957",
            "extra": "mean: 91.52892525980023 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist1-cuda-train]",
            "value": 0.13697783945312952,
            "unit": "iter/sec",
            "range": "stddev: 0.4133170846924649",
            "extra": "mean: 7.300450963399635 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist1-cuda-val]",
            "value": 0.43030626985670806,
            "unit": "iter/sec",
            "range": "stddev: 0.18500137902624691",
            "extra": "mean: 2.323926166200181 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-lazy-hist1-cuda]",
            "value": 0.264820312850419,
            "unit": "iter/sec",
            "range": "stddev: 0.10372836923376672",
            "extra": "mean: 3.7761453766004704 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-lazy-hist1-cuda]",
            "value": 0.01021175007405529,
            "unit": "iter/sec",
            "range": "stddev: 1.044857123665446",
            "extra": "mean: 97.9264075940002 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist1-cuda-train]",
            "value": 0.41418615064607933,
            "unit": "iter/sec",
            "range": "stddev: 0.02382380256760998",
            "extra": "mean: 2.414373340200109 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist1-cuda-val]",
            "value": 0.5845196784963033,
            "unit": "iter/sec",
            "range": "stddev: 0.013094548598177057",
            "extra": "mean: 1.7108063881998532 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist1-cuda]",
            "value": 0.577391015387887,
            "unit": "iter/sec",
            "range": "stddev: 0.11978964494503384",
            "extra": "mean: 1.7319285776004107 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-lazy-hist1-cuda]",
            "value": 0.012359675219251702,
            "unit": "iter/sec",
            "range": "stddev: 0.6190743138796051",
            "extra": "mean: 80.9082748746001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist2-cuda-train]",
            "value": 0.3406866195268654,
            "unit": "iter/sec",
            "range": "stddev: 0.033348406985518685",
            "extra": "mean: 2.935248825999588 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist2-cuda-val]",
            "value": 0.9927724612126039,
            "unit": "iter/sec",
            "range": "stddev: 0.009546461878962489",
            "extra": "mean: 1.0072801564001566 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist2-cuda]",
            "value": 0.8104007391005145,
            "unit": "iter/sec",
            "range": "stddev: 0.16158525359061662",
            "extra": "mean: 1.233957413600001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-lazy-hist2-cuda]",
            "value": 0.012729765543076543,
            "unit": "iter/sec",
            "range": "stddev: 1.0345866028486304",
            "extra": "mean: 78.55604226299984 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist2-cuda-train]",
            "value": 0.5520585746806081,
            "unit": "iter/sec",
            "range": "stddev: 0.011368583109339704",
            "extra": "mean: 1.811401988599755 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist2-cuda-val]",
            "value": 1.2975778352519065,
            "unit": "iter/sec",
            "range": "stddev: 0.0044983647597478995",
            "extra": "mean: 770.6666782003595 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist2-cuda]",
            "value": 0.8643690838403857,
            "unit": "iter/sec",
            "range": "stddev: 0.14319323233006115",
            "extra": "mean: 1.156913196798996 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-eager-hist2-cuda]",
            "value": 0.013301156759795011,
            "unit": "iter/sec",
            "range": "stddev: 0.22187258970228066",
            "extra": "mean: 75.18143106339957 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist2-cuda-train]",
            "value": 0.12089778915259446,
            "unit": "iter/sec",
            "range": "stddev: 0.6701413073117947",
            "extra": "mean: 8.271449850400677 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist2-cuda-val]",
            "value": 0.4465484550164526,
            "unit": "iter/sec",
            "range": "stddev: 0.018521877139172232",
            "extra": "mean: 2.2393986335999214 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-lazy-hist2-cuda]",
            "value": 0.341464736289197,
            "unit": "iter/sec",
            "range": "stddev: 0.07798679828862481",
            "extra": "mean: 2.9285600934003013 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-lazy-hist2-cuda]",
            "value": 0.009980985376450362,
            "unit": "iter/sec",
            "range": "stddev: 1.0527228578003598",
            "extra": "mean: 100.19050848019978 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist2-cuda-train]",
            "value": 0.16061099693016628,
            "unit": "iter/sec",
            "range": "stddev: 0.2813848304357994",
            "extra": "mean: 6.226223727599427 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist2-cuda-val]",
            "value": 0.5404417043863929,
            "unit": "iter/sec",
            "range": "stddev: 0.03568296445078078",
            "extra": "mean: 1.8503383285999007 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-eager-hist2-cuda]",
            "value": 0.3424492427739897,
            "unit": "iter/sec",
            "range": "stddev: 0.018857603117439656",
            "extra": "mean: 2.9201407831991673 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-eager-hist2-cuda]",
            "value": 0.010703656332787897,
            "unit": "iter/sec",
            "range": "stddev: 1.600495831734305",
            "extra": "mean: 93.42601900779991 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist0-cuda-train]",
            "value": 0.46194992145613284,
            "unit": "iter/sec",
            "range": "stddev: 0.026704523406027088",
            "extra": "mean: 2.164736811401235 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist0-cuda-val]",
            "value": 0.9780734259187169,
            "unit": "iter/sec",
            "range": "stddev: 0.019092125582249916",
            "extra": "mean: 1.0224181267993118 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist0-cuda]",
            "value": 0.2866263725480831,
            "unit": "iter/sec",
            "range": "stddev: 0.13791346428066373",
            "extra": "mean: 3.4888624906008774 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-lazy-hist0-cuda]",
            "value": 0.011981624376742387,
            "unit": "iter/sec",
            "range": "stddev: 0.28878566273167716",
            "extra": "mean: 83.46113753499958 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist1-cuda-train]",
            "value": 0.6666757672356268,
            "unit": "iter/sec",
            "range": "stddev: 0.005477135231708168",
            "extra": "mean: 1.4999795239993545 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist1-cuda-val]",
            "value": 0.9315123818059408,
            "unit": "iter/sec",
            "range": "stddev: 0.007493846902717413",
            "extra": "mean: 1.0735230357982801 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist1-cuda]",
            "value": 0.5395250093549211,
            "unit": "iter/sec",
            "range": "stddev: 0.16745123355287014",
            "extra": "mean: 1.8534821976012608 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-eager-hist1-cuda]",
            "value": 0.01324574116039583,
            "unit": "iter/sec",
            "range": "stddev: 0.6173261536499176",
            "extra": "mean: 75.4959641661997 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist0-cuda-train]",
            "value": 0.6821992314551714,
            "unit": "iter/sec",
            "range": "stddev: 0.13350253104685958",
            "extra": "mean: 1.4658474443996965 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist0-cuda-val]",
            "value": 1.2121653560980368,
            "unit": "iter/sec",
            "range": "stddev: 0.008061039872470342",
            "extra": "mean: 824.9699555999541 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist0-cuda]",
            "value": 0.2905477945280062,
            "unit": "iter/sec",
            "range": "stddev: 0.1294641332450152",
            "extra": "mean: 3.441774533599528 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-eager-hist0-cuda]",
            "value": 0.012567194951000041,
            "unit": "iter/sec",
            "range": "stddev: 0.2616636288061462",
            "extra": "mean: 79.57225171560057 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist0-cuda-train]",
            "value": 0.14111382114235652,
            "unit": "iter/sec",
            "range": "stddev: 0.43225032831355115",
            "extra": "mean: 7.086478077800711 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist0-cuda-val]",
            "value": 0.321320196559544,
            "unit": "iter/sec",
            "range": "stddev: 0.008171209489943073",
            "extra": "mean: 3.112160426600167 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-lazy-hist0-cuda]",
            "value": 0.12918008526675512,
            "unit": "iter/sec",
            "range": "stddev: 0.093691118840682",
            "extra": "mean: 7.741131289199984 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-lazy-hist0-cuda]",
            "value": 0.009416207659513816,
            "unit": "iter/sec",
            "range": "stddev: 1.1789118845064819",
            "extra": "mean: 106.19986688479985 sec\nrounds: 5"
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
          "id": "648cd211c95f934b7a03d4ebc68440bfb7ed3499",
          "message": "Turn off dask by default (#168)\n\nAccording to [results\nhere](https://github.com/suryadheeshjith/Ocean_Emulator/issues/105#issuecomment-2773940880),\nturning off dask is 20x faster for the eager data loader and 10x faster\nfor the lazy data loader. This disables dask by default but you can\ntoggle it back on.",
          "timestamp": "2025-04-03T11:56:34-04:00",
          "tree_id": "738bcd88d7610dcd35c276699c47fe9c082c881e",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/648cd211c95f934b7a03d4ebc68440bfb7ed3499"
        },
        "date": 1743709291198,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist0-cuda-train]",
            "value": 2.6848317860471522,
            "unit": "iter/sec",
            "range": "stddev: 0.01442662316110582",
            "extra": "mean: 372.4628132000362 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist0-cuda-val]",
            "value": 3.963376128011508,
            "unit": "iter/sec",
            "range": "stddev: 0.006010305992428696",
            "extra": "mean: 252.31014360015251 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-eager-hist0-cuda]",
            "value": 0.3777031693942022,
            "unit": "iter/sec",
            "range": "stddev: 0.0165590088033409",
            "extra": "mean: 2.647581701800118 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-eager-hist0-cuda]",
            "value": 0.014589835220523146,
            "unit": "iter/sec",
            "range": "stddev: 0.3615070200147524",
            "extra": "mean: 68.54087005679995 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist1-cuda-train]",
            "value": 1.3851267495874746,
            "unit": "iter/sec",
            "range": "stddev: 0.017585489933673176",
            "extra": "mean: 721.9555901998319 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist1-cuda-val]",
            "value": 2.152927700845969,
            "unit": "iter/sec",
            "range": "stddev: 0.00813546165749744",
            "extra": "mean: 464.48378159984713 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-eager-hist1-cuda]",
            "value": 0.45863752906976496,
            "unit": "iter/sec",
            "range": "stddev: 0.01671743989826334",
            "extra": "mean: 2.180371070000001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-eager-hist1-cuda]",
            "value": 0.014403390991189152,
            "unit": "iter/sec",
            "range": "stddev: 0.1000413555073187",
            "extra": "mean: 69.42809513479988 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist1-cuda-train]",
            "value": 1.333845569513277,
            "unit": "iter/sec",
            "range": "stddev: 0.011384606817363025",
            "extra": "mean: 749.7119778003253 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist1-cuda-val]",
            "value": 1.9130998630022098,
            "unit": "iter/sec",
            "range": "stddev: 0.01805633720150815",
            "extra": "mean: 522.7118664002774 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-lazy-hist1-cuda]",
            "value": 0.454141888292495,
            "unit": "iter/sec",
            "range": "stddev: 0.02777683984363791",
            "extra": "mean: 2.2019549963995813 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-lazy-hist1-cuda]",
            "value": 0.014160833074377994,
            "unit": "iter/sec",
            "range": "stddev: 1.0686059591695372",
            "extra": "mean: 70.6173143025997 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist1-cuda-train]",
            "value": 1.8317413675823158,
            "unit": "iter/sec",
            "range": "stddev: 0.019794721451138912",
            "extra": "mean: 545.9285998000269 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist1-cuda-val]",
            "value": 2.221715062933647,
            "unit": "iter/sec",
            "range": "stddev: 0.010010727924564774",
            "extra": "mean: 450.1027231995977 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist1-cuda]",
            "value": 0.5297360271779478,
            "unit": "iter/sec",
            "range": "stddev: 0.02562575674675363",
            "extra": "mean: 1.8877326605994313 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-lazy-hist1-cuda]",
            "value": 0.014133312038125566,
            "unit": "iter/sec",
            "range": "stddev: 0.7579340531106379",
            "extra": "mean: 70.75482359000016 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist2-cuda-train]",
            "value": 1.5429005873580897,
            "unit": "iter/sec",
            "range": "stddev: 0.005384231256791398",
            "extra": "mean: 648.1298978000268 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist2-cuda-val]",
            "value": 2.275770764415048,
            "unit": "iter/sec",
            "range": "stddev: 0.010336688279188024",
            "extra": "mean: 439.41156799992314 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist2-cuda]",
            "value": 0.6296861293696048,
            "unit": "iter/sec",
            "range": "stddev: 0.007871334288412",
            "extra": "mean: 1.588092786800189 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-lazy-hist2-cuda]",
            "value": 0.014282009956711271,
            "unit": "iter/sec",
            "range": "stddev: 0.3957511239697317",
            "extra": "mean: 70.01815591999984 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist2-cuda-train]",
            "value": 1.5159755038848632,
            "unit": "iter/sec",
            "range": "stddev: 0.01378454490503976",
            "extra": "mean: 659.6412655992026 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist2-cuda-val]",
            "value": 2.3057926219071385,
            "unit": "iter/sec",
            "range": "stddev: 0.008315942673734271",
            "extra": "mean: 433.69034600036684 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist2-cuda]",
            "value": 0.6263983245616993,
            "unit": "iter/sec",
            "range": "stddev: 0.004298838457392428",
            "extra": "mean: 1.5964282801996887 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-eager-hist2-cuda]",
            "value": 0.014320257493577689,
            "unit": "iter/sec",
            "range": "stddev: 0.23580242619836553",
            "extra": "mean: 69.83114657320075 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist2-cuda-train]",
            "value": 1.098359733803807,
            "unit": "iter/sec",
            "range": "stddev: 0.017611359073640406",
            "extra": "mean: 910.4485253996245 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist2-cuda-val]",
            "value": 1.6131606284814037,
            "unit": "iter/sec",
            "range": "stddev: 0.004471190353792699",
            "extra": "mean: 619.9010701999214 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-lazy-hist2-cuda]",
            "value": 0.5186590651515599,
            "unit": "iter/sec",
            "range": "stddev: 0.026296768637114255",
            "extra": "mean: 1.928048822799974 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-lazy-hist2-cuda]",
            "value": 0.013942044595413254,
            "unit": "iter/sec",
            "range": "stddev: 0.1372267590449953",
            "extra": "mean: 71.72549141960044 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist2-cuda-train]",
            "value": 1.0572585331407107,
            "unit": "iter/sec",
            "range": "stddev: 0.010030119193207333",
            "extra": "mean: 945.8424487995217 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist2-cuda-val]",
            "value": 1.7416322989699315,
            "unit": "iter/sec",
            "range": "stddev: 0.00708886649180496",
            "extra": "mean: 574.1740093999397 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-eager-hist2-cuda]",
            "value": 0.5288164912987564,
            "unit": "iter/sec",
            "range": "stddev: 0.01112416778774798",
            "extra": "mean: 1.8910151563995896 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-eager-hist2-cuda]",
            "value": 0.013829228972325988,
            "unit": "iter/sec",
            "range": "stddev: 1.0386784637465938",
            "extra": "mean: 72.31061124240004 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist0-cuda-train]",
            "value": 2.169866481175926,
            "unit": "iter/sec",
            "range": "stddev: 0.011160155802174646",
            "extra": "mean: 460.85784940005396 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist0-cuda-val]",
            "value": 2.731909086442494,
            "unit": "iter/sec",
            "range": "stddev: 0.018081311643700967",
            "extra": "mean: 366.0443918000965 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist0-cuda]",
            "value": 0.44161512803800673,
            "unit": "iter/sec",
            "range": "stddev: 0.006273664445216681",
            "extra": "mean: 2.2644151807995514 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-lazy-hist0-cuda]",
            "value": 0.014353741383920279,
            "unit": "iter/sec",
            "range": "stddev: 0.13884986109154027",
            "extra": "mean: 69.6682469924006 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist1-cuda-train]",
            "value": 1.8426208163411573,
            "unit": "iter/sec",
            "range": "stddev: 0.013530695607010852",
            "extra": "mean: 542.7052549996006 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist1-cuda-val]",
            "value": 2.2312040053438427,
            "unit": "iter/sec",
            "range": "stddev: 0.018316173579638474",
            "extra": "mean: 448.18851060008456 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist1-cuda]",
            "value": 0.5327913594923396,
            "unit": "iter/sec",
            "range": "stddev: 0.013514372178847047",
            "extra": "mean: 1.8769073150000621 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-eager-hist1-cuda]",
            "value": 0.014340045613761008,
            "unit": "iter/sec",
            "range": "stddev: 0.3127707779909071",
            "extra": "mean: 69.73478515580027 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist0-cuda-train]",
            "value": 2.235058700752479,
            "unit": "iter/sec",
            "range": "stddev: 0.018867648924759593",
            "extra": "mean: 447.41554200045357 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist0-cuda-val]",
            "value": 2.8860715236974506,
            "unit": "iter/sec",
            "range": "stddev: 0.006689654213858681",
            "extra": "mean: 346.49175939994166 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist0-cuda]",
            "value": 0.4414074978459466,
            "unit": "iter/sec",
            "range": "stddev: 0.005829856641247325",
            "extra": "mean: 2.2654803211997203 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-eager-hist0-cuda]",
            "value": 0.014075317573736163,
            "unit": "iter/sec",
            "range": "stddev: 0.5745316937734435",
            "extra": "mean: 71.04635435479977 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist0-cuda-train]",
            "value": 1.7502482760431544,
            "unit": "iter/sec",
            "range": "stddev: 0.016500122398773832",
            "extra": "mean: 571.3475132000895 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist0-cuda-val]",
            "value": 2.308074476532776,
            "unit": "iter/sec",
            "range": "stddev: 0.011348476654509082",
            "extra": "mean: 433.261582400155 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-lazy-hist0-cuda]",
            "value": 0.35702483613133207,
            "unit": "iter/sec",
            "range": "stddev: 0.0230684973050089",
            "extra": "mean: 2.8009255905999453 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-lazy-hist0-cuda]",
            "value": 0.014151217822142782,
            "unit": "iter/sec",
            "range": "stddev: 1.1270972512796247",
            "extra": "mean: 70.66529627119962 sec\nrounds: 5"
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
          "id": "a2e57c158589e0447799a389b24b96a2d207fa61",
          "message": "Datasets test for TrainData + Test fix (#172)\n\n- Just a test for traindata ensuring underlying data is not manipulated\nby TrainData. Added a test just in case (even with #169 )\n- Removed lat/lon coords in tests data statistics",
          "timestamp": "2025-04-03T15:49:40-04:00",
          "tree_id": "df76883e78482679bd1224e1c34d6eae0c7291e5",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/a2e57c158589e0447799a389b24b96a2d207fa61"
        },
        "date": 1743723098448,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist0-cuda-train]",
            "value": 2.7809761537315487,
            "unit": "iter/sec",
            "range": "stddev: 0.010669205988972127",
            "extra": "mean: 359.58596720010974 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist0-cuda-val]",
            "value": 4.1112880821430355,
            "unit": "iter/sec",
            "range": "stddev: 0.00228324664545204",
            "extra": "mean: 243.23277280018374 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-eager-hist0-cuda]",
            "value": 0.3792109210042899,
            "unit": "iter/sec",
            "range": "stddev: 0.010088283532883806",
            "extra": "mean: 2.6370548541999597 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-eager-hist0-cuda]",
            "value": 0.014765769696773455,
            "unit": "iter/sec",
            "range": "stddev: 0.109890191921478",
            "extra": "mean: 67.72420405680005 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist1-cuda-train]",
            "value": 1.323907880970919,
            "unit": "iter/sec",
            "range": "stddev: 0.009400874512017172",
            "extra": "mean: 755.3395627999635 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist1-cuda-val]",
            "value": 2.1586568450959582,
            "unit": "iter/sec",
            "range": "stddev: 0.013579946293315225",
            "extra": "mean: 463.2510268002079 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-eager-hist1-cuda]",
            "value": 0.4637376165561578,
            "unit": "iter/sec",
            "range": "stddev: 0.010992405922236418",
            "extra": "mean: 2.1563918135998392 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-eager-hist1-cuda]",
            "value": 0.014339323956081814,
            "unit": "iter/sec",
            "range": "stddev: 0.7279779954798303",
            "extra": "mean: 69.73829471059999 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist1-cuda-train]",
            "value": 1.1617436756462285,
            "unit": "iter/sec",
            "range": "stddev: 0.038981621974405974",
            "extra": "mean: 860.7750754001245 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist1-cuda-val]",
            "value": 1.9422041612554013,
            "unit": "iter/sec",
            "range": "stddev: 0.006820022656141418",
            "extra": "mean: 514.8789297998519 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-lazy-hist1-cuda]",
            "value": 0.4609556124123625,
            "unit": "iter/sec",
            "range": "stddev: 0.013173932187202914",
            "extra": "mean: 2.1694062792003024 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-lazy-hist1-cuda]",
            "value": 0.014371270026498168,
            "unit": "iter/sec",
            "range": "stddev: 0.16750393282144402",
            "extra": "mean: 69.58327260960031 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist1-cuda-train]",
            "value": 1.8841150435906504,
            "unit": "iter/sec",
            "range": "stddev: 0.020381705463426828",
            "extra": "mean: 530.7531529997505 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist1-cuda-val]",
            "value": 2.231509502941901,
            "unit": "iter/sec",
            "range": "stddev: 0.005624776334200074",
            "extra": "mean: 448.1271528002253 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist1-cuda]",
            "value": 0.5377776130558946,
            "unit": "iter/sec",
            "range": "stddev: 0.006113245987345213",
            "extra": "mean: 1.8595047017995967 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-lazy-hist1-cuda]",
            "value": 0.01445059877887039,
            "unit": "iter/sec",
            "range": "stddev: 0.38000593300762114",
            "extra": "mean: 69.20128468739968 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist2-cuda-train]",
            "value": 1.4760926908695868,
            "unit": "iter/sec",
            "range": "stddev: 0.037941030691560684",
            "extra": "mean: 677.4642312000651 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist2-cuda-val]",
            "value": 2.295118846537738,
            "unit": "iter/sec",
            "range": "stddev: 0.01173720840634126",
            "extra": "mean: 435.7072843999049 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist2-cuda]",
            "value": 0.6353418021540655,
            "unit": "iter/sec",
            "range": "stddev: 0.016153880865641172",
            "extra": "mean: 1.5739559346002352 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-lazy-hist2-cuda]",
            "value": 0.014373792635008356,
            "unit": "iter/sec",
            "range": "stddev: 0.14049424298710617",
            "extra": "mean: 69.57106070700029 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist2-cuda-train]",
            "value": 1.5664019027348646,
            "unit": "iter/sec",
            "range": "stddev: 0.011986449724006297",
            "extra": "mean: 638.4057617997314 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist2-cuda-val]",
            "value": 2.371933850211478,
            "unit": "iter/sec",
            "range": "stddev: 0.006452848423546822",
            "extra": "mean: 421.59691759989073 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist2-cuda]",
            "value": 0.6302747232274502,
            "unit": "iter/sec",
            "range": "stddev: 0.009900686738624705",
            "extra": "mean: 1.5866097165999236 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-eager-hist2-cuda]",
            "value": 0.014391911585535697,
            "unit": "iter/sec",
            "range": "stddev: 0.2062992623394171",
            "extra": "mean: 69.4834729950002 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist2-cuda-train]",
            "value": 1.123137715694587,
            "unit": "iter/sec",
            "range": "stddev: 0.008201740737027266",
            "extra": "mean: 890.362763200028 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist2-cuda-val]",
            "value": 1.7107597280199722,
            "unit": "iter/sec",
            "range": "stddev: 0.009203782893522696",
            "extra": "mean: 584.5356210000318 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-lazy-hist2-cuda]",
            "value": 0.5268470438979462,
            "unit": "iter/sec",
            "range": "stddev: 0.012345880559037058",
            "extra": "mean: 1.8980841054006306 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-lazy-hist2-cuda]",
            "value": 0.013886546796288448,
            "unit": "iter/sec",
            "range": "stddev: 0.753986239859969",
            "extra": "mean: 72.01214345579974 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist2-cuda-train]",
            "value": 1.025355644031495,
            "unit": "iter/sec",
            "range": "stddev: 0.009478825269278698",
            "extra": "mean: 975.2713663994655 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-eager-hist2-cuda-val]",
            "value": 1.7877126221669513,
            "unit": "iter/sec",
            "range": "stddev: 0.007057913146104774",
            "extra": "mean: 559.3740222004271 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-eager-hist2-cuda]",
            "value": 0.526439397943027,
            "unit": "iter/sec",
            "range": "stddev: 0.031254782656719986",
            "extra": "mean: 1.899553878200095 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-eager-hist2-cuda]",
            "value": 0.013910963935142727,
            "unit": "iter/sec",
            "range": "stddev: 0.7530011814980928",
            "extra": "mean: 71.8857445581998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist0-cuda-train]",
            "value": 2.231161158188111,
            "unit": "iter/sec",
            "range": "stddev: 0.01589913973073146",
            "extra": "mean: 448.1971175995568 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist0-cuda-val]",
            "value": 2.711768905450269,
            "unit": "iter/sec",
            "range": "stddev: 0.009603784129610967",
            "extra": "mean: 368.7629863998154 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-lazy-hist0-cuda]",
            "value": 0.44399402683036193,
            "unit": "iter/sec",
            "range": "stddev: 0.00978197538652535",
            "extra": "mean: 2.2522825524003567 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-lazy-hist0-cuda]",
            "value": 0.014369780215741628,
            "unit": "iter/sec",
            "range": "stddev: 0.10474795894812891",
            "extra": "mean: 69.59048677060018 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist1-cuda-train]",
            "value": 1.7563415372059898,
            "unit": "iter/sec",
            "range": "stddev: 0.04593342073163818",
            "extra": "mean: 569.3653419999464 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist1-cuda-val]",
            "value": 2.247558280442795,
            "unit": "iter/sec",
            "range": "stddev: 0.0359892007901685",
            "extra": "mean: 444.92728339973837 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist1-cuda]",
            "value": 0.5342043617220805,
            "unit": "iter/sec",
            "range": "stddev: 0.009741142053150378",
            "extra": "mean: 1.87194278379975 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-eager-hist1-cuda]",
            "value": 0.014369802561815704,
            "unit": "iter/sec",
            "range": "stddev: 0.31125466356833925",
            "extra": "mean: 69.59037855239986 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist0-cuda-train]",
            "value": 2.3702949737842234,
            "unit": "iter/sec",
            "range": "stddev: 0.007753292095504749",
            "extra": "mean: 421.888419399329 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist0-cuda-val]",
            "value": 2.87150734455982,
            "unit": "iter/sec",
            "range": "stddev: 0.008507200365944151",
            "extra": "mean: 348.2491527993261 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-remote-om4-om4-eager-hist0-cuda]",
            "value": 0.44217509537921845,
            "unit": "iter/sec",
            "range": "stddev: 0.0029971775069961323",
            "extra": "mean: 2.2615475417998825 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-remote-om4-om4-eager-hist0-cuda]",
            "value": 0.014386457845670384,
            "unit": "iter/sec",
            "range": "stddev: 0.24929523327753683",
            "extra": "mean: 69.50981337640042 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist0-cuda-train]",
            "value": 1.684869748949691,
            "unit": "iter/sec",
            "range": "stddev: 0.024395452001760992",
            "extra": "mean: 593.5176892002346 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[train_default.test.yaml-mock-om4-lazy-hist0-cuda-val]",
            "value": 2.045149367800542,
            "unit": "iter/sec",
            "range": "stddev: 0.022854433948139557",
            "extra": "mean: 488.9618410001276 msec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[train_default.test.yaml-mock-om4-lazy-hist0-cuda]",
            "value": 0.36011740348802684,
            "unit": "iter/sec",
            "range": "stddev: 0.015517970396399867",
            "extra": "mean: 2.776872182000079 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[train_default.test.yaml-mock-om4-lazy-hist0-cuda]",
            "value": 0.01435481300251673,
            "unit": "iter/sec",
            "range": "stddev: 0.09236420485193854",
            "extra": "mean: 69.6630461033994 sec\nrounds: 5"
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
          "id": "e66d0ec12f83862c56dbec7a14241671fb8a3846",
          "message": "Update base image to speed up CI, give instructions on how to do this (#184)\n\nWe could consider just making this always be the latest image but that\nwould also be a bit brittle.\n\nYou can see the first \"install nvidia drivers\" step takes about 2\nminutes extra here\nhttps://github.com/suryadheeshjith/Ocean_Emulator/actions/runs/14336583301/job/40185301143\nbecause it's waiting for automated upgrades to complete; this should\nreduce that.",
          "timestamp": "2025-04-08T12:08:32-04:00",
          "tree_id": "e526de0cff987379e6476caf723380fe57e4d866",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/e66d0ec12f83862c56dbec7a14241671fb8a3846"
        },
        "date": 1744131308094,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_LAZY-cuda-mock-train_default.test.yaml]",
            "value": 0.044275393355207256,
            "unit": "iter/sec",
            "range": "stddev: 0.12827653345777282",
            "extra": "mean: 22.58590888120002 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-mock-train_default.test.yaml]",
            "value": 0.052955064477822264,
            "unit": "iter/sec",
            "range": "stddev: 0.07573439840784779",
            "extra": "mean: 18.883935084599944 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-mock-train_default.test.yaml]",
            "value": 0.12803112689151508,
            "unit": "iter/sec",
            "range": "stddev: 0.1314979355463288",
            "extra": "mean: 7.810600627200074 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-mock-train_default.test.yaml]",
            "value": 0.009752624069660031,
            "unit": "iter/sec",
            "range": "stddev: 0.4590864300302267",
            "extra": "mean: 102.53650636560005 sec\nrounds: 5"
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
          "id": "15f367623238390035d81215398ef9e75100fa6a",
          "message": "Torch-first data loader without dask (#178)\n\nI did some profiling and continue to see a huge amount of time spent in\nxarray indexing. I built a dataloader which goes as quickly as possible\ninto (CPU) torch and does not use dask. On my laptop this is 3x faster\nthan the eager data loader, and on my test GPU machine it's about 5x\nfaster in the data loader benchmarks. Also seems to eliminate any\nwaiting for data on my GPU machine with 4 workers when running with\nbatch size == 2.",
          "timestamp": "2025-04-08T13:09:10-04:00",
          "tree_id": "528cef2823966e8e35a9ade9ef153d4a66ead93d",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/15f367623238390035d81215398ef9e75100fa6a"
        },
        "date": 1744134657462,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-mock-train_default.test.yaml]",
            "value": 0.19636419585906828,
            "unit": "iter/sec",
            "range": "stddev: 0.21457173152889442",
            "extra": "mean: 5.092578082400041 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_LAZY-cuda-mock-train_default.test.yaml]",
            "value": 0.5321651955919164,
            "unit": "iter/sec",
            "range": "stddev: 0.08444595282835583",
            "extra": "mean: 1.8791157487999954 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-mock-train_default.test.yaml]",
            "value": 0.05609040924456297,
            "unit": "iter/sec",
            "range": "stddev: 1.2714167802017546",
            "extra": "mean: 17.828359847399998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-mock-train_default.test.yaml]",
            "value": 0.13221566684138705,
            "unit": "iter/sec",
            "range": "stddev: 0.10776091682257781",
            "extra": "mean: 7.563400192200016 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-mock-train_default.test.yaml]",
            "value": 0.010121685898075834,
            "unit": "iter/sec",
            "range": "stddev: 0.6261714788013174",
            "extra": "mean: 98.7977704574001 sec\nrounds: 5"
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
          "id": "975c149cf969ae1284f5efbf187330a4ca5cf7f8",
          "message": "Using local logger in train.py (#189)\n\nStarts to address #102 (incrementally).",
          "timestamp": "2025-04-09T15:40:49-07:00",
          "tree_id": "c5ff4cb3f26f0dd308b139c3d1e0d39f09d2c947",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/975c149cf969ae1284f5efbf187330a4ca5cf7f8"
        },
        "date": 1744241130750,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-mock-train_default.test.yaml]",
            "value": 0.19745868602508437,
            "unit": "iter/sec",
            "range": "stddev: 0.20458492935108674",
            "extra": "mean: 5.064350523800021 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_LAZY-cuda-mock-train_default.test.yaml]",
            "value": 0.5295314128456605,
            "unit": "iter/sec",
            "range": "stddev: 0.08058663407045534",
            "extra": "mean: 1.8884620926000935 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-mock-train_default.test.yaml]",
            "value": 0.05499953658968944,
            "unit": "iter/sec",
            "range": "stddev: 1.4558727502226",
            "extra": "mean: 18.18197137660004 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-mock-train_default.test.yaml]",
            "value": 0.12749486435195784,
            "unit": "iter/sec",
            "range": "stddev: 0.09913262355105171",
            "extra": "mean: 7.843453185999988 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-mock-train_default.test.yaml]",
            "value": 0.010055366574341888,
            "unit": "iter/sec",
            "range": "stddev: 0.6274113854027032",
            "extra": "mean: 99.4493828352 sec\nrounds: 5"
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
          "id": "00a492c73f87cd7c5a23ca0b3011f24f57705014",
          "message": "Inline inference refactors (#193)\n\nComponents — target extraction, aggregator recording, wandb logging, and\nwriting — are now handled in batches rather than singular timesteps.\nThis allowed me to remove the record_every parameter as well.\nAdditionally, some other minor optimizations were made.\n\nHere is a comparison of the time taken in minutes, before and after\noptimization:\n\n| Dataset (Period) | Old Inference (With Write) | Old Inference (Without\nWrite) | Optimized Inference (With Write) | Optimized Inference (Without\nWrite) |\n| ------------- | ------------- | ------------- | ------------- |\n------------- |\n| OM4 (~8 years) |\n[14:08](https://wandb.ai/m2lines/ocean-emulators/runs/pjzsf71m) |\n[13:14](https://wandb.ai/m2lines/ocean-emulators/runs/025m3u3s) |\n[03:52](https://wandb.ai/m2lines/ocean-emulators/runs/rubapmx5) |\n[02:54](https://wandb.ai/m2lines/ocean-emulators/runs/i9kkjb85) |\n| CM4 (20 years) |\n[24:52](https://wandb.ai/m2lines/ocean-emulators/runs/zcbmfvme) |\n[22:45](https://wandb.ai/m2lines/ocean-emulators/runs/v8zazvdi) | [05:27\n](https://wandb.ai/m2lines/ocean-emulators/runs/1ziu9lrt) |\n[03:17](https://wandb.ai/m2lines/ocean-emulators/runs/mxk43iwq) |\n\nI made a comparison between writing the actual zarr and skipping write\nbecause during train we don't actually save the data produced by the\ninference rollout but in standalone inference, we do.\n \nTraining should now be cut down from ~4 days to 2.5 days and Inference\nshould now go from 1152 SYPD (~100x) to 4800 SYPD (400x).\n\nAdd tests\n- [x] Ensure correct data fed to model (Use time indexed values for\ndata)\n- [x] ~~Check if batched output metrics are different from single output\nmetrics~~ This test is not needed since Alex caught my bug\n- [x] Fix time returned so we save appropriate time in zarr\n\n\nNOTE: The above times with the optimization actually had a minor bug in\nchoosing the next prognostic but there is no change in performance\nbecause I just had to change an index. See this\n[comment](https://github.com/suryadheeshjith/Ocean_Emulator/pull/193#discussion_r2037893244).\nA fixed run with OM4 with write is logged\n[here](https://wandb.ai/m2lines/ocean-emulators/runs/b3qznqb0) and time\ntaken was: 03:46\n\nCloses #182",
          "timestamp": "2025-04-11T13:35:46-04:00",
          "tree_id": "c025ae360c18694549e1c5e02bd06d92c2d3c75c",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/00a492c73f87cd7c5a23ca0b3011f24f57705014"
        },
        "date": 1744395349228,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-mock-train_default.test.yaml]",
            "value": 0.20178902750429792,
            "unit": "iter/sec",
            "range": "stddev: 0.5439466087117141",
            "extra": "mean: 4.955670842799918 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_LAZY-cuda-mock-train_default.test.yaml]",
            "value": 0.5276471293263423,
            "unit": "iter/sec",
            "range": "stddev: 0.07685581495253176",
            "extra": "mean: 1.8952059897998879 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-mock-train_default.test.yaml]",
            "value": 0.05529619637487676,
            "unit": "iter/sec",
            "range": "stddev: 0.5482029088732115",
            "extra": "mean: 18.084426516799976 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-mock-train_default.test.yaml]",
            "value": 0.13052038219045328,
            "unit": "iter/sec",
            "range": "stddev: 0.06758211863424776",
            "extra": "mean: 7.661638613200012 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-mock-train_default.test.yaml]",
            "value": 0.011055879264601583,
            "unit": "iter/sec",
            "range": "stddev: 1.208210170044013",
            "extra": "mean: 90.44961292239987 sec\nrounds: 5"
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
          "id": "da87f40dd31ea461e48420355907cdb32dbab072",
          "message": "Document how to use multitons (#209)\n\nFixes #161",
          "timestamp": "2025-04-15T11:05:24-04:00",
          "tree_id": "10ffb70f5dbdc62a15191b8394f03405ac07496f",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/da87f40dd31ea461e48420355907cdb32dbab072"
        },
        "date": 1744732053694,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-mock-train_default.test.yaml]",
            "value": 0.19575619161667948,
            "unit": "iter/sec",
            "range": "stddev: 0.24834558465959877",
            "extra": "mean: 5.108395252999981 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_LAZY-cuda-mock-train_default.test.yaml]",
            "value": 0.5117405872889935,
            "unit": "iter/sec",
            "range": "stddev: 0.09763732232482081",
            "extra": "mean: 1.954115082599992 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-mock-train_default.test.yaml]",
            "value": 0.053341755897587315,
            "unit": "iter/sec",
            "range": "stddev: 1.229636865109804",
            "extra": "mean: 18.74703940979998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-mock-train_default.test.yaml]",
            "value": 0.1284490568339951,
            "unit": "iter/sec",
            "range": "stddev: 0.11585721598724164",
            "extra": "mean: 7.785187565000024 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-mock-train_default.test.yaml]",
            "value": 0.010480676061082972,
            "unit": "iter/sec",
            "range": "stddev: 0.7238962222879853",
            "extra": "mean: 95.41369222480003 sec\nrounds: 5"
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
          "id": "8b2e5b23c4ec05300e6ec0fc9fdf83059da78cd1",
          "message": "Inline Inference - Clarity Refactor (#198)\n\n- Added a comment to provide clarity on what we are doing during\ninference and validation.",
          "timestamp": "2025-04-15T11:11:20-04:00",
          "tree_id": "46679b918f1c5e268e3ab0c4bc9bb344c1b961d5",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/8b2e5b23c4ec05300e6ec0fc9fdf83059da78cd1"
        },
        "date": 1744732545329,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-mock-train_default.test.yaml]",
            "value": 0.1967697502061093,
            "unit": "iter/sec",
            "range": "stddev: 0.4474759892742739",
            "extra": "mean: 5.082081971200023 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_LAZY-cuda-mock-train_default.test.yaml]",
            "value": 0.5266620240735287,
            "unit": "iter/sec",
            "range": "stddev: 0.07784610052618124",
            "extra": "mean: 1.898750914799939 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-mock-train_default.test.yaml]",
            "value": 0.05404427728887859,
            "unit": "iter/sec",
            "range": "stddev: 1.8647627339941053",
            "extra": "mean: 18.50334670320003 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-mock-train_default.test.yaml]",
            "value": 0.13012467792305832,
            "unit": "iter/sec",
            "range": "stddev: 0.0859370256242629",
            "extra": "mean: 7.684937368999999 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-mock-train_default.test.yaml]",
            "value": 0.010893587392582013,
            "unit": "iter/sec",
            "range": "stddev: 0.8852105171975463",
            "extra": "mean: 91.79712467180002 sec\nrounds: 5"
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
          "id": "49404e57add8a29a90a65115a8c3cb323238808b",
          "message": "Better colors for wandb map plots (#201)\n\nBetter coloring of land cells in the plots",
          "timestamp": "2025-04-15T11:57:53-04:00",
          "tree_id": "7211baffd337a8ada8e00e813c1035f14ed63174",
          "url": "https://github.com/suryadheeshjith/Ocean_Emulator/commit/49404e57add8a29a90a65115a8c3cb323238808b"
        },
        "date": 1744735165482,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_TORCH-cuda-mock-train_default.test.yaml]",
            "value": 0.2006857385045352,
            "unit": "iter/sec",
            "range": "stddev: 0.3687329256761139",
            "extra": "mean: 4.9829151161999565 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_LAZY-cuda-mock-train_default.test.yaml]",
            "value": 0.5399750703575027,
            "unit": "iter/sec",
            "range": "stddev: 0.07574183212567544",
            "extra": "mean: 1.8519373484000425 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__loader__1gb[LoaderVersion.OM4_EAGER-cuda-mock-train_default.test.yaml]",
            "value": 0.055354068035197226,
            "unit": "iter/sec",
            "range": "stddev: 1.0272516577711939",
            "extra": "mean: 18.06551958139994 sec\nrounds: 5"
          },
          {
            "name": "tests/test_datasets.py::test_profile__inference_loader__1gb[cuda-mock-train_default.test.yaml]",
            "value": 0.12922816996780573,
            "unit": "iter/sec",
            "range": "stddev: 0.06559718688162444",
            "extra": "mean: 7.7382508801999395 sec\nrounds: 5"
          },
          {
            "name": "tests/test_trainer.py::test_trainer__mini_benchmark[cuda-mock-train_default.test.yaml]",
            "value": 0.010778542145781079,
            "unit": "iter/sec",
            "range": "stddev: 1.1980049456118798",
            "extra": "mean: 92.77692534619987 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}