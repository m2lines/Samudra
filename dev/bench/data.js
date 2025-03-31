window.BENCHMARK_DATA = {
  "lastUpdate": 1743441488663,
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
      }
    ]
  }
}