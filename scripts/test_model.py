import torch
from ocean_emulators.config import TrainConfig
from ocean_emulators.models.samudra import Samudra

# Path to checkpoint and config
ckpt_path = "/scratch/ag11542/models/local_om4_samudra_crps/saved_nets/best_validation_ckpt.pt"
config_path = "configs/train_crps_retrained.yaml"

# Load checkpoint
ckpt = torch.load(ckpt_path, map_location="cpu")
ckpt_state = ckpt["model"]
ckpt_keys = set(k.removeprefix("module.") for k in ckpt_state.keys())

# Load config and build model
cfg = TrainConfig.from_file(config_path)
from ocean_emulators.datasets import prepare_datasets_from_config
train_ds, val_ds, area_weights = prepare_datasets_from_config(cfg)
model = cfg.model.build(
            in_channels=train_ds.in_channels,
                out_channels=train_ds.out_channels,
                    hist=cfg.data.hist,
                        wet=area_weights > 0,
                            area_weights=area_weights,
                                static_data=None,
                                    lat=None,
                                        lon=None,
                                        )
model_keys = set(model.state_dict().keys())

# Compare keys
missing_in_ckpt = model_keys - ckpt_keys
unexpected_in_ckpt = ckpt_keys - model_keys

print("=== Model State Dict Comparison ===")
print(f"Total keys in loaded model: {len(model_keys)}")
print(f"Total keys in checkpoint: {len(ckpt_keys)}")
print(f"\nKeys in model but missing in checkpoint: {len(missing_in_ckpt)}")
for k in sorted(missing_in_ckpt)[:20]:
        print(f"  - {k}")
        if len(missing_in_ckpt) > 20:
                print(f"  ... and {len(missing_in_ckpt) - 20} more")

                print(f"\nKeys in checkpoint but not in model: {len(unexpected_in_ckpt)}")
                for k in sorted(unexpected_in_ckpt)[:20]:
                        print(f"  + {k}")
                        if len(unexpected_in_ckpt) > 20:
                                print(f"  ... and {len(unexpected_in_ckpt) - 20} more")

                                if not missing_in_ckpt and not unexpected_in_ckpt:
                                        print("\n✅ State dicts match perfectly!")
                                    else:
                                            print("\n❌ Differences found. See above.")

