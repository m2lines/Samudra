# 5 levels

# ./.python-perlmutter submitit_hydra.py compute/greene=2x2 compute/greene/node=a100_debug wandb.mode=offline epochs=4 exp=train_unet_global_3D_5_v00 name="$(date +%F)-train_convnextunet_v0.0_5levels_bf16" region=global_3D batch_size=4 scheduler=True rand_seed=15 unet.ch_width=[80,100,150,300,400] hist=1 N_samples=160 --qos=debug

# ./.python-perlmutter submitit_hydra.py compute/greene=2x2 compute/greene/node=a100_debug wandb.mode=offline epochs=4 exp=train_unet_global_3D_all_v021 name="$(date +%F)-train_convnextunet_v021_alllevels_bf16" region=global_3D batch_size=4 scheduler=True rand_seed=15 unet.ch_width=[80,100,150,300,400] hist=1 N_samples=160 --qos=debug

# ./.python-perlmutter submitit_hydra.py compute/greene=2x2 compute/greene/node=a100_debug wandb.mode=offline epochs=4 exp=train_unet_global_3D_5_v0.0 name="$(date +%F)-train_convnextunet_v0.0_5levels_bf16" region=global_3D batch_size=4 scheduler=True rand_seed=15 unet.ch_width=[80,100,150,300,400] hist=1 N_samples=160 --qos=debug
# ./.python-perlmutter submitit_hydra.py compute/greene=2x2 compute/greene/node=a100_debug wandb.mode=offline epochs=4 exp=train_unet_global_3D_5_v0.0 name="$(date +%F)-train_convnextunet_v0.0_5levels_bf16" region=global_3D batch_size=4 scheduler=True rand_seed=15 unet.ch_width=[80,100,150,300,400] hist=1 N_samples=160 --qos=debug

# ./.python-perlmutter submitit_hydra.py compute/greene=2x2 compute/greene/node=a100_debug wandb.mode=offline epochs=4 exp=train_unet_global_3D_5_v0.0 name="$(date +%F)-train_convnextunet_v0.0_5levels_bf16_verify" region=global_3D batch_size=4 scheduler=True rand_seed=15 unet.ch_width=[80,100,150,300,400] hist=1 N_samples=160 --qos=debug

# ./.python-perlmutter submitit_hydra.py compute/greene=2x2 compute/greene/node=a100_debug wandb.mode=offline epochs=4 amp_mode=bf16 enable_fused=True exp=train_unet_global_3D_5_v0.0 name="$(date +%F)-train_convnextunet_v0.0_5levels_bf16_fused" region=global_3D batch_size=4 scheduler=True rand_seed=15 unet.ch_width=[80,100,150,300,400] hist=1 N_samples=160 --qos=debug

# ./.python-perlmutter submitit_hydra.py compute/greene=2x2 compute/greene/node=a100_debug wandb.mode=offline epochs=4 enable_fused=True enable_jit=True amp_mode=bf16 exp=train_unet_global_3D_5_v0.0 name="$(date +%F)-train_convnextunet_v0.0_5levels_bf16_fused_jit" region=global_3D batch_size=4 scheduler=True rand_seed=15 unet.ch_width=[80,100,150,300,400] hist=1 N_samples=160 --qos=debug

# ./.python-perlmutter submitit_hydra.py compute/greene=2x2 compute/greene/node=a100_debug wandb.mode=offline epochs=4 enable_fused=False enable_jit=True amp_mode=None exp=train_unet_global_3D_5_v0.0 name="$(date +%F)-train_convnextunet_v0.0_5levels_jit" region=global_3D batch_size=4 scheduler=True rand_seed=15 unet.ch_width=[80,100,150,300,400] hist=1 N_samples=160 --qos=debug

./.python-perlmutter submitit_hydra.py compute/greene=2x2 compute/greene/node=a100_debug wandb.mode=offline epochs=4 enable_fused=True enable_jit=True amp_mode=bf16 exp=train_unet_global_3D_5_v0.0 name="$(date +%F)-train_convnextunet_v0.0_5levels_bf16_jit_fusedv2" region=global_3D batch_size=4 scheduler=True rand_seed=15 unet.ch_width=[80,100,150,300,400] hist=1 N_samples=160 --qos=debug
# ./.python-perlmutter submitit_hydra.py compute/greene=2x2 compute/greene/node=a100_debug wandb.mode=offline epochs=4 exp=train_unet_global_3D_all_v021 name="$(date +%F)-train_convnextunet_v021_alllevels_bf16" region=global_3D batch_size=4 scheduler=True rand_seed=15 unet.ch_width=[80,100,150,300,400] hist=1 N_samples=160 --qos=debug

