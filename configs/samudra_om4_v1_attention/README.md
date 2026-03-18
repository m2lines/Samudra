# Samudra OM4 V1 - Attention

This config set explores optional U-Net attention blocks for Samudra v1.
The current focus is bottleneck attention, with support for both axial and
full attention.

Axial attention factorizes 2D spatial attention into separate passes over the
height and width axes. Full attention instead flattens the full spatial grid
and attends over all token pairs directly. In both cases, the channel
dimension is treated as the embedding dimension and the attention output is
added back to the feature map residually.

Use [model.yaml](model.yaml) as the base model config and
[train.yaml](train.yaml) to launch training.
