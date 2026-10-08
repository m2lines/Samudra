# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Human-readable definitions of the saved observation comparison models."""

REPORT_BASE = "https://github.com/m2lines/Samudra/blob/experiment/patch-global-om4/docs/experiments/observation-d/"
EARLY_REPORT = REPORT_BASE + "early-fine-annual-2026-10-08.md"
PRESENTATION_REPORT = REPORT_BASE + "presentation-annual-2026-10-08.md"

# Counts include the shared initializer, processor and task-specific modules.
MODEL_DETAILS = {
    "U-global": (
        62_967_680,
        "Standard ConvNeXt U-Net",
        "Global 1° OM4 plus observations; 2,000 OM4 and 2,000 observation updates at the training endpoint.",
    ),
    "U-multitask": (
        62_967_680,
        "Standard ConvNeXt U-Net",
        "Mixes global 1° OM4, regional native ¼° OM4 patches and observations. Endpoint exposure: 1,000 global OM4, 1,000 patch and 2,000 observation updates.",
    ),
    "U-omit-patch": (
        62_967_680,
        "Standard ConvNeXt U-Net",
        "Control that omits the regional updates: 1,000 global OM4 and 2,000 observation updates, with no replacement for the 1,000 patch slots.",
    ),
    "U-patch-loss01": (
        62_967_680,
        "Standard ConvNeXt U-Net",
        "Global/patch/observation multitask training, with the entire native patch objective multiplied by 0.1. The learning rate is unchanged.",
    ),
    "U-aux01": (
        62_967_809,
        "Standard ConvNeXt U-Net with a training-only auxiliary head",
        "Global 1° OM4 and observation tasks, plus weight-0.01 supervision of standardized log-transformed subcell surface-velocity variance from ¼° OM4. No fine-resolution input or patch evolution task; the auxiliary head is unused during rollout.",
    ),
    "U-aux01-static": (
        62_967_809,
        "Standard ConvNeXt U-Net with a training-only auxiliary head",
        "Same auxiliary setup as U-aux01, but its target is the training-time mean spatial pattern of the log-transformed variance target.",
    ),
    "U-aux01-seasonal": (
        62_967_809,
        "Standard ConvNeXt U-Net with a training-only auxiliary head",
        "Same auxiliary setup as U-aux01, using the training-only calendar-month mean of the transformed target instead of date-specific fine-scale variability.",
    ),
    "U-aux01-shuffled": (
        62_967_809,
        "Standard ConvNeXt U-Net with a training-only auxiliary head",
        "Same auxiliary setup as U-aux01, with temporally shuffled targets. This breaks alignment to the evolving state while retaining target spatial structure; it is not white-noise supervision.",
    ),
    "U-aux01-anomaly": (
        62_967_809,
        "Standard ConvNeXt U-Net with a training-only auxiliary head",
        "Same auxiliary setup as U-aux01, supervising departures from the training seasonal target rather than its full spatial pattern.",
    ),
    "W-global": (
        100_659_264,
        "Wider ConvNeXt U-Net",
        "Global 1° OM4 plus observations. Processor widths are 192/288/384/576; the initializer architecture is unchanged. Updates are matched to U-global, not FLOPs.",
    ),
    "W-multitask": (
        100_659_264,
        "Wider ConvNeXt U-Net",
        "The wider processor trained on global 1° OM4, native ¼° patches and observations. Updates are matched to U-multitask, not FLOPs.",
    ),
    "A-global": (
        64_151_938,
        "ConvNeXt U-Net with axial attention",
        "Global 1° OM4 plus observations. Adds row then column attention at the U-Net bottleneck, with eight heads per axis; the initializer architecture is unchanged.",
    ),
    "A-multitask": (
        64_151_938,
        "ConvNeXt U-Net with axial attention",
        "The axial-attention processor trained on global 1° OM4, native ¼° patches and observations. Attention operates over whichever extent is supplied.",
    ),
    "U-multitask-early": (
        62_967_680,
        "Standard ConvNeXt U-Net",
        "Replaces half the recent OM4 updates with earlier 1958–1974 global 1° examples. Endpoint: 1,000 recent OM4, 1,000 earlier OM4 and 2,000 observation updates.",
    ),
    "U-multitask-early-latent": (
        63_364_930,
        "ConvNeXt U-Net with 10 recurrent latent channels",
        "Earlier coarse OM4 setup, with 10 additional learned channels initialized and autoregressed on every task alongside the 77 physical state fields.",
    ),
    "U-multitask-early-fine": (
        63_149_162,
        "ConvNeXt U-Net with fine-specific encoder and decoder",
        "Earlier 1958–1974 OM4 examples are global ¼° fields (720×1440). A 4× downsampling encoder and fine-output decoder surround the shared 180×360 processor. Fine modules are inactive on observation rollouts.",
    ),
    "U-multitask-early-fine-latent": (
        63_552_832,
        "Fine-specific encoder/decoder plus 10 recurrent latent channels",
        "The earlier global ¼° setup, with 10 learned channels initialized and autoregressed on every task. The fine encoder can encode unresolved structure into that state; annual inference still runs on the 1° observation grid.",
    ),
}


def description(model, info):
    """Describe the selected checkpoint without conflating exposure and lead."""
    lineage = info["lineage"]
    if model in MODEL_DETAILS:
        parameters, architecture, training = MODEL_DETAILS[model]
        selection = "Selected by integrated-plus-spectral observation validation; not selected using these annual test cases. Constant learning rate, no cooldown."
    else:
        parameters = 62_966_546
        architecture = (
            "Conditioned ConvNeXt U-Net with a shared initializer and task adapters"
        )
        mixed = bool(lineage["task_counts"].get("om4", 0))
        training = (
            "Starts from random weights and mixes global OM4 and observations throughout training, with increasing observation frequency; no separate OM4-pretraining phase."
            if mixed
            else "Starts from random weights and trains exclusively on observations; no OM4 updates or pretrained weights."
        )
        selection = "Fixed-exposure checkpoint from the original observation report, not a validation-selected checkpoint."
    counts = ", ".join(
        f"{int(v):,} {k.replace('_', ' ')}"
        for k, v in lineage.get("task_counts", {}).items()
    )
    step = lineage.get("global_step")
    checkpoint = f"Total update {step:,}" if step is not None else "Saved checkpoint"
    return (
        f"### What is this model?\n\n**{info['label']}** — {architecture}, approximately **{parameters / 1e6:.1f}M parameters**.\n\n"
        f"{training}\n\n**Displayed checkpoint:** {checkpoint}"
        + (f" ({counts} updates)." if counts else ".")
        + f" {selection}\n\n"
        "The initializer and processor are trained jointly. Own initialized persistence holds this model's inferred initial physical state fixed; inferred interiors can differ between models even when copied surface observations are identical.\n\n"
        "Annual examples initialize once and evolve 73 five-day steps under prescribed ERA5 forcing. Three test origins are available; these are saved forecasts, not live inference."
    )
