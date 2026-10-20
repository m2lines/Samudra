import torch
from common import OceanModel, highpass, interior_mask
from torch.nn import functional as F


def model(arm="full", checkpointing=False):
    torch.manual_seed(15)
    return OceanModel(
        torch.ones(2, 16, 32, dtype=torch.bool),
        arm=arm,
        prog_channels=2,
        boundary_channels=1,
        widths=(4, 8),
        latent_channels=3,
        checkpointing=checkpointing,
    )


def test_pinned_backbone_matches_explicit_feature_path():
    m = model().eval()
    p = torch.randn(1, 4, 16, 32)
    b = torch.randn(1, 2, 16, 32)
    with torch.no_grad():
        f = m.unet(torch.cat((p, b), 1))
        expected = m.decoder(
            F.pad(F.pad(f, (1, 1, 0, 0), mode="circular"), (0, 0, 1, 1))
        )
        actual, _ = m(p, b)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_initial_physical_parameters_match_all_arms():
    base = model().state_dict()
    for arm in ("detach", "latent_full", "latent_only"):
        other = model(arm).state_dict()
        for k, v in base.items():
            torch.testing.assert_close(v, other[k], rtol=0, atol=0)


def test_feedback_gradients_really_follow_requested_paths():
    for arm in ("full", "detach", "latent_full", "latent_only"):
        m = model(arm).eval()
        if arm.startswith("latent"):
            # Exercise learned nonzero latent readout, not only its zero init.
            torch.nn.init.normal_(m.latent_read.weight, std=0.1)
        p, h = m(torch.randn(1, 4, 16, 32), torch.randn(1, 2, 16, 32))
        feedback, state = m.feedback(p, h)
        result, _ = m(feedback, torch.randn(1, 2, 16, 32), state)
        gp = torch.autograd.grad(
            result.square().mean(), p, retain_graph=True, allow_unused=True
        )[0]
        assert (gp is None) == (arm in ("detach", "latent_only"))
        if h is not None:
            gh = torch.autograd.grad(result.square().mean(), h, allow_unused=True)[0]
            assert gh is not None and gh.abs().sum() > 0


def test_checkpoint_replay_preserves_batchnorm_stats_and_gradients():
    a = model(checkpointing=False).train()
    b = model(checkpointing=True).train()
    p = torch.randn(2, 4, 16, 32)
    f = torch.randn(2, 2, 16, 32)
    a(p, f)[0].square().mean().backward()
    b(p, f)[0].square().mean().backward()
    for (ka, va), (kb, vb) in zip(
        a.state_dict().items(), b.state_dict().items(), strict=True
    ):
        assert ka == kb
        torch.testing.assert_close(va, vb, rtol=0, atol=0)
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(pa.grad, pb.grad)


def test_highpass_excludes_land_and_constant_fields():
    wet = torch.ones(1, 2, 16, 32, dtype=torch.bool)
    wet[..., 4, 7] = False
    inside = interior_mask(wet)
    assert not inside[..., 4, 7].any() and not inside[..., 4, 8].any()
    assert not highpass(torch.ones_like(wet, dtype=torch.float32), inside).any()


def test_repeated_gradients_through_checkpoint_match_uncheckpointed():
    for training in (False, True):
        a = model(checkpointing=False).train(training)
        b = model(checkpointing=True).train(training)
        p = torch.randn(2, 4, 16, 32, requires_grad=True)
        q = p.detach().clone().requires_grad_()
        forcing = torch.randn(2, 2, 16, 32)
        ya, _ = a(p, forcing)
        yb, _ = b(q, forcing)
        for weighted in (False, True):
            la = ya.square().mean() if not weighted else ya.abs().mean()
            lb = yb.square().mean() if not weighted else yb.abs().mean()
            ga = torch.autograd.grad(la, p, retain_graph=True)[0]
            gb = torch.autograd.grad(lb, q, retain_graph=True)[0]
            torch.testing.assert_close(ga, gb)
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
