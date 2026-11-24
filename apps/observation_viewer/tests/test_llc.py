# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""LLC snapshot geometry, seasonal context, and actual browser controls."""

import json
import os

import numpy as np
import pytest
from llc_data import LLCCatalog


def fixture_bundle(root):
    root.mkdir(exist_ok=True)
    names = [f"{v}_{k}" for v in ("uo", "vo", "thetao", "so") for k in range(51)] + [
        "zos"
    ]
    wet = np.ones((205, 4, 8), dtype="f4")
    wet[:, 0, 0] = 0
    np.savez(
        root / "grid.npz",
        names=names,
        lat=[-60, -20, 20, 60],
        lon=np.arange(8) * 45 + 22.5,
        depth=np.arange(51) * 20 + 0.5,
        thickness=np.ones(51) * 20,
        wet_area=wet,
    )

    def save(name, n, channels, value):
        values = np.full((n, channels, 4, 8), value, dtype="f4")
        values[:, :, 0, 0] = np.nan
        np.save(root / (name + ".npy"), values)
        return name + ".npy"

    r = dict(
        id="obs-20140101",
        source="obs",
        origin="2014-01-01T00:00:00",
        horizon_hours=720,
        initial_times=["2013-12-31T18:00:00", "2014-01-01T00:00:00"],
        saved_hours=[6, 12, 24, 48, 72, 120, 240, 360, 480, 600, 720],
    )
    r["times"] = np.datetime_as_string(
        np.datetime64("2014-01-01T06") + np.arange(120) * np.timedelta64(6, "h"),
        unit="s",
    ).tolist()
    r["saved_times"] = [r["times"][h // 6 - 1] for h in r["saved_hours"]]
    r["history_times"] = np.datetime_as_string(
        np.datetime64("2013-12-25T00") + np.arange(29) * np.timedelta64(6, "h"),
        unit="s",
    ).tolist()
    r["history"] = save("history", 29, 2, 3.0)
    r["observed_initial"] = save("obs-init", 2, 2, np.nan)
    r["observed_surface"] = save("obs-surface", 120, 2, np.nan)
    obs = np.load(root / r["observed_surface"])
    obs[3::4] = 4.0
    np.save(root / r["observed_surface"], obs)
    r["prediction"] = {}
    for kind, count, channels, times in [
        ("initial", 2, 205, r["initial_times"]),
        ("surface", 120, 2, r["times"]),
        ("interior", 11, 205, r["saved_times"]),
    ]:
        r["prediction"][kind] = save("pred-" + kind, count, channels, 5.0)
        r["llc_" + kind] = save("llc-" + kind, count, channels, 7.0)
        r["llc_" + kind + "_times"] = [
            t.replace("2013", "2011").replace("2014", "2012") for t in times
        ]
        r["llc_" + kind + "_exact"] = [False] * count
    identity = dict(updates=11000, checkpoint_sha256="a" * 64)
    model = dict(
        label="TEST ONLY · OBS only EMA11k", identity=identity, cases={r["id"]: r}
    )
    (root / "catalog.json").write_text(
        json.dumps(
            dict(
                schema="llc-six-hour-viewer-v1", grid="grid.npz", models={"test": model}
            )
        )
    )
    return root


def test_seasonal_fallback_and_observation_priority(tmp_path):
    c = LLCCatalog(fixture_bundle(tmp_path))
    a, label, scored = c.reference("test", "obs-20140101", "surface", 0, "thetao", 0)
    assert (
        "LLC calendar-date match (different year)" in label
        and "2012-01-01T06" in label
        and not scored
    )
    assert a[1, 1] == 7
    a, label, scored = c.reference("test", "obs-20140101", "surface", 3, "thetao", 0)
    assert label.startswith("Observations") and scored and a[1, 1] == 4
    a, label, scored = c.reference("test", "obs-20140101", "interior", 2, "thetao", 20)
    assert not scored and "calendar-date match" in label
    a, label, scored = c.reference(
        "test", "obs-20140101", "surface", 0, "thetao", 0, "observations"
    )
    assert np.isnan(a).all() and not scored


def test_ui_controls_preserve_geometry_and_disable_context_error(tmp_path):
    from llc import LLCViewer

    viewer = LLCViewer(LLCCatalog(fixture_bundle(tmp_path)))
    viewer.variable.value = "vo"
    viewer.depth.value = 50
    assert "calendar-date match" in viewer.caption.object
    assert "No error map" in viewer.maps[2].title.text
    assert len(viewer.catalog.depth) == 51
    viewer.view.value = "surface"
    assert viewer.depth.value == 0 and viewer.variable.value in ("thetao", "zos")
    viewer.frame.value = 3
    assert "Observations" in viewer.maps[1].title.text
    assert viewer.maps[2].title.text == "Model − reference"
    viewer.frame.value = 119
    assert "+720 h" in viewer.caption.object


def test_playback_tracks_saved_frames_and_stops_on_task_change(tmp_path):
    from llc import LLCViewer

    viewer = LLCViewer(LLCCatalog(fixture_bundle(tmp_path)))
    assert viewer.player.end == 1
    viewer.view.value = "surface"
    assert viewer.player.end == 119
    viewer.player.value = 3
    assert viewer.frame.value == 3 and "+24 h" in viewer.caption.object
    viewer.frame.value = 119
    assert viewer.player.value == 119
    viewer.player.direction = 1
    viewer.view.value = "interior"
    assert viewer.player.direction == 0 and viewer.player.end == 10
    viewer.player.value = 10
    assert "+720 h" in viewer.caption.object
    viewer.player.value = 5
    assert "+120 h" in viewer.caption.object
    viewer.view.value = "history"
    assert viewer.player.end == 28
    viewer.player.value = 28
    assert "lead +0 h" in viewer.caption.object
    viewer.view.value = "initial"
    assert viewer.player.end == 1 and viewer.player.value == viewer.frame.value == 0


@pytest.mark.parametrize("source", ["llc", "obs"])
def test_exact_reference_labels_and_score_eligibility(tmp_path, source):
    c = LLCCatalog(fixture_bundle(tmp_path))
    record = c.case("test", "obs-20140101")
    record["source"] = source
    record["llc_initial_exact"][0] = True
    record["llc_initial_times"][0] = record["initial_times"][0]
    _, label, scored = c.reference("test", "obs-20140101", "initial", 0, "so", 20)
    expected = (
        "LLC matched reference" if source == "llc" else "LLC contemporaneous context"
    )
    assert label.startswith(expected) and scored == (source == "llc")


def reference_values(plot):
    return {
        item.label.value: item.renderers[-1].data_source.data
        for legend in plot.legend
        for item in legend.items
        if item.renderers[-1].name == "selected-reference-points"
    }


def test_llc_auto_and_explicit_observations_agree_across_maps_and_series(tmp_path):
    from llc import LLCViewer

    c = LLCCatalog(fixture_bundle(tmp_path))
    r = c.case("test", "obs-20140101")
    r["source"] = "llc"
    r["llc_surface_exact"] = [True] * 120
    r["llc_surface_times"] = r["times"]
    viewer = LLCViewer(c)
    viewer.view.value = "surface"
    viewer.frame.value = 3  # Both real observations and matched LLC are available.
    assert "LLC matched reference" in viewer.maps[1].title.text
    groups = reference_values(viewer.series.object)
    assert list(groups) == ["LLC matched reference"]
    np.testing.assert_array_equal(
        groups["LLC matched reference"]["value"], np.full(120, 7)
    )
    viewer.reference.value = "observations"
    assert "Observations" in viewer.maps[1].title.text
    data = reference_values(viewer.series.object)["Observations"]
    np.testing.assert_array_equal(
        np.flatnonzero(np.isfinite(data["value"])), np.arange(3, 120, 4)
    )
    assert np.all(data["value"][3::4] == 4)
    viewer.reference.value = "llc"
    assert list(reference_values(viewer.series.object)) == ["LLC matched reference"]
    # Archive gaps must retain context provenance and stay unscored.
    r["llc_surface_exact"][3] = False
    r["llc_surface_times"] = list(r["times"])
    r["llc_surface_times"][3] = "2012-01-02T00:00:00"
    viewer.refresh()
    assert "No error map" in viewer.maps[2].title.text
    context = reference_values(viewer.series.object)[
        "LLC calendar-date match (different year)"
    ]
    assert "2012-01-02" in context["reference"][3]
    viewer.reference.value = "auto"
    assert "Observations" in viewer.maps[1].title.text


def test_obs_auto_separates_sources_and_explicit_selection_controls_profile(tmp_path):
    from llc import LLCViewer

    viewer = LLCViewer(LLCCatalog(fixture_bundle(tmp_path)))
    viewer.view.value = "surface"
    groups = reference_values(viewer.series.object)
    assert set(groups) == {"Observations", "LLC calendar-date match (different year)"}
    obs = np.isfinite(groups["Observations"]["value"])
    context = np.isfinite(groups["LLC calendar-date match (different year)"]["value"])
    assert np.all(obs ^ context)
    assert obs.sum() == 30 and context.sum() == 90
    viewer.view.value = "interior"
    viewer.frame.value = 2
    viewer.depth.value = 20
    groups = reference_values(viewer.profile.object)
    obs = next(
        data for label, data in groups.items() if label.startswith("Observations")
    )
    np.testing.assert_array_equal(np.flatnonzero(np.isfinite(obs["value"])), [0])
    viewer.reference.value = "observations"
    assert len(reference_values(viewer.profile.object)) == 1
    assert not reference_values(
        viewer.series.object
    )  # No temperature observations at depth20.
    viewer.variable.value = "so"
    assert not reference_values(viewer.profile.object)
    assert "No selected reference" in viewer.profile.object.title.text
    viewer.reference.value = "llc"
    groups = reference_values(viewer.profile.object)
    assert len(groups) == 1
    data = next(iter(groups.values()))
    np.testing.assert_array_equal(data["value"], np.full(51, 7))
    assert "2012-01-02" in data["reference"][0]
    viewer.lat.value, viewer.lon.value = -60, 22.5  # Land is missing, never invented.
    assert not reference_values(viewer.profile.object)


def test_history_diagnostic_shows_input_window(tmp_path):
    from llc import LLCViewer

    viewer = LLCViewer(LLCCatalog(fixture_bundle(tmp_path)))
    viewer.view.value = "history"
    data = viewer.series.object.renderers[0].data_source.data
    np.testing.assert_array_equal(data["x"], np.arange(-168, 1, 6) / 24)
    np.testing.assert_array_equal(data["y"], np.full(29, 3))
    assert not reference_values(viewer.series.object)


URL = os.environ.get("SAMUDRA_LLC_TEST_URL")


@pytest.mark.skipif(not URL, reason="Needs running LLC test server")
def test_llc_browser():
    from playwright.sync_api import expect, sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True, executable_path=os.environ.get("CHROMIUM_EXECUTABLE")
        )
        page = browser.new_page(viewport=dict(width=1600, height=1000))
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(URL + "?view=interior&variable=vo&depth=50&frame=10")
        expect(page.locator(".llc-task select")).to_have_value(
            "Interior rollout", timeout=60000
        )
        expect(page.locator(".llc-field select")).to_have_value(
            "Northward velocity (m/s)"
        )
        expect(page.locator(".llc-depth select")).to_have_value("1000.5 m")
        expect(page.locator(".llc-frame select")).to_have_value(
            "+720 h · 2014-01-31T00:00:00"
        )
        expect(page.locator(".llc-caption")).to_contain_text(
            "LLC calendar-date match (different year)"
        )
        history = page.evaluate("history.length")
        page.locator(".llc-task select").select_option(label="Surface rollout")
        page.wait_for_function(
            "new URL(location.href).searchParams.get('view') === 'surface'"
        )
        page.locator(".llc-frame select").select_option(
            label="+720 h · 2014-01-31T00:00:00"
        )
        page.wait_for_function(
            "new URL(location.href).searchParams.get('frame') === '119'"
        )
        assert page.evaluate("history.length") == history
        expect(page.locator(".llc-caption")).to_contain_text("+720 h")
        second = browser.new_page()
        second.goto(page.url)
        expect(second.locator(".llc-frame select")).to_have_value(
            "+720 h · 2014-01-31T00:00:00", timeout=60000
        )
        # Exercise the actual draggable control and buttons over the WebSocket.
        slider = page.locator('.llc-player input[type="range"]')
        expect(slider).to_have_value("119")
        slider.focus()
        slider.press("Home")
        expect(page.locator(".llc-caption")).to_contain_text("+6 h")
        slider.press("ArrowRight")
        expect(page.locator(".llc-caption")).to_contain_text("+12 h")
        box = slider.bounding_box()
        page.mouse.move(box["x"] + 3, box["y"] + box["height"] / 2)
        page.mouse.down()
        page.mouse.move(
            box["x"] + box["width"] - 3, box["y"] + box["height"] / 2, steps=8
        )
        page.mouse.up()
        expect(slider).to_have_value("119")
        expect(page.locator(".llc-caption")).to_contain_text("+720 h")
        page.locator(".llc-player button.first").click()
        expect(slider).to_have_value("0")
        page.locator(".llc-player button.play").click()
        page.wait_for_function("new URL(location.href).searchParams.get('frame') >= 2")
        page.locator(".llc-player button.pause").click()
        page.locator(".llc-task select").select_option(label="Interior rollout")
        expect(slider).to_have_attribute("max", "10")
        page.locator(".llc-player button.last").click()
        expect(page.locator(".llc-caption")).to_contain_text("+720 h")
        page.locator(".llc-player button.previous").click()
        expect(page.locator(".llc-caption")).to_contain_text("+600 h")
        assert not errors
        browser.close()
