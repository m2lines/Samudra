# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Exercise real browser controls against the numeric report payloads."""

import os
import re
from pathlib import Path

import numpy as np
import pytest
from playwright.sync_api import expect, sync_playwright

from data import Catalog

URL = os.environ.get("SAMUDRA_VIEWER_TEST_URL")
pytestmark = pytest.mark.skipif(
    not URL, reason="Set SAMUDRA_VIEWER_TEST_URL to a running app"
)
ROOT = Path(__file__).resolve().parents[1]


def select(page, css, label):
    page.locator(f".{css} select").select_option(label=label)


def verify_maps(
    page, catalog, mode, model, origin, variable, index, reference="observations"
):
    prediction, comparison = catalog.values(mode, model, origin, reference)
    mask = catalog.array(catalog.meta[f"{mode}_mask"])
    wet = mask[index, variable] if mode == "interior" else mask[variable]
    a, b = prediction[index, variable][wet], comparison[index, variable][wet]
    for key, expected in zip(
        ["prediction", "reference", "difference"], [a, b, a - b], strict=True
    ):
        indices = np.linspace(0, len(expected) - 1, 41).astype(int)
        sample = [
            float(expected[i]) if np.isfinite(expected[i]) else None for i in indices
        ]
        page.wait_for_function(
            """({key, indices, sample, length}) => {
                const a = Bokeh.documents[0].get_model_by_name('source-' + key).data.value;
                return a.length === length && indices.every((ix,j) => sample[j] === null
                    ? !Number.isFinite(a[ix]) : Math.abs(a[ix]-sample[j]) < 1e-5);
            }""",
            arg=dict(
                key=key, indices=indices.tolist(), sample=sample, length=len(expected)
            ),
            timeout=30000,
        )
        actual = page.evaluate(
            "key => Array.from(Bokeh.documents[0].get_model_by_name('source-' + key).data.value)",
            key,
        )
        actual = np.array([np.nan if v is None else v for v in actual])
        np.testing.assert_allclose(
            actual, expected, rtol=1e-6, atol=1e-6, equal_nan=True
        )


def test_controls_update_real_maps_profiles_and_keep_sessions_independent():
    assert URL is not None
    catalog = Catalog(os.environ.get("SAMUDRA_VIEWER_DATA", ROOT / ".data"))
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=os.environ.get("CHROMIUM_EXECUTABLE"),
            headless=True,
            args=["--no-sandbox"],
        )
        page = browser.new_page(viewport=dict(width=1600, height=1100))
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(URL, wait_until="networkidle")
        page.wait_for_function(
            "window.Bokeh?.documents?.[0]?.get_model_by_name('source-prediction')?.data.value.length > 0",
            timeout=60000,
        )
        verify_maps(page, catalog, "surface", "obs08000", "2015-01-01", 0, 5)
        second = browser.new_page(viewport=dict(width=1600, height=1100))
        second.goto(URL, wait_until="networkidle")
        second.wait_for_function(
            "window.Bokeh?.documents?.[0]?.get_model_by_name('source-prediction')?.data.value.length > 0",
            timeout=60000,
        )
        select(page, "model-select", "Observations only · 16,000 updates")
        select(page, "origin-select", "2021-01-01")
        select(page, "variable-select", "Sea surface height")
        verify_maps(page, catalog, "surface", "scratch16000", "2021-01-01", 1, 5)
        # Drive the actual visible slider with a keyboard action.
        slider = page.locator(".lead-slider [role=slider]")
        slider.focus()
        slider.press("End")
        verify_maps(page, catalog, "surface", "scratch16000", "2021-01-01", 1, 72)
        select(page, "reference-select", "Mixed · 8,000 observation updates")
        verify_maps(
            page, catalog, "surface", "scratch16000", "2021-01-01", 1, 72, "obs08000"
        )
        # The other client retains its original model, variable, origin and lead.
        verify_maps(second, catalog, "surface", "obs08000", "2015-01-01", 0, 5)
        # Click the rendered map, then check the entire selected time series.
        canvas = second.locator("canvas").first.bounding_box()
        assert canvas is not None
        second.mouse.click(
            canvas["x"] + canvas["width"] * 0.6,
            canvas["y"] + canvas["height"] * 0.5,
        )
        expect(second.locator(".longitude-input input")).not_to_have_value(
            re.compile(r"^330(?:\.0+)?$"), timeout=30000
        )
        lat = float(second.locator(".latitude-input input").input_value())
        lon = float(second.locator(".longitude-input input").input_value())
        y = np.abs(catalog.lat - lat).argmin()
        x = np.abs((catalog.lon - lon + 180) % 360 - 180).argmin()
        expected = catalog.values("surface", "obs08000", "2015-01-01")[0][:, 0, y, x]
        selected = second.evaluate(
            "Array.from(Bokeh.documents[0].get_model_by_name('detail-source').data.prediction)"
        )
        np.testing.assert_allclose(
            np.array(selected, dtype=float), expected, equal_nan=True
        )
        # Exercise the actual play and pause buttons and verify the resulting map.
        second.locator("button.play").click()
        expect(second.locator(".lead-slider [role=slider]")).not_to_have_attribute(
            "aria-valuenow", re.compile(r"^30(?:\.0+)?$"), timeout=30000
        )
        second.locator("button.pause").click()
        # Let already queued websocket frames finish after pausing animation.
        second.locator(".lead-slider [role=slider]").evaluate(
            """async el => {
                let previous = el.getAttribute('aria-valuenow'), stable = 0;
                for (let attempt = 0; attempt < 30; attempt++) {
                    await new Promise(resolve => setTimeout(resolve, 500));
                    const current = el.getAttribute('aria-valuenow');
                    stable = current === previous ? stable + 1 : 0;
                    previous = current;
                    if (stable >= 5) return;
                }
                throw new Error('Playback did not settle after pause');
            }"""
        )
        lead_value = second.locator(".lead-slider [role=slider]").get_attribute(
            "aria-valuenow"
        )
        assert lead_value is not None
        lead = int(float(lead_value))
        verify_maps(
            second, catalog, "surface", "obs08000", "2015-01-01", 0, lead // 5 - 1
        )
        screenshot_root = ROOT / ".screenshots"
        screenshot_root.mkdir(exist_ok=True)
        second.screenshot(path=str(screenshot_root / "surface.png"), full_page=True)
        select(page, "view-select", "Initialized ocean interior")
        select(page, "variable-select", "Salinity")
        select(page, "depth-select", "1050 m")
        verify_maps(page, catalog, "interior", "scratch16000", "2021-01-01", 1, 11)
        select(page, "reference-select", "Training December climatology")
        verify_maps(
            page,
            catalog,
            "interior",
            "scratch16000",
            "2021-01-01",
            1,
            11,
            "climatology",
        )
        lat_input = page.locator(".latitude-input input")
        lat_input.fill("-30")
        lat_input.press("Enter")
        lon_input = page.locator(".longitude-input input")
        lon_input.fill("200")
        lon_input.press("Enter")
        y, x = np.abs(catalog.lat + 30).argmin(), np.abs(catalog.lon - 200).argmin()
        expected = catalog.values(
            "interior", "scratch16000", "2021-01-01", "climatology"
        )[0][:, 1, y, x]
        page.wait_for_function(
            "v => Math.abs(Bokeh.documents[0].get_model_by_name('profile-source').data.prediction[9] - v) < 1e-5",
            arg=float(expected[9]),
        )
        profile = page.evaluate(
            "Array.from(Bokeh.documents[0].get_model_by_name('profile-source').data.prediction)"
        )
        np.testing.assert_allclose(
            np.array(profile, dtype=float), expected, equal_nan=True
        )
        page.get_by_text("Along latitude", exact=True).click()
        page.wait_for_function(
            "Bokeh.documents[0].get_model_by_name('section-prediction').data.coordinate.every(v => v >= -90 && v <= 90)"
        )
        section = page.evaluate(
            "Array.from(Bokeh.documents[0].get_model_by_name('section-prediction').data.value)"
        )
        prediction = catalog.values("interior", "scratch16000", "2021-01-01")[0][
            :, 1, :, x
        ]
        wet = catalog.array(catalog.meta["interior_mask"])[:, 1, :, x]
        np.testing.assert_allclose(
            np.array(section, dtype=float), prediction[wet], equal_nan=True
        )
        page.screenshot(path=str(screenshot_root / "interior.png"), full_page=True)
        assert not errors, errors
        browser.close()


def test_monthly_heat_maps_point_series_and_mode_switching():
    assert URL is not None
    catalog = Catalog(os.environ.get("SAMUDRA_VIEWER_DATA", ROOT / ".data"))
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=os.environ.get("CHROMIUM_EXECUTABLE"),
            headless=True,
            args=["--no-sandbox"],
        )
        page = browser.new_page(viewport=dict(width=1600, height=1100))
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(URL, wait_until="networkidle")
        select(page, "view-select", "Monthly ocean heat content")
        verify_maps(page, catalog, "heat", "obs08000", "2015-01-01", 0, 0)
        select(page, "month-select", "2015-06")
        select(page, "variable-select", "700–2000 m")
        verify_maps(page, catalog, "heat", "obs08000", "2015-01-01", 1, 5)
        select(page, "origin-select", "2021-01-01")
        expect(page.locator(".month-select select")).to_have_value(
            "2021-06", timeout=30000
        )
        expect(
            page.get_by_role(
                "heading", name="Ocean heat content · 700–2000 m · 2021-06"
            )
        ).to_be_visible()
        select(page, "model-select", "Observations only · 16,000 updates")
        verify_maps(page, catalog, "heat", "scratch16000", "2021-01-01", 1, 5)
        y, x = np.abs(catalog.lat - 30).argmin(), np.abs(catalog.lon - 330).argmin()
        expected = catalog.values("heat", "scratch16000", "2021-01-01")
        actual = page.evaluate("""() => {
            const d=Bokeh.documents[0].get_model_by_name('detail-source').data;
            return {x:Array.from(d.x),prediction:Array.from(d.prediction),reference:Array.from(d.reference)};
        }""")
        np.testing.assert_array_equal(actual["x"], np.arange(1, 13))
        for key, values in zip(["prediction", "reference"], expected, strict=True):
            np.testing.assert_allclose(
                np.array(actual[key], dtype=float), values[:, 1, y, x], equal_nan=True
            )
        select(page, "reference-select", "Mixed · 8,000 observation updates")
        verify_maps(
            page, catalog, "heat", "scratch16000", "2021-01-01", 1, 5, "obs08000"
        )
        select(page, "reference-select", "IAP monthly heat content")
        verify_maps(page, catalog, "heat", "scratch16000", "2021-01-01", 1, 5)
        (ROOT / ".screenshots").mkdir(exist_ok=True)
        page.screenshot(
            path=str(ROOT / ".screenshots/monthly-heat.png"), full_page=True
        )
        select(page, "view-select", "Annual surface forecasts")
        verify_maps(page, catalog, "surface", "scratch16000", "2021-01-01", 0, 5)
        expect(page.locator(".month-select select")).not_to_be_visible()
        select(page, "view-select", "Initialized ocean interior")
        verify_maps(page, catalog, "interior", "scratch16000", "2021-01-01", 0, 9)
        assert not errors, errors
        browser.close()


def test_om4_gold_velocity_profiles_and_source_switching():
    assert URL is not None
    catalog = Catalog(os.environ.get("SAMUDRA_VIEWER_DATA", ROOT / ".data"))
    if "interior_examples" not in catalog.meta:
        pytest.skip("Requires the extended initialization bundle")
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=os.environ.get("CHROMIUM_EXECUTABLE"),
            headless=True,
            args=["--no-sandbox"],
        )
        page = browser.new_page(viewport=dict(width=1600, height=1100))
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(URL, wait_until="networkidle")
        page.wait_for_function(
            "window.Bokeh?.documents?.[0]?.get_model_by_name('source-prediction')?.data.value.length > 0",
            timeout=60000,
        )
        select(page, "model-select", "Observations only · 16,000 updates")
        select(page, "view-select", "Initialized ocean interior")
        verify_maps(page, catalog, "interior", "scratch16000", "2015-01-01", 0, 9)
        expect(page.locator(".origin-select option")).to_have_text(
            [record["label"] for record in catalog.meta["interior_examples"].values()]
        )
        select(page, "origin-select", "2018-01-01 (om4)")
        verify_maps(
            page, catalog, "interior", "obs08000", "2018-01-01-om4", 0, 9, "om4"
        )
        expect(page.locator(".model-select option")).to_have_text(
            ["Mixed · 8,000 observation updates"]
        )
        expect(page.locator(".reference-select option:checked")).to_have_text(
            "OM4 gold"
        )
        select(page, "depth-select", "1050 m")
        for variable, label in enumerate(
            ["Temperature", "Salinity", "Zonal velocity (U)", "Meridional velocity (V)"]
        ):
            select(page, "variable-select", label)
            verify_maps(
                page,
                catalog,
                "interior",
                "obs08000",
                "2018-01-01-om4",
                variable,
                11,
                "om4",
            )
        lat = float(page.locator(".latitude-input input").input_value())
        lon = float(page.locator(".longitude-input input").input_value())
        y, x = np.abs(catalog.lat - lat).argmin(), np.abs(catalog.lon - lon).argmin()
        predicted, gold = catalog.values(
            "interior", "obs08000", "2018-01-01-om4", "om4"
        )
        page.wait_for_function(
            "v => Math.abs(Bokeh.documents[0].get_model_by_name('profile-source').data.prediction[9] - v) < 1e-5",
            arg=float(predicted[9, 3, y, x]),
        )
        profile = page.evaluate(
            """() => {
                const data = Bokeh.documents[0].get_model_by_name('profile-source').data;
                return {prediction: Array.from(data.prediction), reference: Array.from(data.reference)};
            }"""
        )
        np.testing.assert_allclose(
            np.array(profile["prediction"], dtype=float),
            predicted[:, 3, y, x],
            equal_nan=True,
        )
        np.testing.assert_allclose(
            np.array(profile["reference"], dtype=float),
            gold[:, 3, y, x],
            equal_nan=True,
        )
        section = page.evaluate(
            "Array.from(Bokeh.documents[0].get_model_by_name('section-reference').data.value)"
        )
        wet = catalog.array(catalog.meta["interior_mask"])[:, 3, y, :]
        np.testing.assert_allclose(
            np.array(section, dtype=float), gold[:, 3, y, :][wet], equal_nan=True
        )
        select(page, "origin-select", "2018-01-01 (obs)")
        expect(page.locator(".reference-select option:checked")).to_have_text(
            "OM4 contemporaneous context", timeout=30000
        )
        assert not any(
            "IAP" in label
            for label in page.locator(".reference-select option").all_text_contents()
        )
        select(page, "model-select", "Observations only · 16,000 updates")
        verify_maps(
            page, catalog, "interior", "scratch16000", "2018-01-01", 3, 11, "om4"
        )
        select(page, "variable-select", "Temperature")
        select(page, "reference-select", "IAP monthly context")
        verify_maps(page, catalog, "interior", "scratch16000", "2018-01-01", 0, 11)
        select(page, "origin-select", "2015-01-01 (om4)")
        verify_maps(
            page, catalog, "interior", "obs08000", "2015-01-01-om4", 0, 11, "om4"
        )
        select(page, "view-select", "Annual surface forecasts")
        verify_maps(page, catalog, "surface", "obs08000", "2015-01-01", 0, 5)
        expect(page.locator(".origin-select option:checked")).to_have_text(
            "2015-01-01", timeout=30000
        )
        assert not errors
        browser.close()
