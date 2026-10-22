# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Check all six monthly-replay entries through real WebSocket interactions."""

import os
from pathlib import Path

import numpy as np
import pytest
from playwright.sync_api import expect, sync_playwright
from test_browser import select, verify_maps

from data import Catalog

URL = os.environ.get("SAMUDRA_VIEWER_TEST_URL")
pytestmark = pytest.mark.skipif(not URL, reason="Requires a running viewer")
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("readout", ["mean8", "draw1"])
def test_six_monthly_models_surface_reconstruction_and_consistent_heat_profiles(
    readout,
):
    root = Catalog(os.environ.get("SAMUDRA_VIEWER_DATA", ROOT / ".data"))
    names = ("A", "E", "A-extend", "E-extend", "A-replay", "E-replay")
    if "monthly-A-replay" not in root.all_models:
        pytest.skip("Monthly replay exports are not registered")
    if readout == "draw1" and "monthly-A-replay-draw1" not in root.all_models:
        pytest.skip("Single-draw exports are not registered")
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            executable_path=os.environ.get("CHROMIUM_EXECUTABLE"),
            args=[
                "--no-sandbox",
                "--disable-gpu",
                "--disable-webgl",
                "--disable-webgl2",
            ],
        )
        page = browser.new_page(viewport=dict(width=1600, height=1100))
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(URL, wait_until="networkidle")
        page.wait_for_function(
            "window.Bokeh?.documents?.[0]?.get_model_by_name('source-prediction')?.data.value.length > 0",
            timeout=60000,
        )
        for name in names:
            print(f"Checking monthly model {name} ({readout})", flush=True)
            model = "monthly-" + name + ("-draw1" if readout == "draw1" else "")
            catalog = root.for_model(model)
            select(page, "model-select", root.all_models[model]["label"])
            assert (
                root.all_models[model]["lineage"].get("decoder_readout", "mean8")
                == readout
            )
            select(page, "view-select", "Annual surface forecasts")
            select(page, "origin-select", "2021-01-01")
            select(page, "variable-select", "Sea surface temperature")
            select(page, "reference-select", "Observations")
            slider = page.locator(".lead-slider [role=slider]")
            slider.focus()
            slider.press("End")
            verify_maps(page, catalog, "surface", model, "2021-01-01", 0, 72)
            expect(page.locator(".model-description")).to_contain_text(name)
            select(page, "view-select", "Initialized ocean interior")
            select(page, "variable-select", "Temperature")
            select(page, "depth-select", "1050 m")
            select(page, "reference-select", "IAP monthly context")
            verify_maps(page, catalog, "interior", model, "2021-01-01", 0, 11)
            select(page, "view-select", "Monthly ocean heat content")
            select(page, "variable-select", "0–700 m")
            select(page, "reference-select", "IAP monthly heat content")
            month = page.locator(".month-slider [role=slider]")
            month.focus()
            month.press("End")
            verify_maps(page, catalog, "heat", model, "2021-01-01", 0, 11)
            latitude = float(page.locator(".latitude-input input").input_value())
            longitude = float(page.locator(".longitude-input input").input_value())
            y = np.abs(catalog.lat - latitude).argmin()
            x = np.abs((catalog.lon - longitude + 180) % 360 - 180).argmin()
            expected = catalog.values("temperature", model, "2021-01-01")[0][
                11, :, y, x
            ]
            actual = page.evaluate(
                "Array.from(Bokeh.documents[0].get_model_by_name('profile-source').data.prediction)"
            )
            np.testing.assert_allclose(
                np.array(actual, dtype=float),
                expected,
                rtol=1e-6,
                atol=1e-6,
                equal_nan=True,
            )
            select(page, "reference-select", "Own initialized persistence")
            verify_maps(
                page, catalog, "heat", model, "2021-01-01", 0, 11, "persistence"
            )
            partner = "monthly-" + name + ("-draw1" if readout == "mean8" else "")
            if partner in root.all_models:
                select(page, "reference-select", root.all_models[partner]["label"])
                verify_maps(page, catalog, "heat", model, "2021-01-01", 0, 11, partner)
                expected_reference = catalog.values(
                    "temperature", model, "2021-01-01", partner
                )[1][11, :, y, x]
                actual_reference = page.evaluate(
                    "Array.from(Bokeh.documents[0].get_model_by_name('profile-source').data.reference)"
                )
                np.testing.assert_allclose(
                    np.array(actual_reference, dtype=float),
                    expected_reference,
                    rtol=1e-6,
                    atol=1e-6,
                    equal_nan=True,
                )
            print(f"Verified monthly model {name} ({readout})", flush=True)
        assert not errors
        browser.close()


@pytest.mark.parametrize("readout", ["mean8", "draw1"])
def test_precool_source_forecasts_and_all_om4_initializations(readout):
    root = Catalog(os.environ.get("SAMUDRA_VIEWER_DATA", ROOT / ".data"))
    suffix = "-draw1" if readout == "draw1" else ""
    precool = "monthly-A-replay-pre-cooldown" + suffix
    if precool not in root.all_models:
        pytest.skip("Pre-cooldown exports are not registered")
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            executable_path=os.environ.get("CHROMIUM_EXECUTABLE"),
            args=[
                "--no-sandbox",
                "--disable-gpu",
                "--disable-webgl",
                "--disable-webgl2",
            ],
        )
        page = browser.new_page(viewport=dict(width=1600, height=1100))
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(URL, wait_until="networkidle")
        page.wait_for_function(
            "window.Bokeh?.documents?.[0]?.get_model_by_name('source-prediction')?.data.value.length > 0",
            timeout=60000,
        )
        select(page, "model-select", root.all_models[precool]["label"])
        catalog = root.for_model(precool)
        for source in ("obs", "om4"):
            select(page, "view-select", "Annual surface forecasts")
            origin = "2015-01-01" + ("-om4" if source == "om4" else "")
            select(
                page,
                "origin-select",
                "2015-01-01 (om4)" if source == "om4" else "2015-01-01",
            )
            select(page, "variable-select", "Sea surface temperature")
            select(
                page,
                "reference-select",
                "OM4 gold" if source == "om4" else "Observations",
            )
            slider = page.locator(".lead-slider [role=slider]")
            slider.focus()
            slider.press("End")
            verify_maps(page, catalog, "surface", precool, origin, 0, 72)
            select(page, "view-select", "Monthly ocean heat content")
            select(page, "variable-select", "0–700 m")
            select(
                page,
                "reference-select",
                "OM4 monthly heat content"
                if source == "om4"
                else "IAP monthly heat content",
            )
            slider = page.locator(".month-slider [role=slider]")
            slider.focus()
            slider.press("End")
            verify_maps(page, catalog, "heat", precool, origin, 0, 11)
            if source == "om4":
                expect(
                    page.get_by_text("Profiles show OM4", exact=False)
                ).to_be_visible()
        select(page, "view-select", "Initialized ocean interior")
        for name in (
            "A",
            "E",
            "A-extend",
            "E-extend",
            "A-replay",
            "E-replay",
            "A-replay-pre-cooldown",
        ):
            model = "monthly-" + name + suffix
            select(page, "model-select", root.all_models[model]["label"])
            select(page, "origin-select", "2018-01-01 (om4)")
            select(page, "variable-select", "Temperature")
            select(page, "depth-select", "1050 m")
            select(page, "reference-select", "OM4 gold")
            verify_maps(
                page, catalog, "interior", model, "2018-01-01-om4", 0, 11, "om4"
            )
            latitude = float(page.locator(".latitude-input input").input_value())
            longitude = float(page.locator(".longitude-input input").input_value())
            y = np.abs(catalog.lat - latitude).argmin()
            x = np.abs((catalog.lon - longitude + 180) % 360 - 180).argmin()
            expected = catalog.values("interior", model, "2018-01-01-om4", "om4")[1][
                :, 0, y, x
            ]
            actual = page.evaluate(
                "Array.from(Bokeh.documents[0].get_model_by_name('profile-source').data.reference)"
            )
            np.testing.assert_allclose(
                np.array(actual, dtype=float),
                expected,
                rtol=1e-6,
                atol=1e-6,
                equal_nan=True,
            )
            select(page, "variable-select", "Zonal velocity (U)")
            verify_maps(
                page, catalog, "interior", model, "2018-01-01-om4", 2, 11, "om4"
            )
            print(f"Verified OM4 initialization {name} ({readout})", flush=True)
        select(page, "model-select", root.all_models["monthly-A" + suffix]["label"])
        select(page, "view-select", "Annual surface forecasts")
        select(page, "variable-select", "Sea surface temperature")
        select(page, "reference-select", "Observations")
        # This model has OM4 initialization only. Switching views must preserve
        # the checkpoint and fall back to an available observation forecast.
        verify_maps(page, catalog, "surface", "monthly-A" + suffix, "2018-01-01", 0, 72)
        assert not errors
        browser.close()
