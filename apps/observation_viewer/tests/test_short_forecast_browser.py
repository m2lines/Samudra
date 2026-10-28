# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Live short/annual forecast switching through the actual WebSocket."""

import os

import pytest
from playwright.sync_api import expect, sync_playwright

URL = os.environ.get("SAMUDRA_STOCHASTIC_VIEWER_TEST_URL")
pytestmark = pytest.mark.skipif(
    not URL, reason="Requires combined stochastic/annual catalog"
)


def test_short_ensemble_and_annual_controls_restore():
    assert URL is not None
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=os.environ.get("CHROMIUM_EXECUTABLE"),
            args=["--no-sandbox"],
        )
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(URL)
        models = page.locator(".model-select select")
        expect(models).to_be_visible(timeout=60000)
        legacy = os.environ.get("SAMUDRA_STOCHASTIC_LEGACY_LABEL", "obs08000")
        models.select_option(label=legacy)
        slider = page.locator(".lead-slider [role=slider]")
        slider.press("End")
        expect(slider).to_have_attribute("aria-valuemax", "365.0")
        for label in ("mean of 8", "member 1", "member 2"):
            models.select_option(label=f"Stochastic ConvNeXt C · {label}")
            expect(slider).to_have_attribute("aria-valuemax", "30.0")
            slider.press("End")
            expect(slider).to_have_attribute("aria-valuenow", "30.0")
            page.locator(".view-select select").select_option(
                label="Saved forecast interior"
            )
            expect(page.locator(".task-description")).to_contain_text(
                "days 5, 10, 15, 20, 25, 30"
            )
            page.locator(".view-select select").select_option(
                label="Annual surface forecasts"
            )
        models.select_option(label=legacy)
        expect(slider).to_have_attribute("aria-valuemax", "365.0")
        slider.press("End")
        expect(slider).to_have_attribute("aria-valuenow", "365.0")
        assert not errors
        browser.close()
