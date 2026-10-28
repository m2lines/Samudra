# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
import os
import re

import pytest
from playwright.sync_api import expect, sync_playwright


def test_anatomy_real_controls():
    url = os.environ.get("SAMUDRA_ANATOMY_TEST_URL")
    if not url:
        pytest.skip("Requires live anatomy bundle and Panel server")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            executable_path=os.environ.get("CHROMIUM_EXECUTABLE"),
            args=[
                "--no-sandbox",
                "--disable-gpu",
                "--disable-webgl",
                "--disable-webgl2",
            ],
        )
        page = browser.new_page(viewport={"width": 1800, "height": 1100})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url, wait_until="domcontentloaded", timeout=120000)
        expect(
            page.get_by_text("Original A · OM4 · initial", exact=False)
        ).to_be_visible(timeout=120000)
        page.get_by_text("Selected field at native resolution", exact=True).click()
        page.get_by_role("button", name="OBS · 2k", exact=True).click()
        expect(
            page.get_by_text("Original A · OBS_PRE · initial", exact=False)
        ).to_be_visible(timeout=120000)
        page.get_by_role("button", name="Normalized", exact=True).click()
        expect(
            page.get_by_text(
                "OM4 standard deviations (zero = OM4 mean)", exact=False
            ).first
        ).to_be_visible(timeout=120000)
        page.get_by_role("button", name="OBS · 3k cooled", exact=True).click()
        expect(
            page.get_by_text("Original A · OBS · initial", exact=False)
        ).to_be_visible(timeout=120000)
        selects = page.locator("select")
        expect(selects.nth(3).locator("optgroup")).to_have_count(2)
        expect(selects.nth(3).locator("optgroup").first).to_have_attribute(
            "label", "Available"
        )
        expect(
            selects.nth(3).locator("optgroup").first.locator("option")
        ).to_have_count(2)
        selects.nth(1).select_option(label="One processor step → day 5 decode")
        expect(page.get_by_text("Original A · OBS · day5", exact=False)).to_be_visible(
            timeout=120000
        )
        selects.nth(2).select_option(label="Diffusion step: input + prediction")
        expect(page.get_by_text("Selected call 62:", exact=False)).to_be_visible(
            timeout=120000
        )
        expect(page.get_by_text("Noisy input", exact=True).first).to_be_visible(
            timeout=120000
        )
        expect(page.get_by_text("Denoiser prediction", exact=True).first).to_be_visible(
            timeout=120000
        )
        spark = page.locator("svg.rmse-sparkline").first
        expect(spark).to_be_visible()
        expect(spark.locator("circle")).to_have_count(1)
        expect(spark).to_have_attribute("aria-label", re.compile(".*call 62"))
        comparison = page.locator(".xray-field-grid").first
        maps = comparison.locator("img")
        first, second = maps.nth(0).bounding_box(), maps.nth(1).bounding_box()
        assert first is not None and second is not None
        assert abs(first["y"] - second["y"]) < 1
        assert abs(first["height"] - second["height"]) < 1
        bars = comparison.locator('span[style*="linear-gradient"]')
        first_bar, second_bar = bars.nth(0).bounding_box(), bars.nth(1).bounding_box()
        assert first_bar is not None and second_bar is not None
        assert abs(first_bar["y"] - second_bar["y"]) < 1
        selects.nth(1).select_option(label="Monthly decode (separate task)")
        expect(
            page.get_by_text("Original A · OBS · monthly", exact=False)
        ).to_be_visible(timeout=120000)
        selects.nth(1).select_option(
            label="Initialized full77 diagnostic (OBS: OM4 context)"
        )
        expect(
            page.get_by_text("OM4 context (not OBS truth)", exact=False).first
        ).to_be_visible(timeout=120000)
        page.get_by_text("Latent state", exact=True).click()
        expect(page.get_by_text("Initialized", exact=True).first).to_be_visible(
            timeout=120000
        )
        expect(page.get_by_text("After one step", exact=True).first).to_be_visible(
            timeout=120000
        )
        expect(
            page.get_by_text("Difference (after − initialized)", exact=True).first
        ).to_be_visible(timeout=120000)
        assert not errors, errors
        browser.close()
