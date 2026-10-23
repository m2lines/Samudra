# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
from playwright.sync_api import expect, sync_playwright


def test_anatomy_real_controls():
    url = os.environ.get("SAMUDRA_ANATOMY_TEST_URL")
    if not url:
        pytest.skip("Requires live anatomy bundle and Panel server")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
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
        page.get_by_role("button", name="A 2k · OBS surface inputs", exact=True).click()
        expect(
            page.get_by_text("Original A · OBS_PRE · initial", exact=False)
        ).to_be_visible(timeout=120000)
        page.get_by_role("button", name="Normalized", exact=True).click()
        expect(
            page.get_by_text(
                "OM4 standard deviations (zero = OM4 mean)", exact=False
            ).first
        ).to_be_visible(timeout=120000)
        page.get_by_role(
            "button", name="A 3k cooled · OBS surface inputs", exact=True
        ).click()
        expect(
            page.get_by_text("Original A · OBS · initial", exact=False)
        ).to_be_visible(timeout=120000)
        selects = page.locator("select")
        selects.nth(0).select_option(label="One processor step → day 5 decode")
        expect(page.get_by_text("Original A · OBS · day5", exact=False)).to_be_visible(
            timeout=120000
        )
        selects.nth(1).select_option(label="Denoiser clean estimate")
        expect(page.get_by_text("Selected call 62:", exact=False)).to_be_visible(
            timeout=120000
        )
        selects.nth(0).select_option(label="Monthly decode (separate task)")
        expect(
            page.get_by_text("Original A · OBS · monthly", exact=False)
        ).to_be_visible(timeout=120000)
        selects.nth(0).select_option(
            label="Initialized full77 diagnostic (OBS: OM4 context)"
        )
        expect(
            page.get_by_text("OM4 context (not OBS truth)", exact=False).first
        ).to_be_visible(timeout=120000)
        assert not errors, errors
        browser.close()
