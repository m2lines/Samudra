# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
from playwright.sync_api import expect, sync_playwright


def test_physical_xray_real_controls():
    url = os.environ.get("SAMUDRA_PHYSICAL_XRAY_TEST_URL")
    if not url:
        pytest.skip("Requires completed physical Xray exports and running server")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            executable_path=os.environ.get("CHROMIUM_EXECUTABLE"),
            args=["--no-sandbox", "--disable-gpu"],
        )
        page = browser.new_page(viewport={"width": 1600, "height": 1100})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url, wait_until="domcontentloaded", timeout=120000)
        expect(page.get_by_text("Snapshot RMS", exact=False)).to_be_visible(
            timeout=120000
        )
        selects = page.locator("select")
        assert selects.count() == 3
        selects.nth(1).select_option(label="so_0")
        expect(page.get_by_text("No aligned reference", exact=False)).to_be_visible()
        selects.nth(2).select_option("Member 1")
        selects.nth(1).select_option(label="zos")
        selects.nth(0).select_option(label="test · 2021-01-01 · day 365")
        expect(
            page.get_by_role("heading", name="test · 2021-01-01 · day 365 · zos (m)")
        ).to_be_visible(timeout=120000)
        expect(page.get_by_text("Snapshot RMS", exact=False)).to_be_visible()
        page.get_by_role("button", name="Normalized", exact=True).click()
        expect(page.get_by_text("OM4 standard deviations", exact=False)).to_be_visible()
        page.screenshot(path="/tmp/physical-xray-browser.png", full_page=True)
        assert not errors
        browser.close()
