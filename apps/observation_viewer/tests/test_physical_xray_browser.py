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


def test_om4_and_obs_first_steps():
    url = os.environ.get("SAMUDRA_PHYSICAL_OM4_TEST_URL")
    if not url:
        pytest.skip("Requires appended OM4 and OBS first-step exports")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            executable_path=os.environ.get("CHROMIUM_EXECUTABLE"),
            args=["--no-sandbox", "--disable-gpu"],
        )
        page = browser.new_page(viewport={"width": 1600, "height": 1100})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url, timeout=120000)
        expect(page.get_by_text("Snapshot RMS", exact=False)).to_be_visible(
            timeout=120000
        )
        selects = page.locator("select")
        selects.nth(0).select_option(label="OM4 · 2015-01-01 · initialization")
        selects.nth(1).select_option(label="so_9")
        expect(page.get_by_text("Snapshot RMS", exact=False)).to_be_visible()
        expect(page.get_by_text("not ocean observations", exact=False)).to_be_visible()
        selects.nth(0).select_option(label="OM4 · 2015-01-01 · day 5")
        expect(
            page.get_by_role("heading", name="OM4 · 2015-01-01 · day 5 · so_9 (g/kg)")
        ).to_be_visible()
        selects.nth(2).select_option("Member 1")
        expect(page.locator("b").filter(has_text="Member 1")).to_be_visible()
        expect(page.get_by_text("Snapshot RMS", exact=False)).to_be_visible()
        page.screenshot(path="/tmp/physical-om4-day5.png", full_page=True)
        selects.nth(0).select_option(label="OBS · 2015-01-01 · day 5")
        expect(page.get_by_text("No aligned reference", exact=False)).to_be_visible()
        selects.nth(1).select_option(label="zos")
        expect(page.get_by_text("Snapshot RMS", exact=False)).to_be_visible()
        assert not errors
        browser.close()
