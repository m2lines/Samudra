# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Run against a preview with a two-second render delay and so_0 failure."""

import os

import pytest
from playwright.sync_api import expect, sync_playwright


def test_loading_before_server_response_and_after_error():
    url = os.environ.get("SAMUDRA_XRAY_DELAYED_TEST_URL")
    if not url:
        pytest.skip("Requires test-only delayed render preview")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            executable_path=os.environ.get("CHROMIUM_EXECUTABLE"),
            args=["--no-sandbox", "--disable-gpu"],
        )
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url)
        expect(page.get_by_text("Snapshot RMS", exact=False)).to_be_visible(
            timeout=120000
        )
        for field in ["zos", "so_0", "thetao_0"]:
            page.locator("select").nth(1).select_option(label=field)
            expect(page.get_by_role("status")).to_be_visible(timeout=500)
            expect(page.get_by_role("status")).to_be_hidden(timeout=15000)
            if field == "so_0":
                expect(
                    page.get_by_text("Unable to load this view", exact=False)
                ).to_be_visible()
            else:
                assert page.locator("img").count() == 4
                assert page.locator("img").evaluate_all(
                    "imgs => imgs.every(i => i.complete && i.naturalWidth > 0)"
                )
        page.locator("select").nth(1).select_option(label="zos")
        page.wait_for_timeout(100)
        page.locator("select").nth(1).select_option(label="so_1")
        page.wait_for_timeout(2300)
        expect(page.get_by_role("status")).to_be_visible()
        expect(page.get_by_role("status")).to_be_hidden(timeout=15000)
        expect(
            page.get_by_role(
                "heading", name="validation · 2013-11-01 · initialization · so_1 (g/kg)"
            )
        ).to_be_visible()
        assert not errors
        browser.close()
