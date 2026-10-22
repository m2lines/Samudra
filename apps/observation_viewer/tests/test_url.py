# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared views restore in a new session without growing browser history."""

import json
import os
from urllib.parse import parse_qs, urlsplit

import pytest
from playwright.sync_api import expect, sync_playwright

URL = os.environ.get("SAMUDRA_VIEWER_TEST_URL")
pytestmark = pytest.mark.skipif(not URL, reason="Requires a running viewer")


def test_share_url_restores_controls_without_history_entries():
    assert URL is not None
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=os.environ.get("CHROMIUM_EXECUTABLE"),
            args=["--no-sandbox"],
        )
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(URL + "?unrelated=keep#anchor")
        expect(page.locator(".variable-select select")).to_be_visible(timeout=60000)
        history_length = page.evaluate("history.length")

        def check(name, value):
            page.wait_for_function(
                "([name, value]) => JSON.parse(new URL(location.href).searchParams.get('view') || '{}')[name] === value",
                arg=[name, value],
            )
            assert page.evaluate("history.length") == history_length
            assert parse_qs(urlsplit(page.url).query)["unrelated"] == ["keep"]
            assert urlsplit(page.url).fragment == "anchor"

        page.locator(".variable-select select").select_option(
            label="Sea surface height"
        )
        check("variable", 1)
        page.locator(".lead-slider [role=slider]").press("End")
        check("frame", 365)
        page.locator(".view-select select").select_option(
            label="Initialized ocean interior"
        )
        check("mode", "interior")
        page.locator(".depth-select select").select_option(index=3)
        check("depth", 3)
        page.locator(".latitude-input input").fill("25")
        page.locator(".latitude-input input").press("Enter")
        check("latitude", 25.0)
        page.locator(".longitude-input input").fill("210")
        page.locator(".longitude-input input").press("Enter")
        check("longitude", 210.0)
        page.get_by_text("Along latitude", exact=True).click()
        check("orientation", "Along latitude")
        page.get_by_role("checkbox").uncheck()
        check("lock_colors", False)
        saved = page.url
        second = browser.new_page()
        second.goto(saved)
        expect(second.locator(".depth-select select")).to_be_visible(timeout=60000)
        for css in (
            "view-select",
            "model-select",
            "origin-select",
            "variable-select",
            "reference-select",
            "depth-select",
        ):
            expect(second.locator(f".{css} select")).to_have_value(
                page.locator(f".{css} select").input_value()
            )
        expect(second.locator(".latitude-input input")).to_have_value("25")
        expect(second.locator(".longitude-input input")).to_have_value("210")
        # Force a fresh state write so this compares server state, not the input URL.
        second.locator(".longitude-input input").fill("211")
        second.locator(".longitude-input input").press("Enter")
        second.wait_for_function(
            "JSON.parse(new URL(location.href).searchParams.get('view')).longitude === 211"
        )
        original = json.loads(parse_qs(urlsplit(saved).query)["view"][0])
        restored = json.loads(parse_qs(urlsplit(second.url).query)["view"][0])
        original["longitude"] = 211.0
        original.pop("colors")
        restored.pop("colors")
        assert restored == original
        invalid = browser.new_page()
        invalid.goto(URL + "?view=%7B%22model%22%3A%22missing%22%2C%22frame%22%3A-5%7D")
        expect(invalid.locator(".variable-select select")).to_be_visible(timeout=60000)
        assert not errors
        browser.close()
