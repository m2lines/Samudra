// SPDX-FileCopyrightText: 2026 Samudra Authors
// SPDX-License-Identifier: Apache-2.0
// Run with Node and Playwright installed (or available through NODE_PATH).
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const {chromium} = require('playwright');

const app = path.resolve(__dirname, '..');
const dates = [10, 15, 20, 25, 30].map(day => `2014-10-${day} 12:00:00`);
const ends = [13, 18, 23, 28].map(day => `2014-10-${day} 00:00:00`).concat('2014-11-02 00:00:00');
function dataset(name, times, intervalEnds = times, sampling = 'five-day mean') {
  return {
    name, path: name, times, interval_end_times: intervalEnds, sampling, calendar: 'julian',
    fields: ['vo', 'uo'].map(id => ({id, name: id, levels: [{key: `${id}_0`, label: '2.5 m', units: 'm s-1', sampling}]})),
  };
}
const datasets = [
  dataset('Long record', dates, ends),
  dataset('Record with gaps', [dates[0], dates[2], dates[4]], [ends[0], ends[2], ends[4]]),
  dataset('Short rollout', dates.slice(2, 4), ends.slice(2, 4)),
  dataset('Snapshots', ends, ends, 'snapshot'),
  dataset('Same-day sample', ['2014-10-20 00:00:00']),
];

(async () => {
  const server = http.createServer((req, res) => {
    const file = req.url === '/viewer.js' ? 'viewer.js' : 'index.html';
    res.setHeader('Content-Type', file.endsWith('.js') ? 'text/javascript' : 'text/html');
    res.end(fs.readFileSync(path.join(app, file)));
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const browser = await chromium.launch({headless: true});
  try {
    const page = await browser.newPage();
    const errors = [], requests = [];
    let delayShort = false;
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/api/metadata', route => route.fulfill({json: {datasets}}));
    await page.route('**/api/limits?**', route => route.fulfill({json: {min: -1, max: 1, palette: 'balance', reference: 'test source'}}));
    await page.route('**/api/map?**', async route => {
      const query = new URL(route.request().url()).searchParams;
      const ds = Number(query.get('dataset')), index = Number(query.get('time'));
      assert(query.has('time') && query.get('time') !== 'undefined');
      assert(index >= 0 && index < datasets[ds].times.length);
      requests.push([ds, index]);
      if (delayShort && ds === 2) await new Promise(resolve => setTimeout(resolve, 100));
      const values = new Float32Array(Number(query.get('width')) * Number(query.get('height'))).fill(index + 0.25);
      await route.fulfill({contentType: 'application/octet-stream', body: Buffer.from(values.buffer)});
    });
    await page.goto(`http://127.0.0.1:${server.address().port}`);
    await page.waitForFunction(() => current === 0 && panels.length === 4 && panels[0].values && !panels[3].values);
    assert.equal(await page.evaluate(() => times.length), 5);
    assert.equal(await page.evaluate(() => panels[3].emptyMessage), 'Outside available dates');
    assert(!requests.some(([ds]) => ds === 2), 'No out-of-range rollout fetch');

    const shortPanel = page.locator('.panel').nth(3);
    await shortPanel.getByRole('button', {name: 'Jump to first available date: 2014-10-23', exact: true}).click();
    await page.waitForFunction(() => current === 2 && !document.querySelector('.panel.loading'));
    assert.equal(await page.evaluate(() => panels[3].shownDate), dates[2]);
    assert(await shortPanel.locator('.empty-state').isHidden());
    await page.evaluate(() => show(4));
    await shortPanel.getByRole('button', {name: 'Jump to last available date: 2014-10-28', exact: true}).focus();
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => current === 3 && !document.querySelector('.panel.loading'));
    assert.equal(await page.evaluate(() => panels[3].shownDate), dates[3]);

    await page.evaluate(() => show(2));
    assert.equal(await page.evaluate(() => panels[3].shownDate), dates[2]);
    assert(await page.evaluate(() => panels[3].values !== null));
    await page.keyboard.press('ArrowLeft');
    await page.waitForFunction(() => current === 1);
    assert(await page.evaluate(() => panels[3].values === null));
    assert.equal(await page.locator('.panel').nth(3).locator('.legend').evaluate(e => e.style.visibility), 'hidden');
    assert.equal(await page.evaluate(() => panels[1].emptyMessage), 'No frame at this timestamp');

    await page.evaluate(() => show(4));
    assert.equal(await page.evaluate(() => panels[3].emptyMessage), 'Outside available dates');
    await page.locator('#zoom-in').click();
    await page.waitForFunction(() => panels[0].extent[2] - panels[0].extent[0] === 180);
    const bounds = await page.locator('canvas').evaluateAll(nodes => nodes.map(n => n.dataset.bounds));
    assert(bounds.every(b => b === bounds[0]), 'Blank maps keep linked zoom');
    await shortPanel.getByRole('button', {name: 'Jump to last available date: 2014-10-28', exact: true}).click();
    await page.waitForFunction(() => current === 3 && !document.querySelector('.panel.loading'));
    assert.deepEqual(await page.locator('canvas').evaluateAll(nodes => nodes.map(n => n.dataset.bounds)), bounds, 'Date links preserve linked zoom');

    // Playback traverses both boundaries without stopping for a missing panel.
    await page.evaluate(async () => {await show(0); document.querySelector('#speed').value = '20'; document.querySelector('#loop').checked = false; play();});
    await page.waitForFunction(() => current === times.length - 1 && !playing);
    assert.equal(await page.evaluate(() => panels[0].shownDate), dates[4]);

    // A late in-range response must never repaint a newly out-of-range panel.
    delayShort = true;
    await page.evaluate(async () => {
      cache.clear(); cacheBytes = 0;
      const inRange = show(2);
      await new Promise(resolve => setTimeout(resolve, 25));
      await show(0); await inRange;
    });
    assert(await page.evaluate(() => panels[3].values === null));

    await page.evaluate(() => {
      panels[1].dataset.value = '3'; populateFields(panels[1]);
      document.querySelector('#time-link').value = 'exact'; rebuildTimeline();
      return show(times.indexOf('2014-10-20 12:00:00'));
    });
    assert.equal(await page.evaluate(() => times.length), 10);
    assert.equal(await page.evaluate(() => panels[1].emptyMessage), 'No frame at this timestamp');
    assert.equal(await page.evaluate(() => panels[3].shownDate), dates[2]);
    await page.locator('[aria-label="Panel 3 dataset"]').selectOption('4');
    await page.waitForFunction(() => !document.querySelector('.panel.loading'));
    await page.evaluate(() => show(0));
    await shortPanel.getByRole('button', {name: 'Jump to first available date: 2014-10-20', exact: true}).click();
    await page.waitForFunction(() => times[current] === '2014-10-20 12:00:00' && !document.querySelector('.panel.loading'));
    assert.equal(await page.evaluate(() => panels[3].shownDate), dates[2], 'Exact-mode links select the full timestamp, not an earlier sample on the same day');
    assert.equal(await page.evaluate(() => panels[2].emptyMessage), 'Outside available dates');
    assert.deepEqual(errors, []);
    console.log('Timeline browser checks passed: union, boundaries, gaps, playback, zoom, stale responses, exact timestamps.');
  } finally {
    await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
