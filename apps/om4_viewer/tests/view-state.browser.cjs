// SPDX-FileCopyrightText: 2026 Samudra Authors
// SPDX-License-Identifier: Apache-2.0
// Run with Node and Playwright installed (or available through NODE_PATH).
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const {chromium} = require('playwright');
const app = path.resolve(__dirname, '..');
const dates = [3, 8, 13].map(day => `2000-01-${String(day).padStart(2, '0')} 12:00:00`);
const ends = [6, 11, 16].map(day => `2000-01-${String(day).padStart(2, '0')} 00:00:00`);
const catalog = ['A', 'B', 'Snapshots'].map(name => ({name, path: `https://example.test/${name}/OM4.zarr`,
  calendar: 'julian', times: dates, interval_end_times: ends,
  fields: ['thetao', 'vo', 'uo'].map(id => ({id, name: id,
    levels: [0, 1].map(level => ({key: `${id}_${level}`, label: `${2.5 + level * 10} m`, units: 'test units'}))}))}));
(async () => {
  const server = http.createServer((req, res) => {
    const file = req.url === '/viewer.js' ? 'viewer.js' : 'index.html';
    res.setHeader('Content-Type', file.endsWith('.js') ? 'text/javascript' : 'text/html');
    res.end(fs.readFileSync(path.join(app, file)));
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const browser = await chromium.launch({headless: true});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}}), errors = [];
    page.on('pageerror', e => errors.push(e.message));
    let datasets = catalog;
    await page.route('**/api/metadata', route => route.fulfill({json: {datasets}}));
    await page.route('**/api/limits?**', route => route.fulfill({json: {min: -5, max: 40, palette: 'thermal', reference: 'test'}}));
    await page.route('**/api/map?**', route => {
      const q = new URL(route.request().url()).searchParams;
      const values = new Float32Array(Number(q.get('width')) * Number(q.get('height'))).fill(Number(q.get('time')) + 10);
      return route.fulfill({contentType: 'application/octet-stream', body: Buffer.from(values.buffer)});
    });
    const base = `http://127.0.0.1:${server.address().port}/?keep=1#anchor=maps`;
    const settled = () => page.waitForFunction(() => panels.length && panels.every(p => p.values || p.emptyMessage) && !document.querySelector('.panel.loading'));
    const saved = () => page.evaluate(() => JSON.parse(new URLSearchParams(location.hash.slice(1)).get('view')));
    await page.goto(base); await settled();
    const historyLength = await page.evaluate(() => history.length);
    await page.locator('#add-map').click(); await settled();
    await page.locator('#add-map').click(); await settled();
    assert.equal(await page.locator('#map-count').innerText(), '6 maps');
    await page.locator('#map-columns').selectOption('3'); await settled();
    assert.equal(await page.locator('#panels').evaluate(e => getComputedStyle(e).gridTemplateColumns.split(' ').length), 3);
    await page.locator('[aria-label="Panel 5 dataset"]').selectOption('1'); await settled();
    await page.locator('[aria-label="Panel 5 field"]').selectOption('thetao'); await settled();
    await page.locator('[aria-label="Panel 5 depth"]').selectOption('thetao_1'); await settled();
    await page.locator('[aria-label="Panel 6 dataset"]').selectOption('new-computed');
    await page.locator('#difference-mode').selectOption('previous');
    await page.locator('#difference-a').selectOption('0');
    await page.getByRole('button', {name: 'Apply difference', exact: true}).click(); await settled();
    await page.evaluate(() => show(2));
    await page.locator('[aria-label="Panel 6 color minimum"]').fill('-8');
    await page.locator('[aria-label="Panel 6 color minimum"]').dispatchEvent('change');
    await page.locator('#zoom-in').click();
    await page.waitForFunction(() => panels.every(p => p.extent?.[2] - p.extent?.[0] === 180) && !document.querySelector('.panel.loading'));
    await page.locator('#stride').selectOption('6'); await page.locator('#speed').selectOption('5'); await page.locator('#loop').uncheck();
    const expected = await saved(), firstUrl = page.url();
    assert.equal(expected.time, ends[2]); assert.equal(expected.columns, 3);
    assert.equal(expected.panels[4].source, catalog[1].path);
    assert.equal(expected.panels[4].depth, 'thetao_1');
    assert.equal(expected.panels[5].difference.mode, 'previous');
    assert.equal(expected.panels[5].range.min, -8);
    assert.equal(new URL(firstUrl).searchParams.get('keep'), '1');
    assert.equal(new URLSearchParams(new URL(firstUrl).hash.slice(1)).get('anchor'), 'maps');
    assert.equal(await page.evaluate(() => history.length), historyLength, 'View changes replace the current history entry');

    await page.reload(); await settled();
    assert.deepEqual(await saved(), expected, 'Reload restores the full view');
    assert.equal(await page.evaluate(() => panels[5].values[0]), 1, 'Reload restores the computed calculation');
    // Real URL navigation adds an entry; edits within that entry must not add more.
    const second = structuredClone(expected);
    second.panels = second.panels.slice(0, 2); second.columns = 1;
    second.time = ends[0]; second.bounds = [-180, -90, 180, 90];
    const secondUrl = new URL(firstUrl), fragment = new URLSearchParams(secondUrl.hash.slice(1));
    fragment.set('view', JSON.stringify(second)); secondUrl.hash = fragment.toString();
    await page.evaluate(url => {location.href = url;}, secondUrl.href);
    await page.waitForFunction(() => panels.length === 2 && document.querySelector('#date').textContent === '2000-01-06'); await settled();
    await page.locator('#next').click(); await settled(); // Stride 6 clamps to final frame.
    const secondEdited = await saved();
    assert.equal(await page.evaluate(() => history.length), historyLength + 1);
    await page.goBack();
    await page.waitForFunction(() => panels.length === 6); await settled();
    assert.deepEqual(await saved(), expected, 'Back restores the earlier saved view');
    await page.goForward();
    await page.waitForFunction(() => panels.length === 2); await settled();
    assert.deepEqual(await saved(), secondEdited, 'Forward restores the edited history entry');

    // Source paths, rather than catalog positions, make saved links stable.
    datasets = [catalog[2], catalog[0], catalog[1]];
    await page.goto(firstUrl); await page.reload(); await settled();
    assert.deepEqual(await saved(), expected, 'Catalog reordering preserves dataset identity');
    assert.equal(await page.evaluate(() => panels[5].computed.a), 1);
    await page.getByRole('button', {name: 'Remove map 2', exact: true}).click(); await settled();
    assert((await page.locator('[aria-label="Panel 5 dataset"]').inputValue()).startsWith('computed:'), 'Map labels renumber after removal');
    while(await page.locator('.panel').count() > 1) {
      await page.getByRole('button', {name: 'Remove map 1', exact: true}).click(); await settled();
    }
    assert(await page.getByRole('button', {name: 'Remove map 1', exact: true}).isDisabled());
    assert.equal((await saved()).panels.length, 1);
    for(let count = 1; count < 16; count++) {await page.locator('#add-map').click(); await settled();}
    assert(await page.locator('#add-map').isDisabled());
    assert.equal((await saved()).panels.length, 16);
    await page.reload(); await settled();
    assert.equal(await page.locator('.panel').count(), 16, 'Maximum layout survives reload');
    await page.setViewportSize({width: 989, height: 1294});
    await page.locator('#map-columns').selectOption('4'); await settled();
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Narrow four-column controls do not overflow');
    await page.setViewportSize({width: 390, height: 844});
    assert.equal(await page.locator('#panels').evaluate(e => getComputedStyle(e).gridTemplateColumns.split(' ').length), 1, 'Mobile maps stack');
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Mobile controls do not overflow');

    await page.goto(`${base}&view=${encodeURIComponent('{broken')}`);
    await page.waitForFunction(() => document.querySelector('#status').textContent.includes('Could not restore saved view'));
    assert.equal(await page.locator('.panel').count(), 4, 'Invalid saved views fall back to defaults');
    assert.deepEqual(errors, []);
    console.log('View state checks passed: map layout, full URL round trip, computed maps, stable source identity, replace history, Back/Forward, limits, mobile and invalid links.');
  } finally {
    await browser.close(); await new Promise(resolve => server.close(resolve));
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
