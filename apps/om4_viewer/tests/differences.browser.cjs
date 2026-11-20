// SPDX-FileCopyrightText: 2026 Samudra Authors
// SPDX-License-Identifier: Apache-2.0
// Run with Node and Playwright installed (or available through NODE_PATH).
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const {chromium} = require('playwright');
const app = path.resolve(__dirname, '..');
const dates = [3, 8, 13, 18].map(day => `2000-01-${String(day).padStart(2, '0')} 12:00:00`);
const ends = [6, 11, 16, 21].map(day => `2000-01-${String(day).padStart(2, '0')} 00:00:00`);
function source(name, times, intervalEnds, units = 'degC') {
  return {name, times, interval_end_times: intervalEnds, calendar: 'julian', path: name,
    fields: ['thetao', 'vo', 'uo'].map(id => ({id, name: id, levels: [{key: `${id}_0`, label: '2.5 m', units}]}))};
}
const datasets = [source('A', dates, ends), source('B with a gap', [dates[0], dates[2], dates[3]], [ends[0], ends[2], ends[3]]), source('Snapshots', ends, ends), source('Incompatible units', dates, ends, 'K')];
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
    const errors = [], reads = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.route('**/api/metadata', route => route.fulfill({json: {datasets}}));
    await page.route('**/api/limits?**', route => route.fulfill({json: {min: -5, max: 40, palette: 'thermal', reference: 'test source'}}));
    await page.route('**/api/map?**', async route => {
      const q = new URL(route.request().url()).searchParams, ds = Number(q.get('dataset')), index = Number(q.get('time'));
      assert(q.get('time') !== 'undefined' && index >= 0 && index < datasets[ds].times.length);
      reads.push({ds, index, bounds: q.get('bounds')});
      const value = ds === 0 ? 10 + index * 2 : ds === 1 ? 4 + index * 3 : 20 + index;
      const values = new Float32Array(Number(q.get('width')) * Number(q.get('height'))).fill(value);
      values[0] = -Infinity; values[1] = NaN;
      await route.fulfill({contentType: 'application/octet-stream', body: Buffer.from(values.buffer)});
    });
    await page.goto(`http://127.0.0.1:${server.address().port}`);
    await page.waitForFunction(() => panels.length === 4 && panels[0].values);

    async function build(panel, mode, a, b, alignment = 'exact') {
      await page.locator(`[aria-label="Panel ${panel} dataset"]`).selectOption('new-computed');
      assert(await page.locator('#difference-dialog').evaluate(e => e.open));
      await page.locator('#difference-mode').selectOption(mode);
      await page.locator('#difference-a').selectOption(String(a));
      if(mode === 'datasets') {
        await page.locator('#difference-b').selectOption(String(b));
        await page.locator('#difference-alignment').selectOption(alignment);
      }
      await page.getByRole('button', {name: 'Apply difference', exact: true}).click();
      await page.waitForFunction(() => !document.querySelector('#difference-dialog').open && !document.querySelector('.panel.loading'));
    }
    await page.locator('[aria-label="Panel 4 field"]').selectOption('thetao');
    await build(4, 'datasets', 0, 1);
    const abKey = await page.locator('[aria-label="Panel 4 dataset"]').inputValue();
    assert.equal(await page.locator('[aria-label="Panel 4 dataset"]').evaluate(e => e.selectedOptions[0].textContent), 'A − B with a gap');
    assert(await page.locator('.panel').nth(3).locator('.source-run').evaluate(e => e.hidden));
    assert.equal(await page.locator('[aria-label="Panel 1 dataset"]').locator(`option[value="${abKey}"]`).count(), 1, 'Computed entry is offered in every panel');
    assert.equal(await page.locator('[aria-label="Panel 4 dataset"]').locator('option[value="new-computed"]').innerText(), 'Computed…');
    assert.equal(await page.evaluate(() => panels[3].values[2]), 6);
    assert.equal(await page.evaluate(() => panels[3].values[0]), -Infinity);
    assert(await page.evaluate(() => Number.isNaN(panels[3].values[1])));
    assert.equal(await page.evaluate(() => panels[3].range.palette), 'balance');
    assert.equal(await page.evaluate(() => panels[0].range.palette), 'thermal');
    assert.equal(await page.evaluate(() => panels[3].range.max), 6);

    await page.evaluate(() => show(2));
    assert.equal(await page.evaluate(() => panels[3].values[2]), 7, 'Match timestamp rather than source index');
    assert.equal(await page.evaluate(() => panels[3].range.max), 6, 'Range stays fixed through playback');
    await page.evaluate(() => show(1));
    assert.equal(await page.evaluate(() => panels[3].emptyMessage), 'No matching reference timestamp');
    assert(await page.evaluate(() => panels[3].values === null));
    await page.evaluate(() => show(2));
    const original = await page.evaluate(() => panels[0].range);
    await page.locator('[aria-label="Panel 4 color minimum"]').fill('-9');
    await page.locator('[aria-label="Panel 4 color minimum"]').dispatchEvent('change');
    assert.deepEqual(await page.evaluate(() => panels[0].range), original, 'Difference ranges do not alter raw maps');

    await build(4, 'previous', 0, 0);
    assert.equal(await page.evaluate(() => panels[3].values[2]), 2);
    const previousKey = await page.locator('[aria-label="Panel 4 dataset"]').inputValue();
    assert.equal(await page.evaluate(() => computedCatalog.size), 2, 'New calculations retain earlier entries');
    await page.locator('[aria-label="Panel 4 dataset"]').selectOption('0');
    await page.waitForFunction(() => !document.querySelector('.panel.loading'));
    assert.equal(await page.evaluate(() => panels[3].values[2]), 14);
    assert(await page.locator('.panel').nth(3).locator('.source-run').evaluate(e => !e.hidden));
    await page.locator('[aria-label="Panel 4 dataset"]').selectOption(abKey);
    await page.waitForFunction(() => !document.querySelector('.panel.loading'));
    assert.equal(await page.evaluate(() => panels[3].values[2]), 7, 'Earlier formula can be selected after a raw source');
    await page.locator('[aria-label="Panel 2 dataset"]').selectOption(abKey);
    await page.waitForFunction(() => !document.querySelector('.panel.loading'));
    assert.equal(await page.evaluate(() => panels[1].values[2]), 7, 'Computed entries can be reused in another panel');
    await page.locator('[aria-label="Panel 2 dataset"]').selectOption('1');
    await page.locator('[aria-label="Panel 4 dataset"]').selectOption(previousKey);
    await page.waitForFunction(() => !document.querySelector('.panel.loading'));
    assert.equal(await page.evaluate(() => panels[3].values[2]), 2);
    await build(4, 'previous', 0, 0);
    assert.equal(await page.evaluate(() => computedCatalog.size), 2, 'Identical formulas reuse an entry');
    assert((await page.locator('.panel').nth(3).locator('.source-time').innerText()).includes('2000-01-08'));
    await page.evaluate(() => {document.querySelector('#stride').value = '6'; return show(1);});
    assert.equal(await page.evaluate(() => panels[3].values[2]), 2, 'Previous source frame is independent of playback stride');
    await page.evaluate(() => show(0));
    assert.equal(await page.evaluate(() => panels[3].emptyMessage), 'No previous frame');

    await build(4, 'datasets', 0, 2);
    assert.equal(await page.evaluate(() => panels[3].emptyMessage), 'No matching reference timestamp');
    await build(4, 'datasets', 0, 2, 'interval');
    assert((await page.locator('[aria-label="Panel 4 dataset"]').evaluate(e => e.selectedOptions[0].textContent)).endsWith('same five-day interval'), 'Otherwise identical formulas distinguish matching modes');
    assert.equal(await page.evaluate(() => panels[3].values[2]), -10);
    const label = await page.locator('.panel').nth(3).locator('.source-time').innerText();
    assert(label.includes('2000-01-03') && label.includes('2000-01-06'));
    await page.locator('#zoom-in').click();
    await page.waitForFunction(() => panels[3].extent[2] - panels[3].extent[0] === 180 && !document.querySelector('.panel.loading'));
    assert.equal(await page.evaluate(() => panels[3].values[2]), -10);
    assert(reads.slice(-2).every(r => r.bounds === '-90,-45,90,45'));

    // Computed maps can own the timeline reference, including while unavailable.
    await build(1, 'previous', 0, 0);
    assert.equal(await page.evaluate(() => times.length), 4);
    assert.equal(await page.evaluate(() => panels[0].emptyMessage), 'No previous frame');
    await page.evaluate(() => show(2));
    assert.equal(await page.evaluate(() => panels[0].values[2]), 2);

    await page.locator('.panel').nth(3).getByRole('button', {name: 'Edit', exact: true}).click();
    const position = await page.evaluate(() => requested);
    await page.keyboard.press('ArrowLeft');
    assert.equal(await page.evaluate(() => requested), position, 'Modal arrow keys do not scrub');
    await page.locator('#difference-a').selectOption('0');
    await page.locator('#difference-b').selectOption('3');
    await page.getByRole('button', {name: 'Apply difference', exact: true}).click();
    assert((await page.locator('#difference-error').innerText()).includes('no matching fields'));
    assert(await page.locator('#difference-dialog').evaluate(e => e.open));
    await page.getByRole('button', {name: 'Cancel', exact: true}).click();
    assert.equal(await page.evaluate(() => panels[3].computed.b), 2, 'Cancel preserves previous formula');
    assert.deepEqual(errors, []);
    console.log('Difference checks passed: signed arithmetic, time matching, previous frame, masks, missing pairs, fixed ranges, zoom, modal validation and cancellation.');
  } finally {
    await browser.close(); await new Promise(resolve => server.close(resolve));
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
