// SPDX-FileCopyrightText: 2026 Samudra Authors
// SPDX-License-Identifier: Apache-2.0
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const {chromium} = require('playwright');
const app = path.resolve(__dirname, '..');
function source(name, calendar, times, intervalEnds = times) {
  return {name, path: name, calendar, times, interval_end_times: intervalEnds,
    fields: ['thetao', 'vo', 'uo'].map(id => ({id, name: id,
      levels: [{key: `${id}_0`, label: '2.5 m', units: 'test units'}]}))};
}
const datasets = [
  source('OM4', 'julian', ['2000-02-24 12:00:00', '2000-02-29 12:00:00'],
    ['2000-02-27 00:00:00', '2000-03-03 00:00:00']),
  source('Other OM4', 'julian', ['2000-02-24 12:00:00', '2000-02-29 12:00:00'],
    ['2000-02-27 00:00:00', '2000-03-03 00:00:00']),
  source('CM4', 'noleap', ['0152-02-24 00:00:00', '0152-03-01 00:00:00', '0152-03-06 00:00:00']),
  // Deliberately identical date strings still must not match across calendars.
  source('Noleap overlap', 'noleap', ['2000-02-27 00:00:00', '2000-03-03 00:00:00']),
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
    const page = await browser.newPage({viewport: {width: 1400, height: 1000}});
    const errors = [], requests = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.route('**/api/metadata', route => route.fulfill({json: {datasets}}));
    await page.route('**/api/limits?**', route => route.fulfill({json: {min: -5, max: 40, palette: 'thermal', reference: 'test'}}));
    await page.route('**/api/map?**', route => {
      const q = new URL(route.request().url()).searchParams;
      requests.push([Number(q.get('dataset')), Number(q.get('time'))]);
      const values = new Float32Array(Number(q.get('width')) * Number(q.get('height'))).fill(Number(q.get('time')) + 10);
      return route.fulfill({contentType: 'application/octet-stream', body: Buffer.from(values.buffer)});
    });
    const settled = () => page.waitForFunction(() => panels.length && panels.every(p => p.values || p.emptyMessage) && !document.querySelector('.panel.loading'));
    await page.goto(`http://127.0.0.1:${server.address().port}`); await settled();
    assert.equal(await page.locator('#calendar').inputValue(), 'julian');
    assert.equal(await page.locator('#timeline').getAttribute('max'), '1', 'Noleap years do not expand the Julian timeline');
    assert(!requests.some(([ds]) => ds === 2), 'No CM4 reads before switching calendars');
    await page.locator('#next').click(); await settled();
    await page.locator('#zoom-in').click();
    await page.waitForFunction(() => bounds[2] - bounds[0] === 180 && !document.querySelector('.panel.loading'));
    const extent = await page.evaluate(() => [...bounds]);
    const historyLength = await page.evaluate(() => history.length);
    await page.locator('.panel').nth(3).getByRole('button', {name: 'Switch to noleap calendar', exact: true}).click(); await settled();
    assert.equal(await page.locator('#date').innerText(), '0152-02-24');
    assert.equal(await page.locator('#jump').inputValue(), '0152-02-24');
    assert.equal(await page.evaluate(() => panels[0].emptyMessage), 'Different calendar');
    assert.deepEqual(await page.evaluate(() => [...bounds]), extent, 'Switching preserves linked zoom');
    assert(await page.evaluate(() => panels[3].values !== null && panels[0].values === null));
    await page.locator('#jump').fill('0152-03-01'); await page.locator('#jump').dispatchEvent('change'); await settled();
    assert.equal(await page.evaluate(() => panels[3].shownDate), '0152-03-01 00:00:00', 'Ancient noleap date jump selects the recorded sample');
    await page.locator('#time-link').selectOption('exact'); await settled();
    assert.equal(await page.locator('#date').innerText(), '0152-03-01', 'CM4 means already have interval-end labels');
    await page.locator('.panel').nth(0).getByRole('button', {name: 'Switch to julian calendar', exact: true}).focus();
    await page.keyboard.press('Enter'); await settled();
    assert.equal(await page.locator('#date').innerText(), '2000-02-29', 'Returning restores the Julian position with the new alignment');
    await page.locator('#calendar').selectOption('noleap'); await settled();
    assert.equal(await page.locator('#date').innerText(), '0152-03-01', 'Returning remembers the CM4 date');
    const saved = page.url();
    assert.equal(await page.evaluate(() => history.length), historyLength, 'Calendar changes replace history');
    await page.reload(); await settled();
    assert.equal(await page.locator('#calendar').inputValue(), 'noleap', 'URL restores active calendar independent of panel order');
    assert.equal(await page.locator('#date').innerText(), '0152-03-01');
    assert.deepEqual(await page.evaluate(() => [...bounds]), extent);

    // Cross-calendar calculations are rejected, while CM4 temporal differences work.
    await page.locator('[aria-label="Panel 3 dataset"]').selectOption('new-computed');
    await page.locator('#difference-a').selectOption('0');
    await page.locator('#difference-b').selectOption('2');
    await page.getByRole('button', {name: 'Apply difference', exact: true}).click();
    assert.match(await page.locator('#difference-error').innerText(), /same calendar/);
    await page.locator('#difference-mode').selectOption('previous');
    await page.locator('#difference-a').selectOption('2');
    await page.getByRole('button', {name: 'Apply difference', exact: true}).click(); await settled();
    assert.equal(await page.evaluate(() => panels[2].values[0]), 1);

    // Identical strings do not align different calendars, even in interval mode.
    await page.locator('[aria-label="Panel 4 dataset"]').selectOption('3'); await settled();
    await page.locator('#time-link').selectOption('interval'); await settled();
    await page.locator('#calendar').selectOption('julian'); await settled();
    assert.equal(await page.evaluate(() => panels[3].emptyMessage), 'Different calendar');
    assert(await page.evaluate(() => panels[3].values === null));

    // If all maps in the active calendar disappear, use the remaining calendar.
    await page.goto(saved); await settled();
    await page.getByRole('button', {name: 'Remove map 4', exact: true}).click(); await settled();
    assert.equal(await page.locator('#calendar').inputValue(), 'julian');
    assert(await page.locator('#calendar').isDisabled());
    // Older links without calendar keep their previous first-panel behavior.
    const legacy = new URL(saved), state = JSON.parse(new URLSearchParams(legacy.hash.slice(1)).get('view'));
    delete state.calendar; state.time = '2000-02-29 12:00:00';
    legacy.hash = new URLSearchParams({view: JSON.stringify(state)}).toString();
    await page.goto(legacy.href); await page.reload(); await settled();
    assert.equal(await page.locator('#calendar').inputValue(), 'julian');
    assert.deepEqual(errors, []);
    console.log('Calendar checks passed: isolated timelines, switching, remembered dates, ancient noleap jumps, linked zoom, URL restore, calculations and legacy links.');
  } finally {
    await browser.close(); await new Promise(resolve => server.close(resolve));
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
