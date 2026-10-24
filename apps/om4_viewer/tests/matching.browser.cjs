// SPDX-FileCopyrightText: 2026 Samudra Authors
// SPDX-License-Identifier: Apache-2.0
const assert=require('node:assert/strict');
const fs=require('node:fs');
const http=require('node:http');
const path=require('node:path');
const {chromium}=require('playwright');
const app=path.resolve(__dirname,'..');
const dates=['2000-01-03 12:00:00','2000-01-08 12:00:00'];
const datasets=['truth','model-a','model-b'].map(name=>({name,path:name,calendar:'julian',times:dates,interval_end_times:dates,
 fields:['thetao','uo','vo'].map(id=>({id,name:id,levels:[0,1].map(level=>({key:`${id}_${level}`,label:`${level?1050:2.5} m`,units:id==='thetao'?'degC':'m/s'}))}))}));
const absolute=(source,field,depth)=>({source,field,depth});
const delta=(a,b,depth,mode='datasets')=>({field:'thetao',depth,difference:{mode,a,b,alignment:'exact'}});
const initial={v:1,time:dates[1],calendar:'julian',alignment:'exact',columns:2,bounds:[-180,-90,180,90],panels:[
 absolute('truth','thetao','thetao_0'),absolute('model-a','thetao','thetao_1'),
 absolute('truth','uo','uo_0'),absolute('model-a','vo','vo_1'),
 delta('model-a','truth','thetao_0'),delta('model-b','truth','thetao_1'),delta('model-a','model-a','thetao_0','previous')
]};
(async()=>{
 const server=http.createServer((req,res)=>{const file=req.url==='/viewer.js'?'viewer.js':'index.html';res.setHeader('Content-Type',file.endsWith('.js')?'text/javascript':'text/html');res.end(fs.readFileSync(path.join(app,file)));});
 await new Promise(r=>server.listen(0,'127.0.0.1',r));
 const browser=await chromium.launch({headless:true});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/api/metadata',r=>r.fulfill({json:{datasets}}));
  await page.route('**/api/limits?**',r=>{const field=new URL(r.request().url()).searchParams.get('field');const range=field==='thetao_0'?[-2,30]:field==='thetao_1'?[0,4]:field.startsWith('uo')?[-1,1]:[-5,5];return r.fulfill({json:{min:range[0],max:range[1],palette:field.startsWith('thetao')?'thermal':'balance',reference:'fixture'}});});
  await page.route('**/api/map?**',r=>{const q=new URL(r.request().url()).searchParams,ds=Number(q.get('dataset')),t=Number(q.get('time'));const val=ds===0?0:ds===1?1+t:7+10*t;return r.fulfill({contentType:'application/octet-stream',body:Buffer.from(new Float32Array(Number(q.get('width'))*Number(q.get('height'))).fill(val).buffer)});});
  const url=state=>`http://127.0.0.1:${server.address().port}/#view=${encodeURIComponent(JSON.stringify(state))}`;
  const settled=()=>page.waitForFunction(()=>panels.length===7&&panels.every(p=>p.values||p.emptyMessage)&&!document.querySelector('.panel.loading'));
  const ranges=()=>page.evaluate(()=>panels.map(p=>[p.range.min,p.range.max,p.range.palette]));
  const edit=async(index,value)=>{const input=page.locator(`[aria-label="Panel ${index} color maximum"]`);await input.fill(String(value));await input.dispatchEvent('change');};
  await page.goto(url(initial));await settled();
  assert.equal(await page.locator('.match-range:checked').count(),7,'Old URLs default to matching');
  let r=await ranges();assert.deepEqual(r[0],r[1],'Temperature matches across depths and sources');assert.deepEqual(r[0],[-2,30,'thermal']);
  assert.deepEqual(r[2],r[3],'U and V match across fields and depths');assert.deepEqual(r[2],[-5,5,'balance']);
  assert.deepEqual(r[4],r[5]);assert.deepEqual(r[5],r[6],'Dataset errors and temporal deltas share a scale');assert.deepEqual(r[4],[-17,17,'balance']);
  await page.evaluate(()=>show(0));await settled();assert.deepEqual((await ranges())[4],r[4],'Playback never refits ranges');
  await page.evaluate(()=>show(1));await settled();
  await edit(5,9);r=await ranges();assert.equal(r[5][1],9);assert.equal(r[6][1],9);assert.equal(r[0][1],30,'Delta edit does not affect absolute maps');
  await page.locator('[aria-label="Panel 6 Match"]').uncheck();
  await edit(5,11);assert.equal((await ranges())[5][1],9,'Unmatched panel freezes its range');
  await edit(6,99);assert.equal((await ranges())[4][1],11,'Independent edit stays local');
  await page.reload();await settled();assert.equal(await page.locator('[aria-label="Panel 6 Match"]').isChecked(),false);assert.equal((await ranges())[5][1],99,'Opt-out and range survive URL reload');
  await page.locator('[aria-label="Panel 6 Match"]').check();r=await ranges();assert.equal(r[5][1],11,'Rejoining adopts existing group');
  await page.locator('.panel').nth(4).locator('.rescale').click();r=await ranges();assert.deepEqual(r[4],[-2,2,'balance']);assert.deepEqual(r[4],r[5]);assert.deepEqual(r[4],r[6]);
  await page.locator('.panel').nth(4).locator('.reset').click();await settled();r=await ranges();assert.deepEqual(r[4],[-17,17,'balance'],'Reset covers all matching first-frame suggestions');
  await page.locator('[aria-label="Panel 6 Match"]').uncheck();await edit(6,99);await page.locator('.panel').nth(4).locator('.reset').click();await settled();assert.equal((await ranges())[5][1],99,'Reset excludes opted-out panel');
  await page.locator('[aria-label="Panel 6 Match"]').check();
  await page.locator('[aria-label="Panel 2 field"]').selectOption('uo');await settled();r=await ranges();assert.deepEqual(r[1],r[2],'Changing unit joins the new group');
  const conflict=structuredClone(initial);conflict.panels[0].range={min:1,max:5,palette:'thermal'};conflict.panels[1].range={min:-4,max:8,palette:'thermal'};
  await page.goto(url(conflict));await settled();r=await ranges();assert.deepEqual(r[0],[-4,8,'thermal']);assert.deepEqual(r[0],r[1],'Conflicting saved matched ranges resolve conservatively');
  assert.deepEqual(errors,[]);console.log('Match checks passed: units/modes, depths, fields, temporal deltas, shared reset/fit/edit, opt-out, rejoin, playback, URL and old links.');
 }finally{await browser.close();await new Promise(r=>server.close(r));}
})().catch(e=>{console.error(e);process.exit(1)});
