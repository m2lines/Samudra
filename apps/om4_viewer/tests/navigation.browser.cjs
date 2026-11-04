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
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH||undefined});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/api/metadata',r=>r.fulfill({json:{datasets}}));
  await page.route('**/api/limits?**',r=>{const field=new URL(r.request().url()).searchParams.get('field');const range=field==='thetao_0'?[-2,30]:field==='thetao_1'?[0,4]:field.startsWith('uo')?[-1,1]:[-5,5];return r.fulfill({json:{min:range[0],max:range[1],palette:field.startsWith('thetao')?'thermal':'balance',reference:'fixture'}});});
  await page.route('**/api/map?**',r=>{const q=new URL(r.request().url()).searchParams,ds=Number(q.get('dataset')),t=Number(q.get('time'));const val=ds===0?0:ds===1?1+t:7+10*t;return r.fulfill({contentType:'application/octet-stream',body:Buffer.from(new Float32Array(Number(q.get('width'))*Number(q.get('height'))).fill(val).buffer)});});
  const url=state=>`http://127.0.0.1:${server.address().port}/#view=${encodeURIComponent(JSON.stringify(state))}`;
  const settled=()=>page.waitForFunction(()=>panels.length===7&&panels.every(p=>p.values||p.emptyMessage)&&!document.querySelector('.panel.loading'));
  await page.goto(url(initial));await settled();
  const reset=()=>page.evaluate(()=>{bounds=[-90,-45,90,45];redraw();});
  const span=()=>page.evaluate(()=>bounds[2]-bounds[0]);
  const wheel=opts=>page.locator('canvas').first().evaluate((canvas,opts)=>{
    const r=canvas.getBoundingClientRect();
    canvas.dispatchEvent(new WheelEvent('wheel',{bubbles:true,cancelable:true,clientX:r.left+r.width*.3,clientY:r.top+r.height*.4,...opts}));
  },opts);
  await reset();await wheel({deltaY:-1});
  assert.ok(await span()>179,'Tiny scroll must not jump');
  await reset();await wheel({deltaY:-100});
  assert.ok(await span()>160&&await span()<170,'Wheel notch should zoom moderately');
  await reset();await wheel({deltaY:-100000});
  assert.ok(await span()>159,'Huge event is bounded');
  await reset();await wheel({deltaY:-3,deltaMode:1});const lines=await span();
  await reset();await wheel({deltaY:-48});assert.ok(Math.abs(lines-await span())<1e-8,'Line and pixel units agree');
  await reset();await wheel({deltaY:0});assert.equal(await span(),180);
  await reset();await wheel({deltaY:-10,ctrlKey:true});
  assert.ok(await span()>170,'Trackpad pinch remains gradual');
  await reset();
  const anchor=()=>page.evaluate(()=>{const r=panels[0].canvas.getBoundingClientRect(),x=(Math.floor(r.left+r.width*.3)-r.left)/r.width,y=(Math.floor(r.top+r.height*.4)-r.top)/r.height;return [bounds[0]+x*(bounds[2]-bounds[0]),bounds[3]-y*(bounds[3]-bounds[1])];});
  const before=await anchor();await wheel({deltaY:-100});
  (await anchor()).forEach((v,i)=>assert.ok(Math.abs(v-before[i])<1e-8,'Cursor anchor stays fixed'));
  await reset();
  const cdp=await page.context().newCDPSession(page);
  const box=await page.locator('canvas').first().boundingBox(),x=box.x+box.width/2,y=box.y+box.height/2;
  const touch=(type,points)=>cdp.send('Input.dispatchTouchEvent',{type,touchPoints:points.map(([id,x,y])=>({id,x,y}))});
  await touch('touchStart',[[1,x-40,y],[2,x+40,y]]);
  await touch('touchMove',[[1,x-60,y],[2,x+60,y]]);
  assert.ok(Math.abs(await span()-120)<1,'Touch spread follows finger separation');
  const center=await page.evaluate(()=>[(bounds[0]+bounds[2])/2,(bounds[1]+bounds[3])/2]);
  center.forEach(v=>assert.ok(Math.abs(v)<1,'Pinch midpoint stays anchored'));
  await touch('touchEnd',[[1,x-60,y]]);
  const afterPinch=await span();
  await touch('touchMove',[[1,x-50,y]]);
  assert.ok(Math.abs(await span()-afterPinch)<1e-8,'Remaining finger pans without zoom jump');
  await touch('touchEnd',[]);
  await page.locator('#zoom-reset').click();assert.equal(await span(),360);
  assert.deepEqual(errors,[]);console.log('Zoom checks passed: proportional wheel/pinch, unit normalization, cap, cursor anchor, touch pinch, pinch-to-pan and reset.');
 }finally{await browser.close();await new Promise(r=>server.close(r));}
})().catch(e=>{console.error(e);process.exit(1)});
