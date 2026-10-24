// SPDX-FileCopyrightText: 2026 Samudra Authors
// SPDX-License-Identifier: Apache-2.0
'use strict';
const $ = id => document.getElementById(id);
const palettes = {
  thermal: ['#102444','#244e83','#278c9b','#7bb99a','#e0d67b','#f3a052','#d34342'],
  balance: ['#263e85','#4a7dc2','#9bc3df','#eef1ee','#efbc9b','#cc6756','#851f38']
};
const lookup = Object.fromEntries(Object.entries(palettes).map(([name, colors]) => [name,
  Array.from({length:256}, (_, i) => {
    const t=i/255*(colors.length-1), j=Math.min(Math.floor(t),colors.length-2), f=t-j;
    const rgb=c=>[1,3,5].map(k=>parseInt(c.slice(k,k+2),16));
    return rgb(colors[j]).map((v,k)=>Math.round(v*(1-f)+rgb(colors[j+1])[k]*f));
  })]));
let datasets=[], times=[], timeMaps=[], exactMaps=[], intervalMaps=[], current=0, requested=0, revision=0;
let activeCalendar=null;
const calendarDates=new Map();
let playing=false, timer=null, zoomTimer=null;
let bounds=[-180,-90,180,90];
const panels=[], cache=new Map(), pending=new Map(), limitsCache=new Map(), computedCatalog=new Map();
let cacheBytes=0;
const MAX_MAPS=16;
const fmt=v=>Number(v).toLocaleString(undefined,{maximumSignificantDigits:4});
const isComputed=p=>p.dataset.value.startsWith('computed:');
const info=p=>isComputed(p)?p.computed.metadata:datasets[Number(p.dataset.value)];
const panelSources=p=>isComputed(p)?[p.computed.a,p.computed.b]:[Number(p.dataset.value)];
const rangeUnits=p=>info(p).fields.flatMap(f=>f.levels).find(l=>l.key===p.depth.value)?.units||'';
const rangeMode=p=>isComputed(p)?'delta':'absolute';
// Missing units cannot establish comparability. Unmatched panels key by identity.
const rangeKey=p=>p.match.checked&&rangeUnits(p)?JSON.stringify([rangeUnits(p),rangeMode(p)]):p;
function rangeNote(p){return `${p.match.checked&&rangeUnits(p)?`Matched ${rangeUnits(p)} · ${rangeMode(p)}`:'Independent range'} · ${p.range.note||p.range.reference||'fixed during playback'}`;}
function combinedRange(ranges){
  return {...ranges[0],min:Math.min(...ranges.map(r=>r.min)),max:Math.max(...ranges.map(r=>r.max)),palette:ranges.some(r=>r.palette==='balance')?'balance':'thermal'};
}
function matchRanges(candidates){
  const groups=new Map();
  panels.forEach((p,i)=>{const key=rangeKey(p);if(!groups.has(key))groups.set(key,[]);groups.get(key).push(i);});
  for(const members of groups.values()){
    // Existing ranges stay fixed during playback and when a new map joins.
    // Conflicting old URL ranges resolve to their union, independent of panel order.
    const existing=members.map(i=>panels[i].range).filter(Boolean);
    const choices=existing.length?existing:members.map(i=>candidates[i]).filter(Boolean);
    if(!choices.length)continue;
    const shared=combinedRange(choices);
    for(const i of members)panels[i].range={...shared};
  }
}
const status=(text,error=false)=>{$('status').textContent=text;$('status').classList.toggle('error',error);};
async function request(url, binary=false) {
  const response=await fetch(url);
  if(!response.ok){let error;try{error=(await response.json()).error;}catch{error=response.statusText;}throw new Error(error||`HTTP ${response.status}`);}
  return binary ? new Float32Array(await response.arrayBuffer()) : response.json();
}
function frame(ds,field,time,extent,width,height) {
  const query=new URLSearchParams({dataset:ds,field,time,bounds:extent.join(','),width,height});
  const key=query.toString();
  if(cache.has(key)){const value=cache.get(key);cache.delete(key);cache.set(key,value);return Promise.resolve(value);}
  if(pending.has(key))return pending.get(key);
  const promise=request(`/api/map?${query}`,true).then(values=>{
    if(values.length!==width*height)throw new Error('Frame shape does not match the requested map');
    cache.set(key,values);cacheBytes+=values.byteLength;
    while(cacheBytes>64*1024*1024){const first=cache.keys().next().value;cacheBytes-=cache.get(first).byteLength;cache.delete(first);}
    return values;
  }).finally(()=>pending.delete(key));
  pending.set(key,promise);return promise;
}
function limits(ds,field) {
  const key=`${ds}/${field}`;
  if(!limitsCache.has(key))limitsCache.set(key,request(`/api/limits?dataset=${ds}&field=${encodeURIComponent(field)}`).catch(error=>{limitsCache.delete(key);throw error;}));
  return limitsCache.get(key);
}
function stop(){playing=false;clearTimeout(timer);$('play').textContent='▶ Play';$('play').setAttribute('aria-pressed','false');}
async function tick(){
  if(!playing)return;
  let next=current+Number($('stride').value);
  if(next>=times.length){if(!$('loop').checked){stop();return;}next=0;}
  const start=performance.now(),ok=await show(next);
  if(!ok){stop();return;}
  if(playing)timer=setTimeout(tick,Math.max(0,1000/Number($('speed').value)-(performance.now()-start)));
}
function play(){if(playing){stop();return;}if(!times.length)return;playing=true;$('play').textContent='Ⅱ Pause';$('play').setAttribute('aria-pressed','true');tick();}
function option(value,text){const o=document.createElement('option');o.value=value;o.textContent=text;return o;}
function makePanel(i){
  const root=document.createElement('article');root.className='panel';
  root.innerHTML=`<div class="dataset-row"><span class="number">${String(i+1).padStart(2,'0')}</span><select aria-label="Panel ${i+1} dataset"></select><button class="edit-difference" hidden>Edit</button><button class="remove-map" aria-label="Remove map ${i+1}">×</button></div>
  <div class="panel-top"><select aria-label="Panel ${i+1} field"></select><select aria-label="Panel ${i+1} depth"></select></div>
  <div class="source-time"></div><div class="source-run"></div>
  <div class="map"><canvas width="1024" height="512" aria-label="Panel ${i+1} ocean map" role="img" tabindex="0"></canvas><div class="latlabels"></div><div class="empty-state" hidden><div class="empty-message"></div><div class="available-dates">Available: <button data-boundary="first"></button><span>–</span><button data-boundary="last"></button></div><div class="calendar-description" hidden></div><button class="switch-calendar" hidden></button></div></div>
  <div class="lonlabels"></div><div class="legend"><div class="colorbar"></div><div class="ticks"><span></span><span></span><span></span></div></div>
  <div class="readout">Hover for values · scroll to zoom · drag to pan</div>
  <div class="range-controls"><label>Min <input type="number" step="any" aria-label="Panel ${i+1} color minimum"></label><label>Max <input type="number" step="any" aria-label="Panel ${i+1} color maximum"></label><button class="rescale">Fit this frame</button><button class="reset">Reset range</button><label><input class="match-range" type="checkbox" checked aria-label="Panel ${i+1} Match">Match</label></div><div class="range-note">Matched fixed range · local 1° reference</div>`;
  $('panels').append(root);
  const selects=root.querySelectorAll('select'),inputs=root.querySelectorAll('input');
  const p={root,dataset:selects[0],field:selects[1],depth:selects[2],min:inputs[0],max:inputs[1],match:root.querySelector('.match-range'),canvas:root.querySelector('canvas'),values:null,range:null,units:'',image:document.createElement('canvas')};
  populateDatasetOptions(p);
  p.dataset.value=datasets.length>2&&i===3?2:(datasets.length>1?i%2:0);
  p.selection=p.dataset.value;
  p.dataset.onchange=()=>{stop();if(p.dataset.value==='new-computed'){p.dataset.value=p.selection;openDifference(p);return;}p.selection=p.dataset.value;if(isComputed(p))p.computed=computedCatalog.get(p.selection);populateFields(p);p.range=null;rebuildTimeline();show(requested);};
  root.querySelector('.edit-difference').onclick=()=>openDifference(p);
  root.querySelector('.remove-map').onclick=()=>{
    if(panels.length===1)return;
    stop();p.root.remove();panels.splice(panels.indexOf(p),1);refreshLayout();rebuildTimeline();show(requested);
  };
  p.field.onchange=()=>{populateDepth(p);p.range=null;stop();show(requested);};
  p.depth.onchange=()=>{p.range=null;stop();show(requested);};
  function manual(){const min=Number(p.min.value),max=Number(p.max.value);if(p.min.value===''||p.max.value===''||!Number.isFinite(min)||!Number.isFinite(max)||min>=max){status('Color minimum must be less than maximum.',true);return;}p.range={...p.range,min,max};applyRange(p,'Custom range · fixed during playback');}
  p.min.onchange=manual;p.max.onchange=manual;
  p.match.onchange=()=>{
    stop();
    if(p.match.checked){const other=panels.find(q=>q!==p&&rangeKey(q)===rangeKey(p)&&q.range);if(other)p.range={...other.range};}
    if(p.range)applyRange(p,p.range.note||p.range.reference||'fixed during playback');
    else show(requested);
  };
  root.querySelector('.rescale').onclick=()=>{
    if(!p.values)return;
    const finite=Array.from(p.values).filter(Number.isFinite).sort((a,b)=>a-b);
    if(!finite.length){status('This view has no valid ocean values.',true);return;}
    let min=finite[Math.floor((finite.length-1)*.02)],max=finite[Math.floor((finite.length-1)*.98)];
    if(p.range.palette==='balance'){max=Math.max(Math.abs(min),Math.abs(max),1e-6);min=-max;}
    p.range={...p.range,min,max:max>min?max:min+1};applyRange(p,'Fitted to visible frame · fixed during playback');
  };
  root.querySelector('.reset').onclick=()=>{for(const q of panels){if(rangeKey(q)===rangeKey(p))q.range=null;}stop();show(requested);};
  p.canvas.addEventListener('mousemove',e=>hover(p,e));
  p.canvas.addEventListener('mouseleave',()=>{$('tooltip').style.display='none';});
  root.querySelectorAll('[data-boundary]').forEach(button=>{button.onclick=()=>{
    const index=times.indexOf(button.dataset.time);
    if(index<0)return;
    stop();clearTimeout(scrubTimer);show(index);
  };});
  root.querySelector('.switch-calendar').onclick=()=>switchCalendar(info(p).calendar);
  attachNavigation(p);
  return p;
}
function applyRange(p,note){
  p.range.note=note;
  for(const q of panels){if(rangeKey(q)===rangeKey(p)){q.range={...p.range};q.root.querySelector('.range-note').textContent=rangeNote(q);colorize(q);render(q);}}
  updateUrl();
}
function populateFields(p,desired){
  const previous=desired||p.field.value,d=info(p);
  p.field.replaceChildren(...d.fields.map(f=>option(f.id,f.name)));
  if(d.fields.some(f=>f.id===previous))p.field.value=previous;
  populateDepth(p);
  p.dataset.title=isComputed(p)?p.dataset.selectedOptions[0].textContent:d.path;
  p.root.querySelector('.edit-difference').hidden=!isComputed(p);
  const source=p.root.querySelector('.source-run');
  source.hidden=isComputed(p);
  source.textContent=isComputed(p)?'':d.source_run?`Archive: ${d.source_run}`:'Archive run title not recorded in local store';
  if(d.source_link){
    const link=document.createElement('a');link.href=d.source_link;link.target='_blank';link.rel='noopener noreferrer';link.textContent=d.source_run;link.style.color='inherit';
    source.replaceChildren(document.createTextNode('Evaluation: '),link);
  }
  p.root.querySelector('.source-run').title=d.path;
}
function populateDepth(p){
  const field=info(p).fields.find(f=>f.id===p.field.value),previous=p.depth.value;
  p.depth.replaceChildren(...field.levels.map(l=>option(l.key,l.label)));
  if(field.levels.some(l=>l.key===previous))p.depth.value=previous;
  p.depth.disabled=field.levels.length===1;
}
function linkedTimes(d){return $('time-link').value==='interval'?(d.interval_end_times||d.times):d.times;}
function computedMetadata(config){
  const a=datasets[config.a],b=datasets[config.b];
  if(a.calendar!==b.calendar)throw new Error('Choose datasets using the same calendar.');
  const fields=a.fields.map(field=>{
    const other=b.fields.find(f=>f.id===field.id);
    const levels=field.levels.filter(level=>other?.levels.some(l=>l.key===level.key&&l.label===level.label&&l.units===level.units&&l.units));
    return {...field,levels:levels.map(l=>({...l,sampling:'difference'}))};
  }).filter(f=>f.levels.length);
  if(!fields.length)throw new Error('These datasets have no matching fields, depths, and units.');
  const name=config.mode==='previous'?`${a.name}: current − previous`:`${a.name} − ${b.name}`;
  return {...a,name,path:`${a.path} − ${b.path}`,fields,source_link:null};
}
let differencePanel=null;
function openDifference(p){
  stop();differencePanel=p;
  const config=isComputed(p)?p.computed:{a:Number(p.dataset.value),b:datasets.findIndex((_,i)=>i!==Number(p.dataset.value)),mode:'datasets',alignment:'exact'};
  $('difference-a').replaceChildren(...datasets.map((d,i)=>option(i,d.name)));
  $('difference-b').replaceChildren(...datasets.map((d,i)=>option(i,d.name)));
  $('difference-a').value=config.a;$('difference-b').value=config.b<0?config.a:config.b;
  $('difference-mode').value=config.mode;$('difference-alignment').value=config.alignment;
  $('difference-error').textContent='';updateDifferencePreview();$('difference-dialog').showModal();
}
function updateDifferencePreview(){
  const previous=$('difference-mode').value==='previous';
  $('difference-b-row').hidden=$('difference-alignment-row').hidden=previous;
  const a=datasets[Number($('difference-a').value)],b=datasets[Number($('difference-b').value)];
  $('difference-preview').textContent=previous?`${a.name}: current − previous`:`${a.name} − ${b.name}`;
  $('difference-help').textContent=previous?'Previous means the preceding recorded frame in this series, independent of the playback step size. Choose the field and depth in the panel after applying.':`Choose the field and depth in the panel after applying. ${$('difference-alignment').value==='exact'?'Both source timestamps must match exactly.':'Means at interval midpoints may be paired with snapshots at interval ends; both timestamps are shown.'} Both sources use their nearest cells at the same map locations, in the original units.`;
}
for(const id of ['difference-mode','difference-a','difference-b','difference-alignment'])$(id).onchange=updateDifferencePreview;
$('difference-cancel').onclick=()=>$('difference-dialog').close();
$('difference-form').onsubmit=e=>{
  e.preventDefault();
  const config={a:Number($('difference-a').value),b:Number($('difference-b').value),mode:$('difference-mode').value,alignment:$('difference-alignment').value};
  if(config.mode==='previous'){config.b=config.a;config.alignment='exact';}
  try{
    config.metadata=computedMetadata(config);
    const p=differencePanel;configureDifference(p,config);p.range=null;populateFields(p);rebuildTimeline();$('difference-dialog').close();show(requested);
  }catch(error){$('difference-error').textContent=error.message;}
};
function computedLabel(config){
  const duplicate=[...computedCatalog.values()].some(c=>c!==config&&c.metadata.name===config.metadata.name);
  return config.metadata.name+(duplicate?` · ${config.alignment==='exact'?'exact timestamp':'same five-day interval'}`:'');
}
function populateDatasetOptions(p){
  const previous=p.dataset.value;
  p.dataset.replaceChildren(...datasets.map((d,i)=>option(i,d.name)),...[...computedCatalog].map(([key,c])=>option(key,computedLabel(c))),option('new-computed','Computed…'));
  if([...p.dataset.options].some(o=>o.value===previous))p.dataset.value=previous;
  if(isComputed(p))p.dataset.title=p.dataset.selectedOptions[0].textContent;
}
function configureDifference(p,config){
  const key=`computed:${config.mode}:${config.a}:${config.b}:${config.alignment}`;
  if(!computedCatalog.has(key)){
    computedCatalog.set(key,config);panels.forEach(populateDatasetOptions);
  }
  p.computed=computedCatalog.get(key);
  p.dataset.value=p.selection=key;
}
function panelState(p){
  const state={field:p.field.value,depth:p.depth.value,match:p.match.checked};
  if(isComputed(p)){
    const c=p.computed;state.difference={mode:c.mode,a:datasets[c.a].path,b:datasets[c.b].path,alignment:c.alignment};
  }else state.source=info(p).path;
  if(p.range)state.range={min:p.range.min,max:p.range.max,palette:p.range.palette};
  return state;
}
let urlTimer;
function updateUrl(){
  clearTimeout(urlTimer);
  if(!panels.length||!times.length)return;
  const state={v:1,panels:panels.map(panelState),time:times[requested],calendar:activeCalendar,alignment:$('time-link').value,bounds,columns:Number($('map-columns').value),step:$('stride').value,speed:$('speed').value,loop:$('loop').checked};
  const url=new URL(location.href),fragment=new URLSearchParams(url.hash.slice(1));
  // Keep large multi-map views out of the server's request line.
  url.searchParams.delete('view');fragment.set('view',JSON.stringify(state));url.hash=fragment.toString();
  if(url.href!==location.href)history.replaceState(null,'',url);
}
function scheduleUrlUpdate(){clearTimeout(urlTimer);urlTimer=setTimeout(updateUrl,120);}
function refreshLayout(){
  $('panels').style.setProperty('--map-columns',$('map-columns').value);
  $('map-count').textContent=`${panels.length} map${panels.length===1?'':'s'}`;
  $('add-map').disabled=panels.length>=MAX_MAPS;
  panels.forEach((p,i)=>{
    p.root.querySelector('.number').textContent=String(i+1).padStart(2,'0');
    [p.dataset,p.field,p.depth].forEach((node,j)=>node.setAttribute('aria-label',`Panel ${i+1} ${['dataset','field','depth'][j]}`));
    p.min.setAttribute('aria-label',`Panel ${i+1} color minimum`);p.max.setAttribute('aria-label',`Panel ${i+1} color maximum`);p.match.setAttribute('aria-label',`Panel ${i+1} Match`);
    const remove=p.root.querySelector('.remove-map');remove.disabled=panels.length===1;remove.setAttribute('aria-label',`Remove map ${i+1}`);
  });
}
function statePlan(state){
  if(!state||state.v!==1||!Array.isArray(state.panels)||state.panels.length<1||state.panels.length>MAX_MAPS)throw new Error('Invalid map list in view URL.');
  if(!['exact','interval'].includes(state.alignment)||![1,2,3,4].includes(state.columns))throw new Error('Invalid layout in view URL.');
  const b=state.bounds;
  if(!Array.isArray(b)||b.length!==4||!b.every(Number.isFinite)||b[1]<-90||b[3]>90||b[3]<=b[1]||b[2]-b[0]<.5||b[2]-b[0]>360||Math.abs((b[3]-b[1])-(b[2]-b[0])/2)>1e-6)throw new Error('Invalid zoom in view URL.');
  if(typeof state.time!=='string'||!/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/.test(state.time))throw new Error('Invalid time in view URL.');
  const source=path=>{const i=datasets.findIndex(d=>d.path===path);if(i<0)throw new Error('A dataset saved in this link is unavailable.');return i;};
  const entries=state.panels.map(saved=>{
    if(saved.match!==undefined&&typeof saved.match!=='boolean')throw new Error('Invalid Match setting in view URL.');
    let ds,computed;
    if(saved.difference){
      const c=saved.difference;
      if(!['datasets','previous'].includes(c.mode)||!['exact','interval'].includes(c.alignment))throw new Error('Invalid difference in view URL.');
      computed={mode:c.mode,a:source(c.a),b:source(c.b),alignment:c.alignment};
      if(c.mode==='previous'){computed.b=computed.a;computed.alignment='exact';}
      computed.metadata=computedMetadata(computed);ds=computed.metadata;
    }else ds=datasets[source(saved.source)];
    const field=ds.fields.find(f=>f.id===saved.field);
    if(!field?.levels.some(l=>l.key===saved.depth))throw new Error('A saved field or depth is unavailable.');
    if(saved.range&&(!Number.isFinite(saved.range.min)||!Number.isFinite(saved.range.max)||saved.range.min>=saved.range.max||!Object.hasOwn(palettes,saved.range.palette)))throw new Error('Invalid color range in view URL.');
    return {saved,computed,source:computed?null:source(saved.source)};
  });
  if(state.calendar!==undefined&&!entries.some(e=>(e.computed?.metadata||datasets[e.source]).calendar===state.calendar))throw new Error('Invalid calendar in view URL.');
  return {state,entries};
}
function restorePanels(plan){
  stop();clearTimeout(scrubTimer);clearTimeout(zoomTimer);clearTimeout(urlTimer);++revision;
  if($('difference-dialog').open)$('difference-dialog').close();
  differencePanel=null;panels.splice(0);$('panels').replaceChildren();
  const state=plan?.state;
  bounds=state?[...state.bounds]:[-180,-90,180,90];
  $('map-columns').value=state?String(state.columns):'2';$('time-link').value=state?.alignment||'interval';
  for(const [id,value] of [['stride',state?.step||'1'],['speed',state?.speed||'1']]){
    if([...$(id).options].some(o=>o.value===value))$(id).value=value;
  }
  $('loop').checked=state?.loop!==false;
  times=[];timeMaps=[];current=requested=0;activeCalendar=state?.calendar||null;calendarDates.clear();
  for(let i=0;i<(plan?.entries.length||4);i++){
    const p=makePanel(i),entry=plan?.entries[i];panels.push(p);
    if(entry){
      if(entry.computed)configureDifference(p,entry.computed);else p.dataset.value=p.selection=String(entry.source);
      populateFields(p,entry.saved.field);p.depth.value=entry.saved.depth;
      p.match.checked=entry.saved.match!==false;
      if(entry.saved.range)p.range={...entry.saved.range,note:'Saved range · fixed during playback'};
    }else populateFields(p,i<2?'vo':'uo');
  }
  refreshLayout();rebuildTimeline();
  if(state){const index=times.findIndex(t=>t>=state.time);requested=current=index<0?times.length-1:index;}
  redraw();
}
async function restoreUrl(){
  let plan,warning;
  try{
    const url=new URL(location.href),value=new URLSearchParams(url.hash.slice(1)).get('view')||url.searchParams.get('view');
    if(value){if(value.length>24000)throw new Error('View URL is too large.');plan=statePlan(JSON.parse(value));}
  }catch(error){warning=`Could not restore saved view: ${error.message} Opened the default view.`;}
  restorePanels(plan);const ok=await show(requested);
  if(warning&&ok)status(warning,true);
}
$('add-map').onclick=()=>{
  if(panels.length>=MAX_MAPS)return;
  stop();const saved=panelState(panels.at(-1));const p=makePanel(panels.length);panels.push(p);
  if(saved.difference){const c=panels.at(-2).computed;configureDifference(p,{...c});}else p.dataset.value=p.selection=String(datasets.findIndex(d=>d.path===saved.source));
  populateFields(p,saved.field);p.depth.value=saved.depth;p.match.checked=saved.match!==false;if(saved.range)p.range={...saved.range,note:'Saved range · fixed during playback'};
  refreshLayout();rebuildTimeline();show(requested);
};
$('map-columns').onchange=()=>{stop();refreshLayout();redraw();show(requested);};
for(const id of ['stride','speed','loop'])$(id).addEventListener('change',updateUrl);
window.addEventListener('popstate',()=>{if(datasets.length)restoreUrl();});
function subtractFrames(a,b){
  const values=new Float32Array(a.length);
  for(let i=0;i<values.length;i++)values[i]=a[i]===-Infinity||b[i]===-Infinity?-Infinity:Number.isFinite(a[i])&&Number.isFinite(b[i])?a[i]-b[i]:NaN;
  return values;
}
function differenceLimits(values){
  const finite=values.filter(Number.isFinite).map(Math.abs).sort();
  const high=finite.length?Math.max(finite[Math.floor((finite.length-1)*.98)],1e-12):1;
  return {min:-high,max:high,palette:'balance',reference:'first available difference · fixed during playback'};
}
function panelSpec(p,time,calendar){
  const ds=panelSources(p)[0],key=p.depth.value,index=timeMaps[ds].get(`${calendar}/${time}`);
  const spec={ds,key,index,width:Math.min(1440,Math.max(1024,Math.round(p.canvas.clientWidth*2))),height:0};
  if(index===undefined||!isComputed(p))return spec;
  const c=p.computed;
  const other=c.mode==='previous'?index-1:(c.alignment==='exact'?exactMaps[c.b].get(`${calendar}/${datasets[ds].times[index]}`):intervalMaps[c.b].get(`${calendar}/${datasets[ds].interval_end_times?.[index]||datasets[ds].times[index]}`));
  if(other===undefined||other<0){
    spec.index=undefined;
    spec.problem=c.mode==='previous'?{message:'No previous frame',detail:'This is the first recorded frame in this series.'}:{message:'No matching reference timestamp',detail:`Reference: ${datasets[c.b].times[0].slice(0,10)} – ${datasets[c.b].times.at(-1).slice(0,10)}`};
  }else spec.other={ds:c.b,index:other};
  return spec;
}
async function panelFrame(spec,extent){
  const first=frame(spec.ds,spec.key,spec.index,extent,spec.width,spec.height);
  if(!spec.other)return first;
  const [a,b]=await Promise.all([first,frame(spec.other.ds,spec.key,spec.other.index,extent,spec.width,spec.height)]);
  return subtractFrames(a,b);
}
function rememberCalendar(){
  const time=times[requested];
  if(!time)return;
  const source=datasets.findIndex((d,i)=>d.calendar===activeCalendar&&timeMaps[i]?.has(`${activeCalendar}/${time}`));
  calendarDates.set(activeCalendar,{time,source,index:timeMaps[source]?.get(`${activeCalendar}/${time}`)});
}
function rememberedDate(calendar){
  const saved=calendarDates.get(calendar);
  if(!saved)return null;
  return saved.source>=0?linkedTimes(datasets[saved.source])[saved.index]||saved.time:saved.time;
}
function switchCalendar(calendar){
  if(calendar===activeCalendar)return;
  stop();clearTimeout(scrubTimer);clearTimeout(zoomTimer);
  rememberCalendar();
  activeCalendar=calendar;
  rebuildTimeline(rememberedDate(calendar));show(requested);
}
function rebuildTimeline(targetDate){
  const oldDate=times[requested],selected=[...new Set(panels.flatMap(panelSources))];
  const calendars=[...new Set(selected.map(ds=>datasets[ds].calendar))];
  if(!calendars.includes(activeCalendar)){
    rememberCalendar();
    activeCalendar=calendars[0];targetDate=rememberedDate(activeCalendar);
  }
  const reference=selected.find(ds=>datasets[ds].calendar===activeCalendar),first=datasets[reference];
  const matchedIndex=timeMaps[reference]?.get(`${activeCalendar}/${oldDate}`);
  timeMaps=datasets.map(d=>new Map(linkedTimes(d).map((t,i)=>[`${d.calendar}/${t}`,i])));
  times=[...new Set(selected.filter(ds=>datasets[ds].calendar===activeCalendar).flatMap(ds=>linkedTimes(datasets[ds])))].sort();
  const target=targetDate!==undefined?targetDate:matchedIndex!==undefined?linkedTimes(first)[matchedIndex]:oldDate;
  const dateIndex=target?times.findIndex(t=>t>=target):0;
  requested=current=dateIndex<0?Math.max(0,times.length-1):dateIndex;
  $('timeline').max=Math.max(0,times.length-1);$('timeline').value=requested;
  $('play').disabled=!times.length;
  $('start').textContent=times[0]?.slice(0,10)||'No dates';$('end').textContent=times.at(-1)?.slice(0,10)||'';
  $('jump').min=times[0]?.slice(0,10)||'';$('jump').max=times.at(-1)?.slice(0,10)||'';
  const interval=$('time-link').value==='interval';
  $('timeline-kind').textContent=interval?'Interval ending':'Timestamp';
  $('coverage').textContent=`${times.length.toLocaleString()} ${interval?'five-day intervals':'timestamps'} across selected datasets · ${activeCalendar} calendar · all maps linked`;
  $('calendar').replaceChildren(...calendars.map(c=>option(c,c==='noleap'?'noleap (365-day)':c)));
  $('calendar').value=activeCalendar;$('calendar').disabled=calendars.length<2;
}
function unavailable(p,time,calendar,problem){
  const source=info(p),dates=linkedTimes(source);
  p.values=null;p.shownDate=null;
  p.emptyMessage=source.calendar!==calendar?'Different calendar':time<dates[0]||time>dates.at(-1)?'Outside available dates':'No frame at this timestamp';
  p.emptyDetail=`Available: ${dates[0]?.slice(0,10)} – ${dates.at(-1)?.slice(0,10)}`;
  if(problem){p.emptyMessage=problem.message;p.emptyDetail=problem.detail;}
  p.root.querySelector('.source-time').textContent=`${time.slice(0,16)} · no data`;
  p.root.querySelector('.readout').textContent=p.emptyMessage;
  p.root.classList.remove('loading');render(p);
}
async function show(index){
  const myRevision=++revision;
  if(!times.length){$('date').textContent='—';$('frame-label').textContent='';status('No dates available for the selected calendar.',true);panels.forEach(p=>{p.values=null;p.emptyMessage='No dates available';p.emptyDetail='';render(p);});return false;}
  requested=Math.max(0,Math.min(times.length-1,Math.round(index)));
  updateUrl();
  const time=times[requested],extent=[...bounds];
  const calendar=activeCalendar;
  const specs=panels.map(p=>panelSpec(p,time,calendar));
  for(const spec of specs)spec.height=Math.round(spec.width/2);
  $('timeline').value=requested;status(`Loading ${time} · native OSN reads can take a few seconds…`);
  $('tooltip').style.display='none';
  panels.forEach((p,i)=>{
    p.units=info(p).fields.flatMap(f=>f.levels).find(l=>l.key===specs[i].key).units;
    if(specs[i].index===undefined)unavailable(p,time,calendar,specs[i].problem);
    else {p.root.querySelector('.empty-state').hidden=true;p.root.classList.add('loading');}
  });
  try{
    const rangePromises=specs.map((s,i)=>s.index===undefined||s.other?null:panels[i].range||limits(s.ds,s.key));
    const [frames,ranges]=await Promise.all([
      Promise.all(specs.map(s=>s.index===undefined?null:panelFrame(s,extent))),Promise.all(rangePromises)
    ]);
    if(myRevision!==revision)return false;
    current=requested;
    const candidates=specs.map((s,i)=>s.index===undefined?null:panels[i].range||(s.other?differenceLimits(frames[i]):ranges[i]));
    matchRanges(candidates);
    panels.forEach((p,i)=>{
      if(specs[i].index===undefined){if(p.range){p.root.querySelector('.range-note').textContent=rangeNote(p);render(p);}return;}
      p.emptyMessage=null;p.emptyDetail='';
      p.values=frames[i];p.extent=extent;p.width=specs[i].width;p.height=specs[i].height;p.shownDate=info(p).times[specs[i].index];
      p.root.querySelector('.readout').textContent='Hover for values · scroll to zoom · drag to pan';
      p.root.querySelector('.range-note').textContent=rangeNote(p);
      const sampling=info(p).fields.flatMap(f=>f.levels).find(l=>l.key===specs[i].key).sampling||info(p).sampling||'five-day mean';
      p.root.querySelector('.source-time').textContent=`${p.shownDate.slice(0,16)} · ${sampling}`;
      if(specs[i].other){const other=specs[i].other;p.root.querySelector('.source-time').textContent=`A: ${p.shownDate.slice(0,16)} − B: ${datasets[other.ds].times[other.index].slice(0,16)} · difference`;}
      p.units=info(p).fields.flatMap(f=>f.levels).find(l=>l.key===specs[i].key).units;
      colorize(p);render(p);p.root.classList.remove('loading');
    });
    $('date').textContent=time.slice(0,10);$('date').dateTime=time.slice(0,10);$('jump').value=time.slice(0,10);
    $('frame-label').textContent=`${current+1} / ${times.length}`;$('play').disabled=false;
    updateUrl();
    const missing=specs.filter(s=>s.index===undefined).length;
    status(`${$('time-link').value==='interval'?'Same five-day interval; actual source times shown above each map':'Exact timestamps linked'}${missing?` · ${missing} panel${missing===1?'':'s'} without data on this date`:''} · checked Match panels share unit + absolute/delta scales · gray: land · white: missing`);
    return true;
  }catch(error){if(myRevision===revision){status(`Could not load frame: ${error.message}`,true);panels.forEach(p=>p.root.classList.remove('loading'));stop();}return false;}
}
function colorize(p){
  if(!p.values||!p.range)return;
  const {min,max,palette}=p.range;p.image.width=p.width;p.image.height=p.height;
  const ctx=p.image.getContext('2d'),image=ctx.createImageData(p.width,p.height),colors=lookup[palette];
  for(let i=0;i<p.values.length;i++){
    const v=p.values[i],rgb=v===-Infinity?[57,72,84]:!Number.isFinite(v)?[243,244,239]:colors[Math.max(0,Math.min(255,Math.round((v-min)/(max-min)*255)))];
    const offset=i*4;image.data[offset]=rgb[0];image.data[offset+1]=rgb[1];image.data[offset+2]=rgb[2];image.data[offset+3]=255;
  }
  ctx.putImageData(image,0,0);
}
const lonLabel=v=>{const original=v;v=((v+180)%360+360)%360-180;if(v===-180&&original>0)v=180;return `${fmt(Math.abs(v))}°${v<0?'W':v>0?'E':''}`;};
const latLabel=v=>`${fmt(Math.abs(v))}°${v<0?'S':v>0?'N':''}`;
function render(p){
  const ctx=p.canvas.getContext('2d'),W=p.canvas.width,H=p.canvas.height,[west,south,east,north]=bounds;
  const outside=!p.values&&p.emptyMessage==='Outside available dates',differentCalendar=!p.values&&p.emptyMessage==='Different calendar',empty=p.root.querySelector('.empty-state');
  empty.hidden=!(outside||differentCalendar);
  empty.querySelector('.empty-message').textContent=p.emptyMessage||'';
  empty.querySelector('.available-dates').hidden=!outside;
  const description=empty.querySelector('.calendar-description'),switchButton=empty.querySelector('.switch-calendar');
  description.hidden=switchButton.hidden=!differentCalendar;
  if(differentCalendar){
    const source=info(p),dates=linkedTimes(source);
    description.textContent=`${dates[0].slice(0,10)} – ${dates.at(-1).slice(0,10)} · ${source.calendar}`;
    switchButton.textContent=`Switch to ${source.calendar} calendar`;
  }
  if(outside){
    const dates=linkedTimes(info(p));
    for(const [boundary,time] of [['first',dates[0]],['last',dates.at(-1)]]){
      const button=empty.querySelector(`[data-boundary="${boundary}"]`);
      button.textContent=time.slice(0,10);button.dataset.time=time;button.title=`Jump to ${time}`;
      button.setAttribute('aria-label',`Jump to ${boundary} available date: ${time.slice(0,10)}`);
      button.disabled=!times.includes(time);
    }
  }
  ctx.fillStyle='#394854';ctx.fillRect(0,0,W,H);ctx.imageSmoothingEnabled=false;
  if(p.values&&p.extent){
    const b=p.extent;
    const scaleX=W/(east-west),scaleY=H/(north-south);
    ctx.drawImage(p.image,(b[0]-west)*scaleX,(north-b[3])*scaleY,(b[2]-b[0])*scaleX,(b[3]-b[1])*scaleY);
  }
  ctx.strokeStyle='#d0e9f21e';ctx.lineWidth=1;ctx.beginPath();
  for(const f of [.25,.5,.75]){ctx.moveTo(f*W,0);ctx.lineTo(f*W,H);ctx.moveTo(0,f*H);ctx.lineTo(W,f*H);}ctx.stroke();
  if(!p.values&&p.emptyMessage&&!outside&&!differentCalendar){
    ctx.textAlign='center';ctx.fillStyle='#e9f1f5';ctx.font='28px system-ui';ctx.fillText(p.emptyMessage,W/2,H/2-10);
    ctx.fillStyle='#c1d0d9';ctx.font='22px system-ui';ctx.fillText(p.emptyDetail||'',W/2,H/2+28);
  }
  const xs=Array.from({length:5},(_,i)=>west+(east-west)*i/4),ys=Array.from({length:3},(_,i)=>north-(north-south)*(i+1)/4);
  p.root.querySelector('.lonlabels').replaceChildren(...xs.map(v=>{const s=document.createElement('span');s.textContent=lonLabel(v);return s;}));
  p.root.querySelector('.latlabels').replaceChildren(...ys.map((v,i)=>{const s=document.createElement('span');s.style.top=`${25*(i+1)}%`;s.textContent=latLabel(v);return s;}));
  p.canvas.dataset.bounds=bounds.join(',');
  p.canvas.setAttribute('aria-label',`${info(p).name}, ${p.field.selectedOptions[0]?.textContent}, ${p.depth.selectedOptions[0]?.textContent}, ${p.values?p.shownDate:p.emptyMessage||'no matching data'}, ${p.units}`);
  p.root.querySelector('.legend').style.visibility=p.values?'visible':'hidden';
  p.min.disabled=p.max.disabled=!p.values;
  p.root.querySelector('.rescale').disabled=p.root.querySelector('.reset').disabled=!p.values;
  if(!p.range)return;
  const {min,max,palette}=p.range;
  p.root.querySelector('.colorbar').style.background=`linear-gradient(to right,${palettes[palette].join(',')})`;
  const ticks=p.root.querySelectorAll('.ticks span');ticks[0].textContent=fmt(min);ticks[1].textContent=p.units;ticks[2].textContent=fmt(max);
  p.min.value=Number(min.toPrecision(5));p.max.value=Number(max.toPrecision(5));
}
function redraw(){panels.forEach(render);$('zoom-level').textContent=`${fmt(360/(bounds[2]-bounds[0]))}× zoom`;scheduleUrlUpdate();}
function normalizedBounds(west,south,span){
  span=Math.max(.5,Math.min(360,span));const height=span/2;
  south=Math.max(-90,Math.min(90-height,south));
  return [west,south,west+span,south+height];
}
function zoom(factor,x=.5,y=.5){
  const [west,south,east,north]=bounds,span=east-west,newSpan=Math.max(.5,Math.min(360,span*factor));
  const anchorX=west+x*span,anchorY=north-y*(north-south);
  bounds=normalizedBounds(anchorX-x*newSpan,anchorY-(1-y)*newSpan/2,newSpan);
  stop();redraw();clearTimeout(zoomTimer);zoomTimer=setTimeout(()=>show(requested),180);
}
function attachNavigation(p){
  let drag=null;
  p.canvas.addEventListener('wheel',e=>{e.preventDefault();const r=p.canvas.getBoundingClientRect();zoom(Math.exp(Math.sign(e.deltaY)*.3),(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height);},{passive:false});
  p.canvas.addEventListener('pointerdown',e=>{if(e.button!==0)return;stop();clearTimeout(zoomTimer);drag={x:e.clientX,y:e.clientY,b:[...bounds]};p.canvas.setPointerCapture(e.pointerId);p.canvas.style.cursor='grabbing';});
  p.canvas.addEventListener('pointermove',e=>{if(!drag)return;const r=p.canvas.getBoundingClientRect(),b=drag.b;bounds=normalizedBounds(b[0]-(e.clientX-drag.x)/r.width*(b[2]-b[0]),b[1]+(e.clientY-drag.y)/r.height*(b[3]-b[1]),b[2]-b[0]);redraw();});
  const end=()=>{if(!drag)return;const changed=Math.abs(drag.b[0]-bounds[0])+Math.abs(drag.b[1]-bounds[1])>1e-8;drag=null;p.canvas.style.cursor='grab';if(changed)show(requested);};
  p.canvas.addEventListener('pointerup',end);p.canvas.addEventListener('pointercancel',end);
  p.canvas.addEventListener('dblclick',e=>{const r=p.canvas.getBoundingClientRect();zoom(.5,(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height);});
}
function hover(p,event){
  if(!p.values||!p.extent)return;
  const r=p.canvas.getBoundingClientRect(),lon=bounds[0]+(event.clientX-r.left)/r.width*(bounds[2]-bounds[0]),lat=bounds[3]-(event.clientY-r.top)/r.height*(bounds[3]-bounds[1]);
  const x=Math.floor((lon-p.extent[0])/(p.extent[2]-p.extent[0])*p.width),y=Math.floor((p.extent[3]-lat)/(p.extent[3]-p.extent[1])*p.height);
  if(x<0||x>=p.width||y<0||y>=p.height){$('tooltip').style.display='none';return;}
  const v=p.values[y*p.width+x];
  const text=`${latLabel(lat)} · ${lonLabel(lon)}\n${v===-Infinity?'Land / below seafloor':!Number.isFinite(v)?'Missing ocean data':`${fmt(v)} ${p.units}`}`;
  p.root.querySelector('.readout').textContent=text.replace('\n',' · ');
  $('tooltip').textContent=text;$('tooltip').style.display='block';$('tooltip').style.left=`${Math.min(event.clientX+14,innerWidth-240)}px`;$('tooltip').style.top=`${event.clientY+16}px`;
}
$('time-link').onchange=()=>{stop();rebuildTimeline();show(requested);};
$('calendar').onchange=()=>switchCalendar($('calendar').value);
$('play').onclick=play;
$('previous').onclick=()=>{stop();show(requested-Number($('stride').value));};
$('next').onclick=()=>{stop();show(requested+Number($('stride').value));};
$('zoom-in').onclick=()=>zoom(.5);$('zoom-out').onclick=()=>zoom(2);
$('zoom-reset').onclick=()=>{stop();bounds=[-180,-90,180,90];redraw();show(requested);};
let scrubTimer;
$('timeline').oninput=()=>{stop();clearTimeout(scrubTimer);scrubTimer=setTimeout(()=>show(Number($('timeline').value)),100);};
$('jump').onchange=()=>{const date=$('jump').value;if(!date)return;stop();const i=times.findIndex(t=>t.slice(0,10)>=date);show(i<0?times.length-1:i);};
document.addEventListener('keydown',e=>{
  const target=document.activeElement;
  if(e.defaultPrevented||e.isComposing||e.altKey||e.ctrlKey||e.metaKey)return;
  if($('difference-dialog').open)return;
  if(target.isContentEditable||target.matches('select,textarea,input:not(#timeline)'))return;
  if(e.key==='ArrowLeft'||e.key==='ArrowRight'){
    e.preventDefault();clearTimeout(scrubTimer);
    $(e.key==='ArrowLeft'?'previous':'next').click();
    return;
  }
  // Focused buttons retain their native Space activation.
  if(target.tagName==='BUTTON')return;
  if(e.code==='Space'){e.preventDefault();play();}
  if(e.key==='+')zoom(.5);
  if(e.key==='-')zoom(2);
});
async function init(){
  try{
    datasets=(await request('/api/metadata')).datasets;
    exactMaps=datasets.map(d=>new Map(d.times.map((t,i)=>[`${d.calendar}/${t}`,i])));
    intervalMaps=datasets.map(d=>new Map((d.interval_end_times||d.times).map((t,i)=>[`${d.calendar}/${t}`,i])));
    await restoreUrl();
  }catch(error){status(`Could not open dataset: ${error.message}`,true);}
}
init();
