// DOM/event contract checks using actual API fixtures; no browser-layout claim.
const fs=require('fs'),vm=require('vm'),assert=require('assert'),path=require('path');
const output=process.argv[2],html=fs.readFileSync(path.join(__dirname,'grasp_ui.html'),'utf8');
const fixtures=JSON.parse(fs.readFileSync(path.join(output,'ui-fixtures.json'),'utf8'));
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
const flush=async()=>{for(let i=0;i<50;i++)await Promise.resolve();};
(async()=>{
 const results=[];
 for(const width of [1100,360]){
  function element(dataset={},id=''){
   let value='';return {dataset,hidden:false,disabled:false,textContent:'',children:[],listeners:{},
    classList:{toggle(){},add(){},remove(){}},
    get value(){return value;},set value(v){value=String(['young','yield_strain','density'].includes(id)?Math.round(Number(v)):v);},
    addEventListener(k,fn){this.listeners[k]=fn;},append(e){this.children.push(e);},replaceChildren(){this.children=[];},
    getAttribute(k){return this[k]||null;}};
  }
  const ids=Object.fromEntries([...html.matchAll(/\bid="([^"]+)"/g)].map(m=>[m[1],element({},m[1])]));
  const materials=['hard','bendable','compressible'].map(material=>element({material}));
  const shapes=['box','rounded','pillow','tapered'].map(shape=>element({shape}));
  const requests=[],timers=new Map();let next=0;
  const base=fixtures.previews.box;
  async function fetch(url,options){
   if(options?.body){const body=JSON.parse(options.body);requests.push(body);return {ok:true,json:async()=>({id:'test-job',mode:body.mode,status:'running',settings:body.settings})};}
   const data=url==='/api/meta'?fixtures.meta:url==='/api/history'?[base]:base;
   return {ok:true,json:async()=>data};
  }
  const document={getElementById(id){assert(ids[id],id);return ids[id];},createElement:()=>element(),
   querySelectorAll(q){return q==='[data-material]'?materials:shapes;}};
  vm.runInNewContext(script,{document,window:{innerWidth:width},fetch,
   setTimeout(fn){timers.set(++next,fn);return next;},clearTimeout(id){timers.delete(id);}});
  await flush();
  assert(ids.frame.src?.includes('live.png'),JSON.stringify({status:ids.status.textContent,error:ids['input-error'].textContent}));
  assert(ids['result-title'].textContent.includes('初始模型'));
  await ids.run.listeners.click();await flush();
  assert.equal(requests.at(-1).mode,'run');
  assert.equal(requests.at(-1).settings.young,100000,'untouched log slider preserves exact preset');
  assert.equal(requests.at(-1).settings.density,fixtures.meta.default.density,'untouched integer slider preserves exact preset');
  assert(ids.run.disabled&&ids.preview.disabled);
  shapes[2].listeners.click();await flush();
  ids.roundness.value='.35';ids.roundness.listeners.input();
  ids.width.value='180';ids.width.listeners.input();
  ids.young.value='0';ids.young.listeners.input();
  await ids.preview.listeners.click();await flush();
  let settings=requests.at(-1).settings;
  assert.equal(settings.shape,'pillow');assert.equal(settings.roundness,.35);assert.equal(settings.bulge,.6);
  assert.equal(settings.size[1],.18);assert.equal(settings.young,fixtures.meta.ranges.young[0]);
  ids.young.value='1000';ids.young.listeners.input();
  materials[2].listeners.click();await flush();
  await ids.run.listeners.click();await flush();
  settings=requests.at(-1).settings;
  assert.equal(settings.material,'compressible');assert.equal(settings.young,3000);
  assert.equal(settings.shape,'pillow');assert.equal(settings.size[1],.18);
  assert(!ids['input-error'].textContent);
  results.push({width,live_frame:true,exact_defaults:true,shape_and_material_independent:true,
   physical_parameter_requests:true,busy_controls:true,history:true});
 }
 const result={browser_test:false,checks:results};
 fs.writeFileSync(path.join(output,'ui-dom-check.json'),JSON.stringify(result,null,2));
 console.log(JSON.stringify(result));
})().catch(error=>{console.error(error);process.exitCode=1;});
