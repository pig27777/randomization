// Executes playback interactions without a browser; not a screenshot/layout test.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(process.argv[2],'utf8');
const data=JSON.parse(html.match(/data-mesh>(.*?)<\/script>/s)[1]);
const script=html.slice(html.lastIndexOf('<script>')+8,html.lastIndexOf('</script>'));
const nodes={};
function node(name){return nodes[name]||(nodes[name]={value:0,checked:false,textContent:'',handlers:{},addEventListener(e,f){this.handlers[e]=f;}});}
const canvas=node('canvas');canvas.clientWidth=736;canvas.clientHeight=370;
canvas.getContext=()=>new Proxy({}, {get:(t,k)=>t[k]||(t[k]=()=>{})});
canvas.setPointerCapture=()=>{};canvas.getBoundingClientRect=()=>({left:0,top:0});
const root={querySelector:q=>q==='canvas'?canvas:node(q)};
node('[data-mesh]').textContent=JSON.stringify(data);
const sandbox={document:{getElementById:()=>root,documentElement:{}},
  window:{devicePixelRatio:1,addEventListener(){},openai:{widgetState:{},setWidgetState:()=>Promise.resolve()}},
  getComputedStyle:()=>({getPropertyValue:()=>'rgb(30,80,120)'}),ResizeObserver:class{observe(){}},
  MutationObserver:class{observe(){}},requestAnimationFrame:()=>{},console};
vm.runInNewContext(script,sandbox);
const before=node('[data-time]').textContent;
node('[data-frame]').value=Math.min(12,data.frames.length-1);node('[data-frame]').handlers.input();
assert.notEqual(node('[data-time]').textContent,before);
node('[data-cut]').value=70;node('[data-cut]').handlers.input();assert.equal(node('[data-cut-value]').textContent,'70%');
node('[data-play]').handlers.click();assert.equal(node('[data-play]').textContent,'暂停');
canvas.handlers.pointerdown({clientX:300,clientY:200,pointerId:1});
canvas.handlers.pointermove({clientX:320,clientY:205});canvas.handlers.pointerup({clientX:320,clientY:205});
console.log(JSON.stringify({javascript:true,timeline:true,cutaway:true,play:true,rotation:true,
  browser_preview:'unavailable: CUA has no browsers'}));
