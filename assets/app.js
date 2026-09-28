/* WaveSight console: demo library + upload + wave processing animation + view/download */
(function(){
"use strict";
const $=id=>document.getElementById(id);
const logEl=$('log'), pfill=$('pfill'), stages=[...document.querySelectorAll('.st')];
let BASE='', backendOK=false, current=null, currentSigmf=null, currentFile='';
let selDemo=null, upFile=null, rafMode='idle', rafT=0, energy=0.25, detTones=[];

/* ---------- curated demo metadata ---------- */
const META={
 'BPSK_10dB_cf32.iq':['BPSK · 10 dB','Digital','Textbook PSK @10 kHz, 2 kbaud, ~10 dB.'],
 'QPSK_10dB_cf32.iq':['QPSK · 10 dB','Digital','4-phase PSK, 2 kbaud, tight constellation.'],
 'QAM16_10dB_cf32.iq':['QAM16 · 10 dB','Digital','16-point constellation, amplitude+phase.'],
 'FSK2_10dB_cf32.iq':['FSK2 · 10 dB','Digital','Binary FSK, 800 Hz deviation, 1 kbaud.'],
 'FM_10dB_cf32.iq':['FM · 10 dB','Analog','Wideband FM, 3 kHz deviation signature.'],
 'AM_10dB_cf32.iq':['AM · 10 dB','Analog','Double-sideband AM, 100 Hz message tone.'],
 'SSB_10dB_cf32.iq':['SSB · 10 dB','Analog','Upper-sideband only, razor-narrow.'],
 'CW_10dB_cf32.iq':['CW · 10 dB','Analog','5 Hz on/off keying, thin spectral line.'],
 'OFDM_10dB_cf32.iq':['OFDM · 10 dB','Wideband','64 subcarriers, flat-top wide spectrum.'],
 'CHIRP_10dB_cf32.iq':['CHIRP · 10 dB','Wideband','20 kHz linear sweep, diagonal waterfall.'],
 'MULTI_10dB_cf32.iq':['MULTI ×3','Multi','BPSK + CW + FM in ONE capture — 3 boxes.'],
 'NOISE_cf32.iq':['Pure noise','Edge','No signal — must refuse, zero labels.'],
 'BPSK_5dB_cf32.iq':['BPSK · 5 dB','SNR sweep','Rated floor — still 100% family accuracy.'],
 'BPSK_0dB_cf32.iq':['BPSK · 0 dB','SNR sweep','Low-SNR edge — graceful degradation zone.'],
 'BPSK_-5dB_cf32.iq':['BPSK · −5 dB','SNR sweep','Below floor — explicit UNKNOWN refusal.'],
 'QPSK_5dB_cf32.iq':['QPSK · 5 dB','SNR sweep','Rated floor for 4-phase modes.'],
};
function metaOf(f){
  const m=META[f];
  if(m) return {file:f,title:m[0],cat:m[1],desc:m[2]};
  return {file:f,title:f,cat:'Demo',desc:'Pre-embedded synthetic capture.'};
}

/* ---------- backend probe ---------- */
async function probe(){
  const cand=[ '', 'http://localhost:8000', 'http://127.0.0.1:8000' ];
  for(const b of cand){
    try{
      const r=await fetch(b+'/health',{signal:AbortSignal.timeout(2500)});
      if(r.ok){ BASE=b; backendOK=true;
        $('bstat').textContent='backend ● '+ (b||'same-origin');
        $('bend').textContent=(b||'same-origin');
        $('beacon').classList.add('ok');
        log('› backend connected: '+(b||'same-origin'));
        return;
      }
    }catch(e){}
  }
  $('bstat').textContent='offline — embedded demos';
  log('› no backend — embedded offline demo mode (16 precomputed reports).');
}
function log(s){ logEl.innerHTML+='› '+s+'<br>'; logEl.scrollTop=logEl.scrollHeight; }

/* ---------- demo list ---------- */
async function loadDemos(){
  let files=[];
  if(backendOK){
    try{
      const r=await fetch(BASE+'/api/demos'); const j=await r.json();
      files=(j.demos||[]).map(d=>d.file);
    }catch(e){}
  }
  if(!files.length && window.DEMO_REPORTS) files=Object.keys(window.DEMO_REPORTS);
  if(!files.length) files=Object.keys(META);
  files.sort((a,b)=>{
    const ord=f=>f.includes('MULTI')?0:f.includes('NOISE')?99:(META[f]?1:50);
    return ord(a)-ord(b)||a.localeCompare(b);
  });
  renderDemos(files);
  selDemo=files.includes('MULTI_10dB_cf32.iq')?'MULTI_10dB_cf32.iq':files[0];
  markSel();
}
function renderDemos(files){
  const q=($('q').value||'').toLowerCase();
  const box=$('dlist'); box.innerHTML='';
  files.filter(f=>!q||f.toLowerCase().includes(q)||(META[f]&&(META[f][0]+META[f][1]).toLowerCase().includes(q)))
  .forEach(f=>{
    const m=metaOf(f), d=document.createElement('div');
    d.className='ditem'+(f===selDemo?' on':''); d.dataset.f=f;
    d.innerHTML=`<b>${m.title}</b><small>${m.cat} · ${f}</small><small style="opacity:.8">${m.desc}</small>`;
    d.onclick=()=>{ selDemo=f; markSel(); log('selected demo: '+f); };
    box.appendChild(d);
  });
}
function markSel(){ document.querySelectorAll('.ditem').forEach(el=>el.classList.toggle('on',el.dataset.f===selDemo)); }

/* ---------- tabs / upload ---------- */
$('tabDemo').onclick=()=>{ $('tabDemo').classList.add('on'); $('tabUp').classList.remove('on'); $('paneDemo').style.display=''; $('paneUp').style.display='none'; };
$('tabUp').onclick=()=>{ $('tabUp').classList.add('on'); $('tabDemo').classList.remove('on'); $('paneUp').style.display=''; $('paneDemo').style.display='none'; };
$('drop').onclick=()=>$('file').click();
$('file').onchange=e=>{ upFile=e.target.files[0]; $('fname').textContent=upFile?('📄 '+upFile.name+' ('+Math.round(upFile.size/1024)+' KB)') : ''; };
$('drop').ondragover=e=>e.preventDefault();
$('drop').ondrop=e=>{ e.preventDefault(); upFile=e.dataTransfer.files[0]; $('fname').textContent=upFile?('📄 '+upFile.name):''; };
$('q').oninput=loadDemos;

/* ---------- processing wave animation ---------- */
const cv=$('procWave'), cx=cv.getContext('2d');
function waveFrame(){
  rafT+=.045;
  const W=cv.width,H=cv.height, mid=H/2;
  const gr=cx.createLinearGradient(0,0,0,H); gr.addColorStop(0,'#0b1330'); gr.addColorStop(1,'#060b1a');
  cx.fillStyle=gr; cx.fillRect(0,0,W,H);
  cx.strokeStyle='rgba(120,180,255,.1)';
  for(let y=0;y<H;y+=20){ cx.beginPath(); cx.moveTo(0,y); cx.lineTo(W,y); cx.stroke(); }
  const amp = rafMode==='run' ? (26+energy*54) : 22;
  const spd = rafMode==='run' ? 1.6+energy*2.4 : .7;
  const cols=['#22d3ee','#8b5cf6','#f472b6','#a3e635'];
  const nW = rafMode==='run' ? 4 : 3;
  for(let k=0;k<nW;k++){
    cx.beginPath();
    for(let x=0;x<=W;x+=3){
      let y=mid;
      const tones = detTones.length&&rafMode!=='run' ? detTones : [3+k*2, 7-k];
      tones.forEach((f,i)=>{ y += (amp/(i+2)) * Math.sin(x/W*Math.PI*2*f + rafT*spd*(1+i*.35) + k*1.7); });
      if(rafMode==='run') y += Math.sin(x*.05+rafT*3)*energy*14*(k%2?1:-1);
      x?cx.lineTo(x,y):cx.moveTo(x,y);
    }
    cx.strokeStyle=cols[k%4]; cx.lineWidth=2.2; cx.shadowColor=cols[k%4]; cx.shadowBlur=rafMode==='run'?14:7;
    cx.stroke(); cx.shadowBlur=0;
  }
  // scrolling waterfall dots while running
  if(rafMode==='run'){
    for(let i=0;i<70;i++){
      const x=Math.random()*W, y=Math.random()*H;
      cx.fillStyle=`rgba(34,211,238,${Math.random()*energy*.8})`;
      cx.fillRect(x,y,2,2);
    }
  }
  requestAnimationFrame(waveFrame);
}
function setStage(i){
  stages.forEach((s,k)=>{ s.classList.toggle('on',k===i); s.classList.toggle('done',k<i); });
}
function setProg(p){ pfill.style.width=(p*100).toFixed(0)+'%'; }

/* ---------- staged run ---------- */
const STEPS=[
 ['Ingesting capture — DC removal, RMS normalize, quality gates…',.18],
 ['Detecting signals — STFT + CFAR regions…',.4],
 ['Extracting parameters — center, bandwidth, SNR, symbol rate…',.62],
 ['Classifying modulation — classical discriminators + fusion…',.84],
 ['Writing report — JSON + SigMF + spectrogram…',1],
];
async function stagedRun(fn){
  rafMode='run'; setProg(0);
  for(let i=0;i<STEPS.length;i++){
    setStage(i); energy=.2+i*.18;
    log('['+(i+1)+'/5] '+STEPS[i][0]);
    setProg(STEPS[i][1]*.92);
    await new Promise(r=>setTimeout(r, backendOK?500:650));
    if(i===1&&fn&&fn.onDetect) fn.onDetect();
  }
  const out=await fn.run();
  setStage(5); setProg(1); rafMode='idle'; energy=.25;
  return out;
}

/* ---------- analyze ---------- */
$('go').onclick=async ()=>{
  const isUp=$('tabUp').classList.contains('on');
  logEl.innerHTML='';
  if(isUp){
    if(!upFile){ log('⚠ choose a file first.'); return; }
    if(!backendOK){ log('⚠ uploads need the backend. Run: uvicorn src.api.app:app --port 8000'); log('→ tip: use the Demo library while offline.'); return; }
    await stagedRun({run:async()=>{
      const fd=new FormData(); fd.append('file',upFile);
      const sr=$('sr').value||'100000';
      log('POST '+BASE+'/api/analyze @ '+sr+' Hz …');
      const r=await fetch(BASE+'/api/analyze?sample_rate_hz='+encodeURIComponent(sr),{method:'POST',body:fd});
      if(r.status===422){ const e=await r.json(); log('✖ rejected (422): '+(e.detail||'format error')); throw 0; }
      if(!r.ok) throw 0;
      const j=await r.json(); currentFile=upFile.name;
      current=j; currentSigmf=j.sigmf||null;
      finish(j, currentFile);
    }});
  }else{
    if(!selDemo){ log('⚠ pick a demo first.'); return; }
    const f=selDemo, sr=$('sr').value||'100000';
    try{
      await stagedRun({run:async()=>{
        if(backendOK){
          log('GET '+BASE+'/api/demo/'+f+' …');
          const r=await fetch(BASE+'/api/demo/'+encodeURIComponent(f));
          if(r.ok){ const j=await r.json(); current=j; currentSigmf=j.sigmf||null; currentFile=f; finish(j,f); return; }
          log('backend demo failed — falling back to embedded report.');
        }
        const emb=window.DEMO_REPORTS&&window.DEMO_REPORTS[f];
        if(!emb){ log('✖ no embedded report for '+f); throw 0; }
        current=emb.report; currentSigmf=emb.sigmf; currentFile=f;
        finish(emb.report,f);
      }});
    }catch(e){ log('✖ analysis failed.'); }
  }
};

/* ---------- results ---------- */
function badge(st){
  const s=(st||'unknown').toLowerCase();
  return `<span class="badge b-${s}">${s.toUpperCase()}</span>`;
}
function fmtHz(v){ if(v==null) return '—'; v=+v; if(v>=1e6) return (v/1e6).toFixed(2)+' MHz'; if(v>=1e3) return (v/1e3).toFixed(2)+' kHz'; return v.toFixed(1)+' Hz'; }
/* plain-English one-liner per finding, for first-time readers */
const MOD_PLAIN={PSK:'digital data carried in phase shifts',BPSK:'digital data carried in phase flips',QPSK:'digital data carried in 4 phase states',QAM:'digital data carried in amplitude + phase combos',FSK:'digital data carried in frequency hops',FM:'analog audio carried in frequency wiggles',AM:'analog audio carried in amplitude',SSB:'a narrow voice channel (single sideband)',CW:'an on/off keyed carrier, Morse-style',OFDM:'wideband data over many subcarriers (Wi-Fi-style)',CHIRP:'a sweeping tone, radar-style'};
/* click-to-expand guide for every field: WHAT it is + HOW this value was reached */
const FIELD_GUIDE={
 cf:{t:'Center frequency',what:'The middle of the signal\u2019s slice of spectrum — the dial position you\u2019d tune a receiver to in order to listen to it.'},
 bw:{t:'Bandwidth',what:'How wide a chunk of spectrum the signal occupies. Wider usually means more data — or more noise.'},
 snr:{t:'SNR',what:'Signal-to-noise ratio — how loudly the signal stands out above the background hiss. Higher is easier to read; near 0 dB it gets dicey.'},
 sr:{t:'Symbol rate',what:'How many data symbols per second the transmitter sends. Only meaningful for digital modes — analog and wideband signals show N/A by design.'},
 mod:{t:'Modulation',what:'The language the signal speaks — data encoded in phase, frequency or amplitude tricks. Decided by classical signal tests, sometimes confirmed by the ML model.'},
 fec:{t:'FEC / demod',what:'Whether the signal could be sliced into raw bits, and whether those bits look protected by an error-correcting code.'}
};
function plainSummary(d){
  const mod=d.modulation&&d.modulation.value||'UNKNOWN';
  const cf=d.center_freq_hz&&d.center_freq_hz.value!=null?fmtHz(d.center_freq_hz.value):'an unknown frequency';
  const bw=d.bandwidth_hz&&d.bandwidth_hz.value!=null?fmtHz(d.bandwidth_hz.value):null;
  const snr=d.snr_db&&d.snr_db.value!=null?+d.snr_db.value:null;
  if(mod==='UNKNOWN') return `A faint trace sits near <b>${cf}</b>, but it's too weak to identify — so the system refused to guess instead of risking a wrong label.`;
  const desc=MOD_PLAIN[mod]||'a signal of unknown type';
  const strength=snr==null?'':snr>=10?' It comes in <b>very strong</b>':snr>=5?' It comes in <b>clearly</b>':snr>=0?' It is <b>weak but readable</b>':' It is <b>barely above the noise</b>';
  return `Likely <b>${mod}</b> around <b>${cf}</b> — ${desc}${bw?`, about <b>${bw}</b> wide`:''}.${strength}${snr!=null?` (${snr.toFixed(1)} dB)`:''}.`;
}
function finish(rep, fname){
  current=rep; currentFile=fname;
  log('✔ done: '+(rep.detections||[]).length+' detection(s) in '+fname);
  detTones=(rep.detections||[]).map(d=>{
    const cf=d.center_freq_hz&&d.center_freq_hz.value?+d.center_freq_hz.value:8000;
    return Math.max(2,Math.min(12, Math.abs(cf)/4000));
  });
  $('results').style.display='';
  const dets=rep.detections||[];
  $('sumCards').innerHTML=`
    <div class="res-card"><small>Detections</small><b>${dets.length}</b></div>
    <div class="res-card"><small>Noise floor</small><b>${rep.noise_floor_db!=null?rep.noise_floor_db.toFixed(1)+' dB':'—'}</b></div>
    <div class="res-card"><small>Dynamic range</small><b>${rep.dynamic_range_db!=null?rep.dynamic_range_db.toFixed(1)+' dB':'—'}</b></div>
    <div class="res-card"><small>Pipeline</small><b>${rep.pipeline_version||'—'}</b></div>`;
  $('resFile').textContent=fname+' · '+fmtHz(rep.sample_rate_hz)+' sample rate';
  $('sigList').innerHTML = dets.length? dets.map((d,i)=>{
    const F=(key,o,fmt,extra)=>{ const g=FIELD_GUIDE[key]||{t:key,what:''};
      if(!o) return `<div class="fx"><small>${g.t}</small><span class="fv-na">—</span></div>`;
      const v=o.value==null?'—':(fmt?fmt(o.value):o.value);
      const pct=o.confidence!=null?`<em> · ${(o.confidence*100).toFixed(0)}%</em>`:'';
      return `<div class="fx"><details><summary><small>${g.t}</small><span class="fv">${v}${pct}</span></summary>
        <div class="fev">${badge(o.state)}<span><b>How:</b> ${o.evidence||''}</span></div>
        ${extra||''}<p class="fx-why">${g.what}</p></details></div>`; };
    return `<div class="sig"><div class="sig-head"><span class="sig-id">${d.id}</span>
      <span class="sig-mod">${d.modulation.value||'UNKNOWN'}</span>${badge(d.modulation.state)}
      <span class="sig-meta">${(d.t_end_s-d.t_start_s).toFixed(2)}s · ${fmtHz(d.f_low_hz)} – ${fmtHz(d.f_high_hz)}</span></div>
      <p class="sig-plain">${plainSummary(d)}</p>
      <div class="kv">
        ${F('cf',d.center_freq_hz,v=>fmtHz(v))}
        ${F('bw',d.bandwidth_hz,v=>fmtHz(v))}
        ${F('snr',d.snr_db,v=>(+v).toFixed(1)+' dB')}
        ${F('sr',d.symbol_rate_hz,v=>fmtHz(v))}
        ${F('mod',Object.assign({},d.modulation,{value:d.modulation.value||'UNKNOWN'}),null,`<div class="fx-method">Decided by: <b>${d.modulation_evidence_method||'classical'}</b> (classical tests${d.modulation_evidence_method==='both'?' + ML confirmation':d.modulation_evidence_method==='ml'?' overruled by ML':''})</div>`)}
        ${d.fec_identification?F('fec',d.fec_identification):`<div class="fx"><small>${FIELD_GUIDE.fec.t}</small><span class="fv-na">${d.demod_available?'bits recovered, no code found':'—'}</span></div>`}
      </div></div>`;
  }).join('') : `<div class="sig"><div class="sig-head"><span class="sig-mod">No detections</span>${badge('unknown')}</div>
    <p style="color:var(--mut);font-size:13px;margin-top:8px">Pure noise / below SNR floor — the system refused to guess. That honesty is the feature.</p></div>`;
  $('rawJson').textContent=JSON.stringify(rep,null,2);
  drawSpectrum(rep); drawConstellation(dets[0]&&dets[0].modulation.value);
  document.getElementById('results').scrollIntoView({behavior:'smooth',block:'start'});
}

/* spectrum sketch */
function drawSpectrum(rep){
  const c=$('specCanvas'), g=c.getContext('2d'), W=c.width,H=c.height;
  g.fillStyle='#070c1b'; g.fillRect(0,0,W,H);
  // noise floor
  g.strokeStyle='rgba(148,163,184,.5)'; g.beginPath();
  for(let x=0;x<W;x++){ const y=H-40+Math.random()*8; x?g.lineTo(x,y):g.moveTo(x,y);} g.stroke();
  const sr=rep.sample_rate_hz||100000, dets=rep.detections||[];
  const cols={detected:'#a3e635',estimated:'#22d3ee',inferred:'#fbbf24',unknown:'#f472b6'};
  if(!dets.length){ g.fillStyle='#f472b6'; g.font='13px monospace'; g.fillText('— noise floor only, no confident peaks —',20,H/2); return; }
  dets.forEach((d,i)=>{
    const f0=d.f_low_hz??0, f1=d.f_high_hz??sr/2;
    const x0=W/2+ (f0/(sr/2))*(W/2)*.9, x1=W/2+(f1/(sr/2))*(W/2)*.9;
    const peak=H*0.25 - i*12;
    const grad=g.createLinearGradient(0,peak,0,H-40);
    grad.addColorStop(0,cols[(d.modulation.state||'estimated').toLowerCase()]||'#22d3ee');
    grad.addColorStop(1,'rgba(34,211,238,0)');
    g.fillStyle=grad; g.beginPath(); g.moveTo(x0,H-40);
    g.quadraticCurveTo((x0+x1)/2,peak,x1,H-40); g.closePath(); g.fill();
    g.strokeStyle=cols[(d.modulation.state||'').toLowerCase()]||'#fff'; g.setLineDash(d.modulation.state==='inferred'?[6,4]:[]);
    g.strokeRect(x0,20,Math.max(4,x1-x0),H-60); g.setLineDash([]);
    g.fillStyle='#fff'; g.font='11px monospace';
    g.fillText((d.modulation.value||'?')+' '+(d.modulation.confidence!=null?(d.modulation.confidence*100).toFixed(0)+'%':''),x0+4,32+i*14);
  });
  g.fillStyle='rgba(148,163,184,.8)'; g.font='10px monospace'; g.fillText('−fs/2',4,H-8); g.fillText('+fs/2',W-40,H-8); g.fillText('0 Hz',W/2-14,H-8);
}
/* constellation sketch */
function drawConstellation(mod){
  const c=$('constCanvas'), g=c.getContext('2d'), W=c.width,H=c.height;
  g.fillStyle='#070c1b'; g.fillRect(0,0,W,H);
  g.strokeStyle='rgba(120,180,255,.25)'; g.beginPath(); g.moveTo(W/2,0); g.lineTo(W/2,H); g.moveTo(0,H/2); g.lineTo(W,H/2); g.stroke();
  const pts=[];
  const R=90, cx0=W/2, cy0=H/2;
  if(mod==='QAM') for(let i=-1.5;i<=1.5;i++) for(let q=-1.5;q<=1.5;q++) pts.push([cx0+i*R/1.5,cy0+q*R/1.5]);
  else if(mod==='PSK'||mod==='BPSK'||mod==='QPSK'){ const n=mod==='BPSK'?2:4; for(let i=0;i<n;i++){ const a=i/n*Math.PI*2+Math.PI/4; pts.push([cx0+Math.cos(a)*R, cy0+Math.sin(a)*R]); } }
  else if(mod==='FSK') for(let i=0;i<40;i++) pts.push([cx0+(Math.random()<.5?-R:R)+ (Math.random()-.5)*16, cy0+(Math.random()-.5)*160]);
  else if(mod==='OFDM') for(let i=0;i<160;i++) pts.push([cx0+(Math.random()-.5)*R*2, cy0+(Math.random()-.5)*R*2]);
  else if(mod==='CHIRP'){ for(let x=0;x<W;x+=4) pts.push([x, cy0+Math.sin(x*.05)*R*.9]); }
  else if(mod==='FM'||mod==='AM') for(let i=0;i<120;i++){ const a=i/120*Math.PI*2; pts.push([cx0+Math.cos(a)*R, cy0+Math.sin(a)*R*.9]); }
  else if(mod==='CW'||mod==='SSB') pts.push([cx0+R*.7,cy0]);
  else { g.fillStyle='#f472b6'; g.font='13px monospace'; g.fillText(mod==='UNKNOWN'?'refused: below SNR floor':'no constellation',20,H/2); return; }
  pts.forEach(p=>{ g.fillStyle='#22d3ee'; g.shadowColor='#22d3ee'; g.shadowBlur=6;
    g.beginPath(); g.arc(p[0]+(Math.random()-.5)*8,p[1]+(Math.random()-.5)*8,2.6,0,7); g.fill(); g.shadowBlur=0; });
  g.fillStyle='rgba(232,238,252,.75)'; g.font='11px monospace'; g.fillText(mod||'',10,18);
}

/* ---------- downloads ---------- */
function dl(name, text, type){
  const b=new Blob([text],{type}); const a=document.createElement('a');
  a.href=URL.createObjectURL(b); a.download=name; a.click(); setTimeout(()=>URL.revokeObjectURL(a.href),4000);
}
$('dlJson').onclick=()=>{ if(!current) return; dl(currentFile.replace(/\.\w+$/,'')+'_report.json', JSON.stringify(current,null,2), 'application/json'); };
$('dlSigmf').onclick=()=>{ if(!current) return;
  const s=currentSigmf||{global:{'core:sample_rate':current.sample_rate_hz},annotations:[]};
  dl(currentFile.replace(/\.\w+$/,'')+'.sigmf-meta', JSON.stringify(s,null,2), 'application/json'); };
$('dlPng').onclick=()=>{
  if(current&&current.id&&backendOK){ window.open(BASE+'/api/report/'+current.id+'/spectrogram','_blank'); return; }
  const a=document.createElement('a'); a.download=currentFile.replace(/\.\w+$/,'')+'_spectrum.png';
  a.href=$('specCanvas').toDataURL('image/png'); a.click();
};
$('copyJson').onclick=async()=>{ if(!current) return; await navigator.clipboard.writeText(JSON.stringify(current,null,2)); log('JSON copied to clipboard.'); };

/* ---------- dynamic glass nav ---------- */
const topnav=document.getElementById('topnav');
addEventListener('scroll',()=>{ if(topnav) topnav.classList.toggle('scrolled',scrollY>24); },{passive:true});

/* ---------- boot ---------- */
probe().then(loadDemos);
loadDemos();
waveFrame();
})();
