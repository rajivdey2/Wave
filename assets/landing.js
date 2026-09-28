/* Landing: Three.js 3D wave surface + 2D HUD animations + reveals */
(function(){
  "use strict";
  /* ---------- scroll reveal ---------- */
  const io = new IntersectionObserver(es=>es.forEach(e=>{ if(e.isIntersecting){ e.target.classList.add('on'); io.unobserve(e.target);} }),{threshold:.12});
  document.querySelectorAll('.rv').forEach(el=>io.observe(el));

  /* ---------- animated counters ---------- */
  const cio = new IntersectionObserver(es=>es.forEach(e=>{
    if(!e.isIntersecting) return; cio.unobserve(e.target);
    const el=e.target, target=+el.dataset.count||0, suf=el.textContent.includes('%')?'%':'';
    const t0=performance.now();
    (function tick(t){ const p=Math.min(1,(t-t0)/1400), v=Math.round(target*(1-Math.pow(1-p,3)));
      el.textContent=v+suf; if(p<1) requestAnimationFrame(tick); })(t0);
  }),{threshold:.5});
  document.querySelectorAll('[data-count]').forEach(el=>cio.observe(el));

  /* ---------- demo preview cards ---------- */
  const demos=[
    ['BPSK','Digital · 10 dB','PSK @ 10 kHz, 2 kbaud, 9.9 dB — textbook detection.',95],
    ['MULTI ×3','Wideband mix','BPSK + CW + FM in ONE capture, 3 boxes.',100],
    ['OFDM','Wideband · 10 dB','64-subcarrier, 50 active — flat-top spectrum.',88],
    ['CHIRP','Radar-like sweep','20 kHz linear sweep, diagonal waterfall.',92],
    ['FM','Analog · 10 dB','Wideband FM, 6.2 kHz deviation signature.',97],
    ['CW','Morse-style keying','5 Hz on/off keying, razor-thin line.',90],
    ['NOISE','Refusal case','Pure noise → zero confident labels. Honest.',100],
    ['−5 dB','SNR floor','Below floor → explicit UNKNOWN, never a bluff.',100],
  ];
  const g=document.getElementById('demoGrid');
  if(g) g.innerHTML=demos.map(d=>`<div class="dcard rv on" onclick="location.href='app.html'"><div class="m">${d[0]}</div><h4>${d[1]}</h4><p>${d[2]}</p><div class="bar"><i style="width:${d[3]}%"></i></div></div>`).join('');

  /* ---------- HUD 2D wave ---------- */
  const hud=document.getElementById('wave2d');
  if(hud){
    const c=hud.getContext('2d'); let t=0;
    (function draw(){
      t+=.05; const W=hud.width,H=hud.height;
      const gr=c.createLinearGradient(0,0,0,H); gr.addColorStop(0,'#0b1330'); gr.addColorStop(1,'#070c1b');
      c.fillStyle=gr; c.fillRect(0,0,W,H);
      c.strokeStyle='rgba(120,180,255,.12)'; c.lineWidth=1;
      for(let y=0;y<H;y+=18){ c.beginPath(); c.moveTo(0,y); c.lineTo(W,y); c.stroke(); }
      const waves=[{f:3,a:34,col:'#22d3ee'},{f:5,a:18,col:'#8b5cf6'},{f:9,a:9,col:'#f472b6'}];
      waves.forEach((w,k)=>{
        c.beginPath();
        for(let x=0;x<=W;x+=2){
          const y=H/2 + w.a*Math.sin(x/W*Math.PI*2*w.f + t*(1+k*.4)) * Math.sin(t*.3+k);
          x?c.lineTo(x,y):c.moveTo(x,y);
        }
        c.strokeStyle=w.col; c.lineWidth=2; c.shadowColor=w.col; c.shadowBlur=8; c.stroke(); c.shadowBlur=0;
      });
      requestAnimationFrame(draw);
    })();
  }

  /* ---------- 3D hero: animated RF wave surface ---------- */
  function fallback2D(){
    const cv=document.getElementById('hero2d'); if(!cv) return;
    cv.style.display='block'; cv.width=innerWidth; cv.height=innerHeight;
    const c=cv.getContext('2d'); let t=0;
    (function draw(){
      t+=.02; const W=cv.width,H=cv.height;
      c.fillStyle='#05070f'; c.fillRect(0,0,W,H);
      for(let r=0;r<4;r++){
        c.beginPath();
        for(let x=0;x<=W;x+=4){
          const y=H*.55 + Math.sin(x*.008+t*(1+r*.3)+r*2)*(40+r*22)*Math.sin(x*.002-t*.5);
          x?c.lineTo(x,y):c.moveTo(x,y);
        }
        c.strokeStyle=['#22d3ee','#8b5cf6','#f472b6','#a3e635'][r]+'55'; c.lineWidth=2; c.stroke();
      }
      requestAnimationFrame(draw);
    })();
  }
  /* ---------- dynamic glass nav ---------- */
  const topnav=document.getElementById('topnav');
  addEventListener('scroll',()=>{ if(topnav) topnav.classList.toggle('scrolled',scrollY>24); },{passive:true});
  try{
    if(typeof THREE==='undefined') throw 0;
    const canvas=document.getElementById('bg3d');
    const renderer=new THREE.WebGLRenderer({canvas,alpha:true,antialias:true});
    renderer.setPixelRatio(Math.min(devicePixelRatio,2));
    const scene=new THREE.Scene();
    scene.fog=new THREE.FogExp2(0x05070f,.028);
    const cam=new THREE.PerspectiveCamera(55,1,.1,200);
    cam.position.set(0,14,34); cam.lookAt(0,0,0);
    // wave surface
    const SEG=90, SIZE=70;
    const geo=new THREE.PlaneGeometry(SIZE,SIZE,SEG,SEG);
    geo.rotateX(-Math.PI/2.3);
    const mat=new THREE.MeshBasicMaterial({color:0x22d3ee,wireframe:true,transparent:true,opacity:.32});
    const mesh=new THREE.Mesh(geo,mat); mesh.position.y=-4; scene.add(mesh);
    const mat2=new THREE.MeshBasicMaterial({color:0x8b5cf6,wireframe:true,transparent:true,opacity:.14});
    const mesh2=new THREE.Mesh(geo.clone(),mat2); mesh2.position.y=-4.4; mesh2.scale.set(1.15,1,1.15); scene.add(mesh2);
    // particle field (samples)
    const N=700, pos=new Float32Array(N*3);
    for(let i=0;i<N;i++){ pos[i*3]=(Math.random()-.5)*90; pos[i*3+1]=Math.random()*26-6; pos[i*3+2]=(Math.random()-.5)*90; }
    const pg=new THREE.BufferGeometry(); pg.setAttribute('position',new THREE.BufferAttribute(pos,3));
    const pts=new THREE.Points(pg,new THREE.PointsMaterial({color:0x8b5cf6,size:.35,transparent:true,opacity:.8}));
    scene.add(pts);
    // orbiting "satellite" markers = detections
    const sats=[];
    [[-14,4,-8,0xa3e635],[10,6,-12,0xfbbf24],[2,3,10,0xf472b6]].forEach(([x,y,z,col])=>{
      const s=new THREE.Mesh(new THREE.SphereGeometry(.7,16,16),new THREE.MeshBasicMaterial({color:col}));
      s.position.set(x,y,z); scene.add(s);
      const ring=new THREE.Mesh(new THREE.RingGeometry(1.1,1.35,40),new THREE.MeshBasicMaterial({color:col,transparent:true,opacity:.6,side:THREE.DoubleSide}));
      ring.position.copy(s.position); ring.rotation.x=-Math.PI/2.4; scene.add(ring);
      sats.push({s,ring,ph:Math.random()*9});
    });
    const base=geo.attributes.position.array.slice();
    const base2=mesh2.geometry.attributes.position.array.slice();
    let t=0, mx=0;
    addEventListener('pointermove',e=>{ mx=(e.clientX/innerWidth-.5)*2; });
    function size(){ const w=innerWidth,h=Math.max(innerHeight,document.querySelector('.hero').offsetHeight);
      renderer.setSize(w,h,false); cam.aspect=w/h; cam.updateProjectionMatrix(); }
    size(); addEventListener('resize',size);
    (function anim(){
      t+=.016;
      const p=geo.attributes.position;
      for(let i=0;i<p.count;i++){
        const x=base[i*3], z=base[i*3+2];
        p.array[i*3+1]=base[i*3+1]+Math.sin(x*.35+t*1.6)*1.5+Math.cos(z*.3+t*1.1)*1.5+Math.sin((x+z)*.15+t*.6)*1.1;
      }
      p.needsUpdate=true;
      const p2=mesh2.geometry.attributes.position;
      for(let i=0;i<p2.count;i++){
        const x=base2[i*3], z=base2[i*3+2];
        p2.array[i*3+1]=base2[i*3+1]+Math.sin(x*.3-t*1.2)*1.7+Math.cos(z*.35+t*.9)*1.4;
      }
      p2.needsUpdate=true;
      pts.rotation.y=t*.03;
      sats.forEach((o,i)=>{ o.s.position.y+=Math.sin(t*2+o.ph)*.008; o.ring.rotation.z=t*(.4+i*.2); o.ring.scale.setScalar(1+Math.sin(t*2+o.ph)*.08); });
      cam.position.x+=((mx*6)-cam.position.x)*.03;
      cam.lookAt(0,0,0);
      renderer.render(scene,cam);
      requestAnimationFrame(anim);
    })();
  }catch(e){ fallback2D(); }
})();
