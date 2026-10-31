"""Interactive design-space explorer embedded in the report (no external resources).

Reads design_space from the report's embedded <script id="rcd-data"> blob. The client sets criterion weights,
a budget cap and whether a relief operator is acceptable; the shortlist re-ranks live, and any two designs can be
put side by side (A vs B) with per-criterion winners, P(A preferred) over random weightings and what each arm does.
"""

CSS = """
.hl-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px;margin:12px 0}
.hl{border:1px solid var(--line);border-radius:12px;padding:14px 16px}.hl.rec{border:2px solid var(--accent)}
.hl-k{font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--accent)}
.hl h4{margin:4px 0 4px;font-size:16px}.hl-who{font-size:12.5px;color:var(--muted);margin-bottom:6px}
.hl table{font-size:12.5px}.hl td{padding:3px 4px}
.ds-ctl{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px 18px;background:#f3f5f8;
border-radius:10px;padding:14px 16px;margin:10px 0}
.ds-ctl label{font-size:12.5px;color:var(--muted);display:block}.ds-ctl input[type=range]{width:100%}
.ds-ctl b{color:var(--ink)}
.ds-vs{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:10px 0}
.ds-card{border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.ds-card h4{margin:0 0 4px;font-size:15px}.ds-card .lab{font-size:12.5px;color:var(--muted);margin-bottom:8px}
.ds-card.win{border:2px solid var(--accent)}
.ds-row{display:grid;grid-template-columns:130px 1fr 92px;gap:8px;align-items:center;font-size:12.5px;margin:3px 0}
.ds-bar{height:9px;border-radius:5px;background:#e8ecf0;position:relative}
.ds-bar i{position:absolute;left:0;top:0;bottom:0;border-radius:5px;background:var(--human)}
.ds-bar i.w{background:var(--accent)}
.ds-pill{display:inline-block;font-size:11px;font-weight:600;border-radius:999px;padding:1px 8px;background:#e8ecf0;margin:1px 2px}
.ds-verdict{background:#f1f9f6;border-left:4px solid var(--accent);padding:10px 14px;border-radius:0 10px 10px 0;margin:8px 0}
@media(max-width:820px){.ds-vs{grid-template-columns:1fr}}
"""

HTML = """
<div id="ds-app">
 <div class="ds-ctl" id="ds-w"></div>
 <div class="ds-ctl">
  <div><label>Budget cap (installed cost) <b id="ds-bv"></b></label><input type="range" id="ds-budget" step="5000"></div>
  <div><label><input type="checkbox" id="ds-relief"> allow a relief operator</label>
       <label><input type="checkbox" id="ds-front" checked> final Pareto set only</label></div>
  <div><label>Reset to a stakeholder profile</label><select id="ds-persona"></select></div>
 </div>
 <div class="tw"><table id="ds-rank"></table></div>
 <h3>A vs B</h3>
 <div class="ds-ctl"><div><label>Design A</label><select id="ds-a"></select></div>
  <div><label>Design B</label><select id="ds-b"></select></div></div>
 <div id="ds-cmp"></div>
</div>
"""

JS = r"""
(function(){
const blob=JSON.parse(document.getElementById('rcd-data').textContent);const S=blob.design_space;if(!S)return;
const C=S.criteria, all=S.designs.filter(d=>d.simulated);
const fmt=(k,v)=>k.endsWith('usd')?(v<0?'-':'')+'$'+Math.round(Math.abs(v)/1000)+'k':k==='n_robots'?String(v):
  k==='takt_violation_pct'?v.toFixed(1)+'%':v.toFixed(1)+' s';
let W={};C.forEach(c=>W[c.key]=0.2);
const P=S.personas;const wbox=document.getElementById('ds-w');
C.forEach(c=>{const d=document.createElement('div');d.innerHTML=`<label>${c.label} (${c.sense==='max'?'higher':'lower'} is better) <b id="wv-${c.key}"></b></label><input type="range" min="0" max="100" value="20" id="w-${c.key}">`;wbox.appendChild(d);
  d.querySelector('input').oninput=e=>{W[c.key]=+e.target.value/100;draw();};});
const sel=document.getElementById('ds-persona');sel.innerHTML='<option value="">-</option>'+P.map((p,i)=>`<option value="${i}">${p.name}</option>`).join('');
sel.onchange=()=>{if(sel.value==='')return;const p=P[+sel.value];C.forEach(c=>{W[c.key]=p.weights[c.key];document.getElementById('w-'+c.key).value=Math.round(100*W[c.key]);});draw();};
const bud=document.getElementById('ds-budget');const mx=Math.max(...all.map(d=>d.metrics.capex_usd));
bud.min=0;bud.max=Math.ceil(mx/5000)*5000;bud.value=bud.max;bud.oninput=draw;
document.getElementById('ds-relief').onchange=draw;document.getElementById('ds-front').onchange=draw;
function pool(){const b=+bud.value,rl=document.getElementById('ds-relief').checked,fr=document.getElementById('ds-front').checked;
  return all.filter(d=>d.metrics.capex_usd<=b+1e-6&&(rl||!d.relief_operators_per_shift)&&(!fr||d.pareto_final));}
function vals(ds){const r={};C.forEach(c=>{const x=ds.map(d=>d.metrics[c.key]);r[c.key]=[Math.min(...x),Math.max(...x)];});
  return d=>{const v={};C.forEach(c=>{const [lo,hi]=r[c.key];v[c.key]=hi-lo<1e-12?1:(c.sense==='max'?(d.metrics[c.key]-lo)/(hi-lo):(hi-d.metrics[c.key])/(hi-lo));});return v;};}
function util(v,w){let s=0,t=0;C.forEach(c=>{s+=w[c.key]*v[c.key];t+=w[c.key];});return t?s/t:0;}
function arms(d){return S.stations.map(st=>`${st} ${d.arms[st]||'-'}`).join(' · ');}
function armsC(d){return S.stations.map(st=>`<span style="white-space:nowrap">${st} <b>${d.arms[st]||'-'}</b></span>`).join('<br>');}
function extras(d){const c={};d.layout_levers.filter(x=>!x.endsWith('@LINE')).forEach(x=>{const k=x.split('@')[0].replace(/_/g,' ');c[k]=(c[k]||0)+1;});
  const s=Object.entries(c).map(([k,v])=>`${k} x${v}`);if(d.relief_operators_per_shift)s.push(d.relief_operators_per_shift+' relief op');return s.join('<br>')||'-';}
function draw(){C.forEach(c=>document.getElementById('wv-'+c.key).textContent=Math.round(100*W[c.key])+'%');
  document.getElementById('ds-bv').textContent='$'+Math.round(+bud.value/1000)+'k';
  const ds=pool();const V=vals(ds);const rk=ds.map(d=>({d,u:util(V(d),W)})).sort((a,b)=>b.u-a.u);
  const T=document.getElementById('ds-rank');
  T.innerHTML='<thead><tr><th>#</th><th>design</th><th>arms</th><th>extras</th>'+C.map(c=>`<th>${c.label}</th>`).join('')+'<th>score</th><th>also known as</th></tr></thead><tbody>'+
   rk.slice(0,15).map((r,i)=>`<tr class="${r.d.id===S.recommended_design?'rec':''}"><td>${i+1}</td><td>${r.d.id}</td><td>${armsC(r.d)}</td><td style="white-space:nowrap">${extras(r.d)}</td>`+
   C.map(c=>`<td>${fmt(c.key,r.d.metrics[c.key])}</td>`).join('')+`<td>${(100*r.u).toFixed(0)}</td><td>${r.d.named_scenarios.join(', ')||''}</td></tr>`).join('')+
   (rk.length?'':'<tr><td colspan="9">No design fits these filters.</td></tr>')+'</tbody>';
  const A=document.getElementById('ds-a'),B=document.getElementById('ds-b');const keep=[A.value,B.value];
  const opts=all.map(d=>`<option value="${d.id}">${d.id} - ${arms(d)}${d.named_scenarios.length?' ('+d.named_scenarios.join(', ')+')':''}</option>`).join('');
    const defA=S.recommended_design||(rk[0]?rk[0].d.id:all[0].id);
  const defB=(rk.find(r=>r.d.id!==defA)||{d:all.find(d=>d.id!==defA)}).d.id;
  A.innerHTML=opts;B.innerHTML=opts;A.value=keep[0]||defA;B.value=keep[1]||defB;cmp();}
function cmp(){const a=all.find(d=>d.id===document.getElementById('ds-a').value),b=all.find(d=>d.id===document.getElementById('ds-b').value);if(!a||!b)return;
  const V=vals(all);const va=V(a),vb=V(b);let wa=0,n=20000,seed=7;const rnd=()=>{seed=(seed*16807)%2147483647;return seed/2147483647;};
  for(let i=0;i<n;i++){const w={};let s=0;C.forEach(c=>{w[c.key]=-Math.log(rnd());s+=w[c.key];});if(util(va,w)>util(vb,w))wa++;}
  const ua=util(va,W),ub=util(vb,W);const winA=ua>=ub;
  const row=(d,v,o,vo)=>C.map(c=>{const better=v[c.key]>vo[c.key]+1e-9;return `<div class="ds-row"><span>${c.label}</span><span class="ds-bar"><i class="${better?'w':''}" style="width:${Math.round(100*v[c.key])}%"></i></span><span>${fmt(c.key,d.metrics[c.key])}${better?' ✓':''}</span></div>`;}).join('');
  const robots=d=>d.robots.map(r=>`<div class="cite"><b>${r.model}</b> @ ${r.station} (${r.maker||''}): ${r.tasks.map(t=>`<span class="ds-pill">${t}</span>`).join('')} robot ${r.robot_s.toFixed(1)} s/door${r.tool_changes_per_cycle?', '+r.tool_changes_per_cycle+' tool changes':''}</div>`).join('')||'<div class="cite">no arm</div>';
  const card=(d,v,o,vo,w)=>`<div class="ds-card ${w?'win':''}"><h4>${d.id}${d.id===S.recommended_design?' (recommended)':''}</h4><div class="lab">${d.label}</div>${row(d,v,o,vo)}<div style="margin-top:8px">${robots(d)}</div><div class="cite">station cycle: ${S.stations.map(st=>st+' '+d.station_cycle_s[st].toFixed(0)+' s').join(', ')} (takt ${S.takt_s} s); DES ${d.des.throughput_per_hr.toFixed(2)} doors/h</div></div>`;
  const fc=S.featured_comparisons.find(f=>(f.a===a.id&&f.b===b.id)||(f.a===b.id&&f.b===a.id));
  document.getElementById('ds-cmp').innerHTML=`<div class="ds-vs">${card(a,va,b,vb,winA)}${card(b,vb,a,va,!winA)}</div>
   <div class="ds-verdict"><b>With your weights: ${winA?a.id:b.id} is better</b> (score ${(100*Math.max(ua,ub)).toFixed(0)} vs ${(100*Math.min(ua,ub)).toFixed(0)}).
   Over all possible weightings ${a.id} is preferred in ${(100*wa/n).toFixed(0)}% of cases.${fc&&fc.tradeoff?' '+fc.tradeoff+'.':''}
   ${fc?` Simulated throughput difference (paired, common random numbers): ${(fc.a===a.id?1:-1)*fc.throughput_diff_per_hr>=0?'+':''}${((fc.a===a.id?1:-1)*fc.throughput_diff_per_hr).toFixed(2)} ± ${fc.throughput_diff_ci95.toFixed(2)} doors/h.`:''}</div>`;}
document.getElementById('ds-a').onchange=cmp;document.getElementById('ds-b').onchange=cmp;draw();
})();
"""
