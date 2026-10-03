"""Generate an offline evaluation dashboard and publication-ready metric figure."""
import argparse
import html as html_module
import json
from collections import defaultdict
from pathlib import Path


def load_results(run_dir, summary_path):
    summary = json.loads(Path(summary_path).read_text())
    windows = {}
    with (run_dir / 'windows.jsonl').open() as f:
        for line in f:
            row = json.loads(line)
            key = tuple(row[name] for name in ('partition', 'task', 'episode', 'start', 'horizon', 'seed'))
            row.update(image_trace=[], tokenizer_trace=[], latent_trace=[])
            windows[key] = row
    with (run_dir / 'steps.jsonl').open() as f:
        for line in f:
            row = json.loads(line)
            key = tuple(row[name] for name in ('partition', 'task', 'episode', 'start', 'horizon', 'seed'))
            window = windows[key]
            window['image_trace'].append(row['image_rms'])
            window['tokenizer_trace'].append(row['tokenizer_image_rms'])
            window['latent_trace'].append(row['latent_rms'])
    for window in windows.values():
        if len(window['image_trace']) != window['horizon']:
            raise ValueError(f'incomplete window: {window}')
        window['image_rms'] = window['image_trace'][-1]
        window['tokenizer_rms'] = window['tokenizer_trace'][-1]
        window['latent_rms'] = window['latent_trace'][-1]
        window['visual'] = (f"visuals/{window['partition']}/{window['task']}/"
                            f"ep{window['episode']}_s{window['start']}_"
                            f"h{window['horizon']}_seed{window['seed']}.png")
    return summary, list(windows.values())


def write_png(summary, windows, path, manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    tasks = sorted({row['task'] for row in summary})
    horizons = sorted({row['horizon'] for row in summary})
    colors = plt.get_cmap('tab10')
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), constrained_layout=True,
                             gridspec_kw={'width_ratios': [1.1, 1.1, 1]})
    ax = axes[0]
    for index, task in enumerate(tasks):
        rows = {row['horizon']: row for row in summary if row['task'] == task}
        ax.plot(horizons, [rows[h]['terminal_image_rms'] for h in horizons],
                marker='o', linewidth=2, color=colors(index), label=task)
    ax.set(title='Prediction error by rollout horizon', xlabel='Predicted transitions',
           ylabel='Endpoint image RMS (pixels scaled to 0–1)', xticks=horizons)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, ncol=2, loc='upper left')

    ax = axes[1]
    for index, task in enumerate(tasks):
        rows = {row['horizon']: row for row in summary if row['task'] == task}
        ax.plot(horizons, [rows[h]['terminal_latent_rms'] for h in horizons],
                marker='o', linewidth=2, color=colors(index), label=task)
    ax.set(title='Latent prediction error by rollout horizon', xlabel='Predicted transitions',
           ylabel='Endpoint latent RMS', xticks=horizons)
    ax.grid(alpha=.2)

    ax = axes[2]
    final_horizon = max(horizons)
    rows = {row['task']: row for row in summary if row['horizon'] == final_horizon}
    ordered = sorted(tasks, key=lambda task: rows[task]['terminal_image_rms'])
    y = np.arange(len(ordered))
    ax.barh(y + .18, [rows[t]['terminal_image_rms'] for t in ordered], height=.34,
            color='#2563eb', label='World-model prediction')
    ax.barh(y - .18, [rows[t]['terminal_tokenizer_image_rms'] for t in ordered], height=.34,
            color='#94a3b8', label='Tokenizer-only reconstruction')
    ax.set(yticks=y, yticklabels=ordered, xlabel='Endpoint image RMS',
           title=f'Prediction versus reconstruction at {final_horizon} transitions')
    ax.invert_yaxis()
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=.2)
    partition = ', '.join(sorted({row['partition'] for row in summary}))
    start_label = ('matched starts' if manifest['config']['sampling'].get('paired_horizons', False)
                   else 'separate starts by horizon')
    fig.suptitle(f'Robotics world model · {partition} recorded-action evaluation · {start_label}', fontsize=16,
                 fontweight='bold')
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def write_dashboard(summary, windows, path, manifest):
    tasks = sorted({row['task'] for row in summary})
    horizons = sorted({row['horizon'] for row in summary})
    dataset = {'summary': summary, 'windows': windows, 'tasks': tasks, 'horizons': horizons,
               'manifest': manifest}
    payload = json.dumps(dataset, separators=(',', ':')).replace('<', '\\u003c')
    partition = html_module.escape(', '.join(sorted({row['partition'] for row in summary})))
    start_note = ('The same start frame is used across horizons within each episode.'
                  if manifest['config']['sampling'].get('paired_horizons', False)
                  else 'Horizons use separately sampled start frames, so their curves mix horizon and episode phase.')
    html = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Robotics evaluation · __PARTITION__</title>
<style>
:root{color-scheme:light;--ink:#142033;--muted:#617089;--line:#dce4ed;--blue:#2563eb;--base:#94a3b8;--surface:#fff;--bg:#f4f7fb}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
header{background:#111e35;color:white;padding:30px max(24px,calc((100vw - 1280px)/2))}h1{margin:4px 0;font-size:29px;letter-spacing:-.04em}header p{margin:5px 0;color:#c7d3e8}.tag{display:inline-block;padding:3px 10px;border:1px solid #6d83aa;border-radius:99px;font-size:12px;color:#d7e3f8}
main{max-width:1280px;margin:0 auto;padding:22px 24px 50px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px}.card,.panel{background:white;border:1px solid var(--line);border-radius:14px;box-shadow:0 3px 14px #20385a09}.card{padding:15px}.card strong{display:block;font-size:26px;line-height:1.2}.card span{font-size:12px;color:var(--muted)}
.controls{display:flex;gap:12px;align-items:center;margin:12px 0 18px;flex-wrap:wrap}.controls label{font-size:13px;font-weight:600}.controls select{margin-left:8px;padding:8px 12px;border:1px solid var(--line);border-radius:8px;background:white;color:var(--ink)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.panel{padding:18px;min-width:0}.panel h2{margin:0 0 3px;font-size:18px;letter-spacing:-.02em}.sub{color:var(--muted);font-size:12px;margin:0 0 15px}.chart{width:100%;height:280px}.chart text{fill:#65748b;font-size:11px}.chart .gridline{stroke:#e5eaf1}.chart .axis{stroke:#8a98aa}
.wide{grid-column:1/-1}.heatmap{display:grid;grid-template-columns:82px repeat(2,minmax(65px,1fr));gap:4px;max-width:450px;max-height:440px;overflow:auto}.heatmap .cell{border:0;border-radius:5px;cursor:pointer;height:23px;font-size:11px;text-align:center}.heatmap .cell:hover,.heatmap .cell.active{outline:2px solid #0f172a;outline-offset:1px}.heatmap .rowlabel{font-size:11px;color:var(--muted);align-self:center}.heatmap .head{font-size:11px;text-align:center;color:var(--muted)}
.pair{display:grid;grid-template-columns:minmax(320px,450px) 1fr;gap:20px}.list{max-height:580px;overflow:auto;border:1px solid var(--line);border-radius:8px}.list button{display:flex;width:100%;justify-content:space-between;text-align:left;background:#fff;border:0;border-bottom:1px solid var(--line);padding:8px 10px;cursor:pointer;color:var(--ink);font:inherit}.list button:hover,.list button.active{background:#eff6ff}.list small{color:var(--muted)}.selected-metrics{display:flex;gap:16px;flex-wrap:wrap;margin:8px 0 12px;color:var(--muted);font-size:13px}.selected-metrics b{color:var(--ink)}
.image-wrap{overflow:auto;background:#f0f3f8;border:1px solid var(--line);border-radius:8px;max-height:740px}.image-wrap img{display:block;max-width:none}.note{font-size:12px;color:var(--muted);margin:10px 0 0}details{margin-top:12px;color:var(--muted);font-size:12px}pre{overflow:auto;background:#f6f8fb;border:1px solid var(--line);padding:10px;border-radius:8px;color:var(--ink)}
@media(max-width:850px){.cards{grid-template-columns:repeat(2,1fr)}.grid,.pair{grid-template-columns:1fr}.wide{grid-column:auto}}
</style></head><body>
<header><span class="tag">Recorded-action evaluation · __PARTITION__</span><h1>Where the model predicts well—and where it drifts</h1><p>Compare task, horizon, episode, and stochastic seed. Scores are pixel/latent errors, not hallucination labels. __START_NOTE__</p></header>
<main><div id="cards" class="cards"></div><div class="controls"><label>Task<select id="task"></select></label><label>Horizon<select id="horizon"></select></label></div>
<div class="grid"><section class="panel"><h2>How image error changes with horizon</h2><p class="sub">Endpoint image RMS for the selected task; lower is better.</p><svg id="trend" class="chart" viewBox="0 0 560 280" role="img" aria-label="Image error by horizon"></svg></section>
<section class="panel"><h2>Tasks at the selected horizon</h2><p class="sub">World-model prediction compared with tokenizer-only reconstruction.</p><svg id="bars" class="chart" viewBox="0 0 560 280" role="img" aria-label="Task error comparison"></svg></section>
<section class="panel"><h2>How latent error changes with horizon</h2><p class="sub">Endpoint RMS between predicted and encoded real latents; lower is better.</p><svg id="latentTrend" class="chart" viewBox="0 0 560 280" role="img" aria-label="Latent error by horizon"></svg></section>
<section class="panel"><h2>Latent error across tasks</h2><p class="sub">Endpoint latent RMS at the selected horizon.</p><svg id="latentBars" class="chart" viewBox="0 0 560 280" role="img" aria-label="Task latent error comparison"></svg></section>
<section class="panel wide"><h2>Inspect the recorded rollouts</h2><p class="sub">The heatmap shows terminal image RMS for each episode and seed. Click a cell or row to inspect its real, predicted, and tokenizer frames.</p>
<div class="pair"><div><div id="heatmap" class="heatmap"></div><p class="note">Darker blue means larger image error. It does not indicate a physical hallucination.</p><h3>Windows, largest error first</h3><div id="windowList" class="list"></div></div>
<div><div id="selectedTitle"></div><div id="selectedMetrics" class="selected-metrics"></div><div class="image-wrap"><img id="contactSheet" alt="Real, predicted, and tokenizer-only reconstruction strips"></div><p class="note">Top: recorded real frames. Middle: predicted frames. Bottom: tokenizer-only reconstruction. The first predicted frame aligns one column after the real start.</p><details><summary>Recorded actions for this window</summary><pre id="actions"></pre></details></div></div></section></div></main>
<script id="results" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('results').textContent),$=id=>document.getElementById(id);
let selectedTask=data.tasks[0],selectedHorizon=data.horizons.at(-1),selectedWindow=null;
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const f=x=>Number(x).toFixed(4), rows=(task,h)=>data.summary.filter(r=>r.task===task&&(h==null||r.horizon===h));
const windows=()=>data.windows.filter(w=>w.task===selectedTask&&w.horizon===selectedHorizon);
const svgText=(x,y,value,anchor='middle')=>`<text x="${x}" y="${y}" text-anchor="${anchor}">${esc(value)}</text>`;
function setup(){let episodes=new Set(data.windows.map(w=>`${w.task}/${w.episode}`)).size;
 $('cards').innerHTML=[[data.tasks.length,'tasks'],[episodes,'task–episodes'],[data.windows.length.toLocaleString(),'rollout windows'],[data.windows.reduce((a,w)=>a+w.horizon,0).toLocaleString(),'predicted transitions']].map(([n,label])=>`<div class="card"><strong>${n}</strong><span>${label}</span></div>`).join('');
 $('task').innerHTML=data.tasks.map(t=>`<option value="${esc(t)}">${esc(t)}</option>`).join('');
 $('horizon').innerHTML=data.horizons.map(h=>`<option value="${h}">${h} transitions</option>`).join('');
 $('horizon').value=selectedHorizon;$('task').onchange=e=>{selectedTask=e.target.value;selectedWindow=null;render()};$('horizon').onchange=e=>{selectedHorizon=+e.target.value;selectedWindow=null;render()};render()}
function renderTrend(){let r=rows(selectedTask),max=Math.max(...r.flatMap(v=>[v.terminal_image_rms,v.terminal_tokenizer_image_rms]))*1.14;
 let left=52,right=530,top=20,bottom=227,y=v=>bottom-v/max*(bottom-top),lo=Math.log2(data.horizons[0]),span=Math.log2(data.horizons.at(-1))-lo||1,x=h=>left+(Math.log2(h)-lo)/span*(right-left),parts=[];
 for(let i=0;i<=4;i++){let v=max*i/4,Y=y(v);parts.push(`<line class="gridline" x1="${left}" y1="${Y}" x2="${right}" y2="${Y}"/>${svgText(left-8,Y+4,v.toFixed(3),'end')}`)}
 for(let h of data.horizons)parts.push(svgText(x(h),bottom+18,h));
 for(let [name,key,color] of [['prediction','terminal_image_rms','#2563eb'],['tokenizer','terminal_tokenizer_image_rms','#94a3b8']]){let pts=r.map(v=>`${x(v.horizon)},${y(v[key])}`).join(' ');parts.push(`<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="3"/>`);for(let v of r)parts.push(`<circle cx="${x(v.horizon)}" cy="${y(v[key])}" r="4" fill="${color}"><title>${esc(name)} · ${v.horizon} steps: ${f(v[key])}</title></circle>`)}
 parts.push(`<circle cx="165" cy="267" r="5" fill="#2563eb"/>${svgText(178,271,'Prediction','start')}<circle cx="315" cy="267" r="5" fill="#94a3b8"/>${svgText(328,271,'Tokenizer only','start')}`);$('trend').innerHTML=parts.join('')}
function renderBars(){let r=data.summary.filter(v=>v.horizon===selectedHorizon).sort((a,b)=>b.terminal_image_rms-a.terminal_image_rms),max=Math.max(...r.map(v=>v.terminal_image_rms))*1.12,left=140,right=530,parts=[];
 $('bars').setAttribute('viewBox',`0 0 560 ${Math.max(280,30+r.length*29)}`);$('bars').style.height=`${Math.max(280,30+r.length*29)}px`;
 r.forEach((v,i)=>{let y=21+i*29;parts.push(svgText(left-9,y+10,v.task,'end'));parts.push(`<rect x="${left}" y="${y}" width="${v.terminal_image_rms/max*(right-left)}" height="10" rx="2" fill="#2563eb"><title>${esc(v.task)} prediction ${f(v.terminal_image_rms)}</title></rect>`);parts.push(`<rect x="${left}" y="${y+12}" width="${v.terminal_tokenizer_image_rms/max*(right-left)}" height="8" rx="2" fill="#94a3b8"><title>${esc(v.task)} tokenizer ${f(v.terminal_tokenizer_image_rms)}</title></rect>`)});
 $('bars').innerHTML=parts.join('')}
function renderLatentTrend(){let r=rows(selectedTask),max=Math.max(...r.map(v=>v.terminal_latent_rms))*1.14;
 let left=52,right=530,top=20,bottom=227,y=v=>bottom-v/max*(bottom-top),lo=Math.log2(data.horizons[0]),span=Math.log2(data.horizons.at(-1))-lo||1,x=h=>left+(Math.log2(h)-lo)/span*(right-left),parts=[];
 for(let i=0;i<=4;i++){let v=max*i/4,Y=y(v);parts.push(`<line class="gridline" x1="${left}" y1="${Y}" x2="${right}" y2="${Y}"/>${svgText(left-8,Y+4,v.toFixed(3),'end')}`)}
 for(let h of data.horizons)parts.push(svgText(x(h),bottom+18,h));
 parts.push(`<polyline points="${r.map(v=>`${x(v.horizon)},${y(v.terminal_latent_rms)}`).join(' ')}" fill="none" stroke="#7c3aed" stroke-width="3"/>`);
 for(let v of r)parts.push(`<circle cx="${x(v.horizon)}" cy="${y(v.terminal_latent_rms)}" r="4" fill="#7c3aed"><title>${v.horizon} steps: ${f(v.terminal_latent_rms)}</title></circle>`);
 $('latentTrend').innerHTML=parts.join('')}
function renderLatentBars(){let r=data.summary.filter(v=>v.horizon===selectedHorizon).sort((a,b)=>b.terminal_latent_rms-a.terminal_latent_rms),max=Math.max(...r.map(v=>v.terminal_latent_rms))*1.12,left=140,right=530,parts=[];
 $('latentBars').setAttribute('viewBox',`0 0 560 ${Math.max(280,30+r.length*29)}`);$('latentBars').style.height=`${Math.max(280,30+r.length*29)}px`;
 r.forEach((v,i)=>{let y=25+i*29;parts.push(svgText(left-9,y+9,v.task,'end'));parts.push(`<rect x="${left}" y="${y}" width="${v.terminal_latent_rms/max*(right-left)}" height="16" rx="3" fill="#7c3aed"><title>${esc(v.task)} latent ${f(v.terminal_latent_rms)}</title></rect>`)});
 $('latentBars').innerHTML=parts.join('')}
function renderInspector(){let list=windows().sort((a,b)=>b.image_rms-a.image_rms);if(!selectedWindow||!list.includes(selectedWindow))selectedWindow=list[0];let min=Math.min(...list.map(w=>w.image_rms)),max=Math.max(...list.map(w=>w.image_rms));
 let byRow=new Map();for(let w of list)byRow.set(`${w.episode}/${w.start}/${w.seed}`,w);let episodeStarts=[...new Set(list.map(w=>`${w.episode}/${w.start}`))].sort((a,b)=>{let [ae,as]=a.split('/').map(Number),[be,bs]=b.split('/').map(Number);return ae-be||as-bs}),seeds=[...new Set(list.map(w=>w.seed))].sort((a,b)=>a-b);
 $('heatmap').style.gridTemplateColumns=`82px repeat(${seeds.length},minmax(65px,1fr))`;
 let html='<div></div>'+seeds.map(s=>`<div class="head">seed ${s}</div>`).join('');for(let key of episodeStarts){let [ep,start]=key.split('/').map(Number);html+=`<div class="rowlabel">ep ${ep} · s${start}</div>`;for(let seed of seeds){let w=byRow.get(`${ep}/${start}/${seed}`),v=w.image_rms,t=(v-min)/(max-min||1),light=94-t*46;html+=`<button class="cell ${w===selectedWindow?'active':''}" data-ep="${ep}" data-start="${start}" data-seed="${seed}" style="background:hsl(213 88% ${light}%);color:${t>.55?'white':'#0f294a'}" title="episode ${ep}, start ${start}, seed ${seed}: ${f(v)}">${f(v)}</button>`}}$('heatmap').innerHTML=html;
 $('heatmap').querySelectorAll('button').forEach(b=>b.onclick=()=>{selectedWindow=byRow.get(`${b.dataset.ep}/${b.dataset.start}/${b.dataset.seed}`);renderInspector()});
 $('windowList').innerHTML=list.map(w=>`<button class="${w===selectedWindow?'active':''}" data-ep="${w.episode}" data-start="${w.start}" data-seed="${w.seed}"><span>Episode ${w.episode} · start ${w.start} · seed ${w.seed}</span><small>${f(w.image_rms)}</small></button>`).join('');$('windowList').querySelectorAll('button').forEach(b=>b.onclick=()=>{selectedWindow=byRow.get(`${b.dataset.ep}/${b.dataset.start}/${b.dataset.seed}`);renderInspector()});
 let w=selectedWindow;$('selectedTitle').innerHTML=`<h3>Episode ${w.episode} · frame ${w.start} → ${w.start+w.horizon} · seed ${w.seed}</h3>`;
 $('selectedMetrics').innerHTML=`<span>Prediction image RMS <b>${f(w.image_rms)}</b></span><span>Tokenizer baseline <b>${f(w.tokenizer_rms)}</b></span><span>Latent RMS <b>${f(w.latent_rms)}</b></span>`;
 $('contactSheet').src=w.visual;$('actions').textContent=JSON.stringify(w.actions,null,2)}
function render(){renderTrend();renderBars();renderLatentTrend();renderLatentBars();renderInspector()}setup();
</script></body></html>'''
    Path(path).write_text(html.replace('__DATA__', payload).replace('__PARTITION__', partition)
                          .replace('__START_NOTE__', html_module.escape(start_note)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    summary, windows = load_results(args.run, args.summary)
    manifest = json.loads((args.run / 'manifest.json').read_text())
    write_dashboard(summary, windows, args.run / 'dashboard.html', manifest)
    write_png(summary, windows, args.run / 'summary.png', manifest)
    missing = sum(not (args.run / window['visual']).exists() for window in windows)
    if missing:
        raise SystemExit(f'{missing} review images are missing')
    print(f"{args.run / 'dashboard.html'}\n{args.run / 'summary.png'}\n{len(windows)} windows visualized")


if __name__ == '__main__':
    main()
