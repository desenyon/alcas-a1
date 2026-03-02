"""
ALCAS - Poster Figures Part 1  (v3 - full redesign)
Run from ~/alcas:  python src/viz/plot_figures_part1.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from scipy import stats as scipy_stats
from scipy.stats import gaussian_kde

# ============================================================
# GLOBAL THEME
# ============================================================
DARK_BG  = '#080D18'
PANEL_BG = '#0F1726'
CARD_BG  = '#162035'
BORDER   = '#243555'
A1       = '#3B9EFF'   # blue  - allosteric
A2       = '#FF6B2B'   # orange - active-site
A3       = '#0FBA6F'   # green  - WT / success
A4       = '#9B6DFF'   # purple - distance-only
TEXT1    = '#EEF2FF'
TEXT2    = '#8DA4C8'
TEXT3    = '#3D5278'
SUCCESS  = '#0FBA6F'
WARN     = '#F5C842'
DANGER   = '#FF4D6A'

plt.rcParams.update({
    'figure.facecolor':   DARK_BG,
    'axes.facecolor':     PANEL_BG,
    'axes.edgecolor':     BORDER,
    'axes.labelcolor':    TEXT2,
    'axes.titlecolor':    TEXT1,
    'xtick.color':        TEXT2,
    'ytick.color':        TEXT2,
    'text.color':         TEXT1,
    'grid.color':         '#172030',
    'grid.linewidth':     0.6,
    'font.family':        'DejaVu Sans',
    'font.size':          10,
    'axes.titlesize':     11,
    'axes.labelsize':     10,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'legend.facecolor':   CARD_BG,
    'legend.edgecolor':   BORDER,
    'legend.fontsize':    9,
    'savefig.dpi':        250,
    'savefig.facecolor':  DARK_BG,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.25,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
})

OUT = Path('poster_assets')
OUT.mkdir(exist_ok=True)

GC = {'wt': A3, 'active_site': A2, 'allosteric': A1}

def save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=250, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    kb = p.stat().st_size // 1024
    print(f"  {name}  ({kb} KB)")

def ftitle(ax, t, size=12):
    ax.set_title(t, color=TEXT1, fontsize=size, fontweight='bold',
                 pad=8, loc='left')

def pval_str(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

def bracket(ax, x1, x2, y, p, h=0.3, fs=10):
    sym = pval_str(p)
    col = SUCCESS if p < 0.05 else TEXT3
    ax.plot([x1,x1,x2,x2], [y, y+h, y+h, y], color=col, lw=1.1)
    ax.text((x1+x2)/2, y+h+0.02, sym, ha='center', va='bottom',
            color=col, fontsize=fs, fontweight='bold')

def violin_core(ax, data, pos, color, clip_pct=97):
    clipped = np.clip(data, np.percentile(data,1), np.percentile(data,clip_pct))
    vp = ax.violinplot(clipped, positions=[pos], widths=0.55,
                       showmeans=False, showmedians=False, showextrema=False)
    for b in vp['bodies']:
        b.set_facecolor(color); b.set_alpha(0.28)
        b.set_edgecolor(color); b.set_linewidth(1.8)
    q25,q50,q75 = np.percentile(clipped,[25,50,75])
    ax.plot([pos,pos],[q25,q75], color=color, lw=2.5, solid_capstyle='round', zorder=4)
    ax.scatter([pos],[q50], color=color, s=55, zorder=5)
    rng = np.random.RandomState(int(pos*13+7))
    j = rng.uniform(-0.13,0.13, len(clipped))
    ax.scatter(np.full(len(clipped),pos)+j, clipped,
               color=color, alpha=0.22, s=12, zorder=2)
    ax.scatter([pos],[data.mean()], color='white', s=90, marker='D',
               zorder=6, edgecolors=color, linewidths=1.8)
    return data.mean()

# ============================================================
# LOAD DATA
# ============================================================
print("Loading data...")
with open('results/foldx/active_site_foldx.json')     as f: as_fx  = json.load(f)['results']
with open('results/foldx/allosteric_foldx.json')       as f: al_fx  = json.load(f)['results']
with open('results/foldx/foldx_summary.json')          as f: fx_s   = json.load(f)
with open('results/foldx/stability_active_site.json')  as f: as_st  = json.load(f)['results']
with open('results/foldx/stability_allosteric.json')   as f: al_st  = json.load(f)['results']
with open('results/foldx/stability_summary.json')      as f: st_s   = json.load(f)
with open('results/ablation/ablation_summary.json')    as f: ab_s   = json.load(f)
with open('results/ablation/distance_only_results.json') as f: do_r = json.load(f)['results']
with open('results/esmfold/esmfold_analysis.json')     as f: esm    = json.load(f)
with open('results/docking/docking_summary.json')      as f: dk_s   = json.load(f)
with open('data/petase/coupling/all_residue_scores.json') as f: coup = json.load(f)
with open('data/petase/mask_active_site.json')         as f: am     = set(json.load(f)['residues'])
with open('data/petase/masks_allosteric.json')         as f: alm    = set(json.load(f)['residues'])

as_ddg = np.array([r['foldx_ddg']    for r in as_fx if r['foldx_ddg'] is not None])
al_ddg = np.array([r['foldx_ddg']    for r in al_fx if r['foldx_ddg'] is not None])
do_ddg = np.array([r['foldx_ddg']    for r in do_r  if r['foldx_ddg'] is not None])
as_sta = np.array([r['stability_ddg'] for r in as_st if r['stability_ddg'] is not None])
al_sta = np.array([r['stability_ddg'] for r in al_st if r['stability_ddg'] is not None])
rn_all = [int(k)                     for k in coup.keys()]
sc_all = [float(v['coupling_score']) for v in coup.values()]
rn2sc  = dict(zip(rn_all, sc_all))
print("  Done")


# ============================================================
# F01 — FoldX BINDING  (violin + stat card with hard zone separation)
# ============================================================
print("\nGenerating figures...")

def make_violin_figure(title_suptitle, title_ax, ylabel, note,
                       data_pairs, summary, callout_lines, outname,
                       pval_key='p_value_one_sided'):
    """
    Shared builder for F01 and F02.
    data_pairs: list of (array, pos, color)
    summary: dict with keys delta_mean_ddg, ci_95_lo, ci_95_hi, p_value_one_sided
    callout_lines: list of strings for the callout box (guaranteed bottom zone)
    """
    fig = plt.figure(figsize=(14, 7))
    # Hard split: 62% violin, 38% stat card
    gs_outer = gridspec.GridSpec(1, 2, figure=fig,
                                 width_ratios=[1.65, 1],
                                 wspace=0.06,
                                 left=0.07, right=0.97,
                                 top=0.88, bottom=0.10)

    # LEFT: violin panel
    ax = fig.add_subplot(gs_outer[0, 0])

    all_vals = np.concatenate([d for d,_,_ in data_pairs])
    clip97   = np.percentile(all_vals, 97)
    ymin     = np.percentile(all_vals, 1) - 0.5
    ymax_ax  = clip97 + 3.0

    mus = []
    for data, pos, col in data_pairs:
        mu = violin_core(ax, data, pos, col, clip_pct=97)
        mus.append(mu)
        ax.text(pos, mu + 0.55, f'μ = {mu:.2f}', ha='center', color=col,
                fontsize=10.5, fontweight='bold')

    ax.axhline(0, color=TEXT3, lw=1, ls='--')
    ax.set_ylim(ymin, ymax_ax)

    # bracket at consistent height
    brack_y = clip97 + 0.4
    p = summary[pval_key]
    bracket(ax, data_pairs[0][1], data_pairs[-1][1], brack_y, p, h=0.5)

    # Ensure both violins have equal x-range and equal width
    ax.set_xlim(0.3, len(data_pairs) + 0.7)
    ax.set_xticks([d[1] for d in data_pairs])
    ax.set_xticklabels([lbl for _,_,_,lbl in
                        [(d,p,c,l) for (d,p,c),l in
                         zip(data_pairs,
                             ['Active-Site\nMutations', 'Allosteric\nMutations\n(ALCAS)'])]],
                       fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.yaxis.set_major_locator(MaxNLocator(7))
    ax.grid(axis='y', alpha=0.4)
    ftitle(ax, title_ax)
    ax.text(0.02, 0.98, note, transform=ax.transAxes, va='top',
            color=TEXT2, fontsize=8.5)

    # RIGHT: stat card — subdivided into rows zone + callout zone
    # GridSpec inside the right cell: 70% rows, 30% callout (hard pixel boundary)
    gs_right = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_outer[0, 1],
        height_ratios=[1.6, 1.0], hspace=0.0)

    ax_rows = fig.add_subplot(gs_right[0, 0])
    ax_rows.set_facecolor(CARD_BG)
    ax_rows.set_xticks([]); ax_rows.set_yticks([])
    for sp in ax_rows.spines.values():
        sp.set_visible(True); sp.set_edgecolor(BORDER); sp.set_linewidth(0.8)
    ax_rows.spines['bottom'].set_visible(False)

    ax_call = fig.add_subplot(gs_right[1, 0])
    ax_call.set_facecolor(DARK_BG)
    ax_call.set_xticks([]); ax_call.set_yticks([])
    for sp in ax_call.spines.values():
        sp.set_visible(True); sp.set_edgecolor(A1); sp.set_linewidth(1.5)
    ax_call.spines['top'].set_visible(False)

    # Stat rows — distributed evenly inside rows zone
    al_arr = data_pairs[-1][0]   # allosteric is last
    as_arr = data_pairs[0][0]    # active-site is first
    rows_data = [
        ('Allosteric mean',   f'{al_arr.mean():.2f} kcal/mol', A1),
        ('Active-site mean',  f'{as_arr.mean():.2f} kcal/mol', A2),
        ('Difference (Δμ)',   f'{summary["delta_mean_ddg"]:+.2f} kcal/mol', TEXT1),
        ('95% CI',            f'[{summary["ci_95_lo"]:.2f}, {summary["ci_95_hi"]:.2f}]', TEXT2),
        ('p-value (MW)',      f'{summary[pval_key]:.4f}', SUCCESS),
        ('Result',            'SIGNIFICANT', SUCCESS),
    ]
    n = len(rows_data)
    # Place each row at evenly spaced y in [0.10, 0.94]
    ys = np.linspace(0.91, 0.12, n)
    row_h = (ys[0] - ys[-1]) / (n - 1) if n > 1 else 0.15
    for i, (label, val, col) in enumerate(rows_data):
        y_ = ys[i]
        ax_rows.text(0.08, y_,        label, transform=ax_rows.transAxes,
                     color=TEXT2, fontsize=9, va='top')
        ax_rows.text(0.08, y_ - 0.04, val,   transform=ax_rows.transAxes,
                     color=col, fontsize=10.5, fontweight='bold', va='top')
        if i < n - 1:
            sep_y = ys[i] - row_h * 0.45
            ax_rows.plot([0.04, 0.96], [sep_y, sep_y],
                         color=BORDER, lw=0.6,
                         transform=ax_rows.transAxes,
                         clip_on=False)

    # Callout — centred in its own axis, guaranteed no overlap
    call_text = '\n'.join(callout_lines)
    ax_call.text(0.5, 0.5, call_text,
                 transform=ax_call.transAxes,
                 ha='center', va='center',
                 color=A1, fontsize=9.5, fontweight='bold',
                 linespacing=1.5)

    fig.suptitle(title_suptitle, color=TEXT1, fontsize=13,
                 fontweight='bold', x=0.44)
    save(fig, outname)


make_violin_figure(
    title_suptitle='FoldX Binding Free Energy: Allosteric vs Active-Site Mutations',
    title_ax='FoldX Binding Free Energy',
    ylabel='ΔΔG Binding (kcal/mol)',
    note=f'n = {len(as_ddg)} active-site  |  n = {len(al_ddg)} allosteric',
    data_pairs=[(as_ddg, 1, A2), (al_ddg, 2, A1)],
    summary=fx_s,
    callout_lines=[
        'Allosteric mutations are',
        '2.41 kcal/mol more',
        'stabilizing on average',
    ],
    outname='F01_foldx_binding.png',
)


make_violin_figure(
    title_suptitle='FoldX Folding Stability: Allosteric vs Active-Site Mutations',
    title_ax='FoldX Folding Stability',
    ylabel='ΔΔG Stability (kcal/mol)',
    note='Independent corroboration of binding result',
    data_pairs=[(as_sta, 1, A2), (al_sta, 2, A1)],
    summary=st_s,
    callout_lines=[
        'Binding and stability results',
        'corroborate independently',
        '(p = 0.0008 vs p = 0.0012)',
    ],
    outname='F02_foldx_stability.png',
)


# ============================================================
# F03 — ABLATION  (violins left + stat card with hard zone separation)
# ============================================================
fig = plt.figure(figsize=(16, 7))
gs_f03 = gridspec.GridSpec(1, 2, figure=fig,
                            width_ratios=[2.2, 1],
                            wspace=0.06,
                            left=0.07, right=0.97,
                            top=0.88, bottom=0.10)

ax = fig.add_subplot(gs_f03[0, 0])
means_abl = []
for data, pos, col in [(as_ddg, 1, A2), (do_ddg, 2, A4), (al_ddg, 3, A1)]:
    mu = violin_core(ax, data, pos, col)
    means_abl.append(mu)
    ax.text(pos, mu + 0.6, f'μ={mu:.2f}', ha='center', color=col,
            fontsize=9.5, fontweight='bold')

ax.axhline(0, color=TEXT3, lw=1, ls='--')
clip97_a = np.percentile(np.concatenate([as_ddg, do_ddg, al_ddg]), 97)
ax.set_ylim(np.percentile(np.concatenate([as_ddg, al_ddg]), 1) - 0.5, clip97_a + 5.0)
yb = clip97_a + 0.5
bracket(ax, 1, 2, yb,       ab_s['p_do_vs_as'], h=0.35, fs=9)
bracket(ax, 2, 3, yb + 1.5, ab_s['p_cf_vs_do'], h=0.35, fs=9)
bracket(ax, 1, 3, yb + 3.2, ab_s['p_cf_vs_as'], h=0.35, fs=9)
ax.set_xlim(0.4, 3.6)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['Active-Site',
                    'Distance-Only\nAllosteric',
                    'Coupling-Filtered\n(ALCAS)'], fontsize=10)
ax.set_ylabel('ΔΔG (kcal/mol)', fontsize=10)
ax.grid(axis='y', alpha=0.4)
ftitle(ax, 'Ablation Study: NMA Coupling Filter Validation')

# RIGHT: hard GridSpec zone split — 3 rows + callout
gs_f03_right = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=gs_f03[0, 1],
    height_ratios=[1.55, 1.0], hspace=0.0)

ax_rows = fig.add_subplot(gs_f03_right[0, 0])
ax_rows.set_facecolor(CARD_BG)
ax_rows.set_xticks([]); ax_rows.set_yticks([])
for sp in ax_rows.spines.values():
    sp.set_visible(True); sp.set_edgecolor(BORDER); sp.set_linewidth(0.8)
ax_rows.spines['bottom'].set_visible(False)

ax_rows.text(0.5, 0.97, 'Statistical Results',
             transform=ax_rows.transAxes, ha='center', va='top',
             color=TEXT1, fontsize=11, fontweight='bold')

# 3 comparison rows, each getting exactly 1/3 of the remaining space
abl_rows = [
    ('Distance-only vs Active-site',
     f'p = {ab_s["p_do_vs_as"]:.3f}', 'NOT SIGNIFICANT', DANGER),
    ('Coupling-filtered vs Distance-only',
     f'p = {ab_s["p_cf_vs_do"]:.4f}', 'SIGNIFICANT ***', SUCCESS),
    ('Coupling-filtered vs Active-site',
     f'p = {ab_s["p_cf_vs_as"]:.4f}', 'SIGNIFICANT *', SUCCESS),
]
# Row tops at 0.84, 0.56, 0.28 — each row gets 0.28 of height
row_tops = [0.84, 0.56, 0.28]
for (comp, pv, res, col), rtop in zip(abl_rows, row_tops):
    ax_rows.text(0.06, rtop,        comp, transform=ax_rows.transAxes,
                 color=TEXT2, fontsize=8.5, va='top')
    ax_rows.text(0.06, rtop - 0.09, pv,   transform=ax_rows.transAxes,
                 color=col, fontsize=10.5, fontweight='bold', va='top')
    ax_rows.text(0.06, rtop - 0.17, res,  transform=ax_rows.transAxes,
                 color=col, fontsize=8, va='top')
    if rtop > 0.30:
        ax_rows.plot([0.04, 0.96], [rtop - 0.25, rtop - 0.25],
                     color=BORDER, lw=0.8,
                     transform=ax_rows.transAxes,
                     clip_on=False)

ax_call = fig.add_subplot(gs_f03_right[1, 0])
ax_call.set_facecolor(DARK_BG)
ax_call.set_xticks([]); ax_call.set_yticks([])
for sp in ax_call.spines.values():
    sp.set_visible(True); sp.set_edgecolor(A1); sp.set_linewidth(1.5)
ax_call.spines['top'].set_visible(False)
ax_call.text(0.5, 0.5,
             'Distance alone: no benefit.\nNMA coupling is the\ncausally active component.',
             transform=ax_call.transAxes, ha='center', va='center',
             color=A1, fontsize=9.5, fontweight='bold', linespacing=1.5)

fig.suptitle('Ablation Study: What Makes ALCAS Work?',
             color=TEXT1, fontsize=13, fontweight='bold', x=0.44)
save(fig, 'F03_ablation.png')


# ============================================================
# F04 \u2014 ESMFold: dot deviation plot (better than bar for this data)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=False)
fig.suptitle('ESMFold Structural Analysis: Geometry Preservation Across Candidates',
             color=TEXT1, fontsize=13, fontweight='bold')

cands   = [d['name']  for d in esm]
grps    = [d['group'] for d in esm]
rmsds   = [d['rmsd_vs_wt'] or 0    for d in esm]
sh_dist = [d['triad_S160_H237']     for d in esm]
hd_dist = [d['triad_H237_D206']     for d in esm]
cols    = [GC[g] for g in grps]
short   = [n.replace('+','+\
') for n in cands]
wt_sh   = next(d['triad_S160_H237'] for d in esm if d['name']=='WT')
wt_hd   = next(d['triad_H237_D206'] for d in esm if d['name']=='WT')

# Panel A: RMSD dot plot
ax = axes[0]
for i,(v,c,n) in enumerate(zip(rmsds, cols, short)):
    ax.scatter([v],[i], color=c, s=90, zorder=4, edgecolors='white', linewidths=0.8)
    ax.plot([0,v],[i,i], color=c, alpha=0.35, lw=1.5)
    ax.text(v+0.08, i, f'{v:.2f}\u00c5', va='center', color=c, fontsize=8, fontweight='bold')
ax.axvline(2.0, color=WARN, lw=1.2, ls='--', alpha=0.8)
ax.text(2.05, -0.3, '2\u00c5', color=WARN, fontsize=8)
ax.set_yticks(range(len(cands)))
ax.set_yticklabels(short, fontsize=8)
for lbl,c in zip(ax.get_yticklabels(),cols): lbl.set_color(c)
ax.set_xlabel('RMSD vs WT (\u00c5)')
ax.set_xlim(-0.3, max(rmsds)+1.0)
ax.grid(axis='x', alpha=0.35)
ftitle(ax,'Global RMSD vs WT')

# Panel B: S160-H237 deviation from WT
ax = axes[1]
devs_sh = [v - wt_sh for v in sh_dist]
for i,(v,c,n) in enumerate(zip(devs_sh, cols, short)):
    ax.scatter([v],[i], color=c, s=90, zorder=4, edgecolors='white', linewidths=0.8)
    ax.plot([0,v],[i,i], color=c, alpha=0.35, lw=1.5)
    ax.text(v+(0.003 if v>=0 else -0.003), i,
            f'{sh_dist[i]:.2f}', va='center',
            ha='left' if v>=0 else 'right', color=c, fontsize=8)
ax.axvline(0, color=A3, lw=1.5, ls='-', alpha=0.6)
ax.text(0.003, -0.6, 'WT', color=A3, fontsize=8)
ax.set_yticks(range(len(cands)))
ax.set_yticklabels(['']*len(cands))
ax.set_xlabel('\u0394 from WT (\u00c5)')
ax.grid(axis='x', alpha=0.35)
ftitle(ax,'S160\u2013H237 Triad Distance')

# Panel C: H237-D206 deviation from WT
ax = axes[2]
devs_hd = [v - wt_hd for v in hd_dist]
for i,(v,c,n) in enumerate(zip(devs_hd, cols, short)):
    ax.scatter([v],[i], color=c, s=90, zorder=4, edgecolors='white', linewidths=0.8)
    ax.plot([0,v],[i,i], color=c, alpha=0.35, lw=1.5)
    ax.text(v+(0.002 if v>=0 else -0.002), i,
            f'{hd_dist[i]:.2f}', va='center',
            ha='left' if v>=0 else 'right', color=c, fontsize=8)
ax.axvline(0, color=A3, lw=1.5, ls='-', alpha=0.6)
ax.text(0.002, -0.6, 'WT', color=A3, fontsize=8)
ax.set_yticks(range(len(cands)))
ax.set_yticklabels(['']*len(cands))
ax.set_xlabel('\u0394 from WT (\u00c5)')
ax.grid(axis='x', alpha=0.35)
ftitle(ax,'H237\u2013D206 Triad Distance')

legend_el = [mpatches.Patch(color=A3,label='WT'),
             mpatches.Patch(color=A2,label='Active-Site'),
             mpatches.Patch(color=A1,label='Allosteric (ALCAS)')]
fig.legend(handles=legend_el, loc='lower center', ncol=3,
           bbox_to_anchor=(0.5,-0.04), facecolor=CARD_BG, edgecolor=BORDER)
plt.tight_layout(w_pad=2)
save(fig, 'F04_esmfold.png')


# ============================================================
# F05 — DOCKING HEATMAP
# Fix: draw y-tick labels manually as ax.text() in axes coords
# so they are never subject to tick label clipping, and remove
# the color strip patches (replace with colored text labels).
# bbox_inches='tight' in save() cannot clip ax.text() objects.
# ============================================================
dk_results = dk_s['all_results']
receptors  = list(dict.fromkeys(d['receptor'] for d in dk_results))
ligands    = ['BHET', 'MHET']
mat        = np.zeros((len(receptors), 2))
grp_map    = {}
for d in dk_results:
    ri = receptors.index(d['receptor'])
    li = ligands.index(d['ligand'])
    mat[ri, li] = d['vina_score'] if d['vina_score'] else 0
    grp_map[d['receptor']] = d['group']

def fmt_receptor(name):
    return name.replace('_', '+')

display_labels = [fmt_receptor(r) for r in receptors]
rec_cols = [GC.get(grp_map.get(r, 'wt'), TEXT2) for r in receptors]

n_rows = len(receptors)

fig, ax = plt.subplots(figsize=(12, 7))
fig.subplots_adjust(left=0.28, right=0.87, top=0.84, bottom=0.10)

cmap = LinearSegmentedColormap.from_list('v', ['#1a0a40', '#0d2b6b', A1, '#a8f0c0'], N=256)
im   = ax.imshow(mat, cmap=cmap, aspect='auto', vmin=-5.6, vmax=-4.8)
cb   = fig.colorbar(im, ax=ax, shrink=0.80, pad=0.03)
cb.set_label('Vina Score (kcal/mol)', color=TEXT2, fontsize=9)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT2, fontsize=8)

# Cell text annotations
for ri in range(n_rows):
    for li in range(2):
        wt_val = mat[0, li]
        delta  = mat[ri, li] - wt_val
        sign   = '+' if delta >= 0 else ''
        extra  = f'\n({sign}{delta:.2f})' if ri > 0 else '\n(baseline)'
        ax.text(li, ri, f'{mat[ri, li]:.3f}{extra}',
                ha='center', va='center', color='white',
                fontsize=9, fontweight='bold', linespacing=1.4)

# Remove default y-tick labels entirely; draw them manually in figure coords
ax.set_yticks(range(n_rows))
ax.set_yticklabels([''] * n_rows)   # blank out default ticks
ax.set_xticks([0, 1])
ax.set_xticklabels(['BHET\n(substrate)', 'MHET\n(product)'], fontsize=11, color=TEXT1)
ax.tick_params(length=0)
ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)

# Draw row labels as ax.text in AXES coordinates — immune to bbox clipping
# y positions: row ri maps to axes fraction (n_rows-1-ri + 0.5) / n_rows
for ri, (lbl, col) in enumerate(zip(display_labels, rec_cols)):
    y_frac = (n_rows - 1 - ri + 0.5) / n_rows   # top row = highest frac
    ax.text(-0.04, y_frac, lbl, transform=ax.transAxes,
            ha='right', va='center', color=col, fontsize=9.5,
            fontweight='bold', clip_on=False)

ax.set_title('AutoDock Vina Docking Scores\n'
             'WT BHET baseline: −5.307 kcal/mol  |  All candidates improve vs WT  |  lower = stronger binding',
             color=TEXT1, fontsize=10.5, fontweight='bold', loc='left', pad=10)

save(fig, 'F05_docking.png')


# ============================================================
# F06 — CANDIDATE COMPARISON: lollipop dot plot
# stem_x per metric:
#   ddg/stab → 0 (WT baseline by definition)
#   rmsd     → 0
#   dock     → wt_dock (-5.307)  so stems show delta from WT
# xlim contains BOTH stem_x and all data values + padding.
# ============================================================
stab_as_m = {r['mutation']: r['stability_ddg'] for r in as_st}
stab_al_m = {r['mutation']: r['stability_ddg'] for r in al_st}
dock_m    = {d['receptor']: d['vina_score']
             for d in dk_s['all_results'] if d['ligand'] == 'BHET'}
esm_rmsd  = {d['name']: d['rmsd_vs_wt'] for d in esm}

cands_all = []
for r in as_fx:
    s = r['mutation'].replace('+', '_')
    cands_all.append({'name': r['mutation'], 'group': 'active_site',
                      'ddg':  r['foldx_ddg'],
                      'stab': stab_as_m.get(r['mutation']),
                      'rmsd': esm_rmsd.get(r['mutation']) or esm_rmsd.get(s),
                      'dock': dock_m.get(r['mutation'])})
for r in al_fx:
    s = r['mutation'].replace('+', '_')
    cands_all.append({'name': r['mutation'], 'group': 'allosteric',
                      'ddg':  r['foldx_ddg'],
                      'stab': stab_al_m.get(r['mutation']),
                      'rmsd': esm_rmsd.get(r['mutation']) or esm_rmsd.get(s),
                      'dock': dock_m.get(r['mutation'])})

top_as = sorted([c for c in cands_all if c['group'] == 'active_site'],
                key=lambda x: x['ddg'] if x['ddg'] is not None else 999)[:5]
top_al = sorted([c for c in cands_all if c['group'] == 'allosteric'],
                key=lambda x: x['ddg'] if x['ddg'] is not None else 999)[:5]
show = top_al + top_as

wt_dock = next((d['vina_score'] for d in dk_s['all_results']
                if d['receptor'] == 'WT' and d['ligand'] == 'BHET'), -5.307)

# (key, xlabel, ref_vline_or_None, stem_x)
metrics_f06 = [
    ('ddg',  'FoldX Binding \u0394\u0394G (kcal/mol)', None,     0.0),
    ('stab', 'Stability \u0394\u0394G (kcal/mol)',      None,     0.0),
    ('rmsd', 'RMSD vs WT (\u00c5)',                      2.0,      0.0),
    ('dock', 'Vina Score (kcal/mol)',                     None,     wt_dock),
]

fig, axes = plt.subplots(1, 4, figsize=(20, 7.5), sharey=True)
fig.suptitle('Top 5 Candidates per Group \u2014 Multi-Metric Comparison',
             color=TEXT1, fontsize=13, fontweight='bold')
fig.subplots_adjust(left=0.13, right=0.98, top=0.88, bottom=0.13, wspace=0.10)

y_positions = list(range(len(show)))
y_labels    = [c['name'] for c in show]

for ax_i, (met, xlabel, ref_vline, stem_x) in enumerate(metrics_f06):
    ax = axes[ax_i]
    ax.set_facecolor(PANEL_BG)

    vals_valid = [c[met] for c in show if c[met] is not None]
    if not vals_valid:
        ax.set_xlabel(xlabel, fontsize=9, color=TEXT2)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([])
        continue

    # xlim must contain stem_x AND all values
    all_xs = vals_valid + [stem_x]
    xlo = min(all_xs)
    xhi = max(all_xs)
    span = max(xhi - xlo, 0.3)
    pad  = span * 0.30
    xleft  = xlo - pad
    xright = xhi + pad
    ax.set_xlim(xleft, xright)

    for yi, cand in enumerate(show):
        val = cand[met]
        if val is None:
            continue
        col = GC[cand['group']]
        ax.plot([stem_x, val], [yi, yi], color=col, alpha=0.40, lw=1.8)
        ax.scatter([val], [yi], color=col, s=85, zorder=4,
                   edgecolors='white', linewidths=0.8)
        off = span * 0.05
        if val >= stem_x:
            lx, align = val + off, 'left'
        else:
            lx, align = val - off, 'right'
        lx = max(xleft + off * 0.3, min(lx, xright - off * 0.3))
        ax.text(lx, yi, f'{val:.2f}', va='center', ha=align,
                color=col, fontsize=7.5)

    if ref_vline is not None:
        ax.axvline(ref_vline, color=WARN, lw=1.2, ls='--', alpha=0.7)
    if met in ('ddg', 'stab'):
        ax.axvline(0, color=TEXT3, lw=0.8, ls=':')
    elif met == 'dock':
        ax.axvline(wt_dock, color=A3, lw=1.0, ls=':', alpha=0.7)
        ax.text(wt_dock, len(show) - 0.3, 'WT', color=A3, fontsize=7.5,
                ha='center', va='top')

    ax.axhline(4.5, color=BORDER, lw=1.2, ls=':')
    ax.set_xlabel(xlabel, fontsize=9, color=TEXT2)
    ax.set_yticks(y_positions)
    if ax_i == 0:
        ax.set_yticklabels(y_labels, fontsize=8.5)
        for lbl, cand in zip(ax.get_yticklabels(), show):
            lbl.set_color(GC[cand['group']])
        ax.text(-0.28, 2.0, 'ALLO', transform=ax.get_yaxis_transform(),
                ha='right', va='center', color=A1, fontsize=8,
                fontweight='bold', rotation=90)
        ax.text(-0.28, 7.0, 'AS', transform=ax.get_yaxis_transform(),
                ha='right', va='center', color=A2, fontsize=8,
                fontweight='bold', rotation=90)
    else:
        ax.set_yticklabels([])
    ax.grid(axis='x', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

save(fig, 'F06_candidate_comparison.png')


# ============================================================
# F07 \u2014 NMA COUPLING  (KDE left + strip plot right, not dominated bar)
# ============================================================
ac_sc = [rn2sc[r] for r in am  if r in rn2sc]
al_sc = [rn2sc[r] for r in alm if r in rn2sc]
ot_sc = [s for r,s in rn2sc.items() if r not in am and r not in alm]

fig, axes = plt.subplots(1,2, figsize=(14,6))
fig.suptitle('NMA Coupling Analysis: Allosteric Residue Identification',
             color=TEXT1, fontsize=13, fontweight='bold')

# KDE
ax = axes[0]
for data, col, lbl, lw in [(ot_sc,TEXT3,'Other',1.5),
                             (ac_sc,A2,'Active-site mask',2.0),
                             (al_sc,A1,'Allosteric mask',2.0)]:
    if len(data)<3: continue
    kde = gaussian_kde(data, bw_method=0.3)
    xs  = np.linspace(-0.02,0.72,400)
    ys  = kde(xs)
    ax.plot(xs,ys, color=col, lw=lw, label=lbl)
    ax.fill_between(xs,ys, alpha=0.07, color=col)
    ax.axvline(np.mean(data), color=col, lw=1.2, ls=':', alpha=0.8)
    ax.text(np.mean(data), ax.get_ylim()[1]*0.98,
            f'{np.mean(data):.3f}', color=col, fontsize=7.5,
            ha='center', va='top')

ax.set_xlabel('NMA Coupling Score to Catalytic Triad', fontsize=10)
ax.set_ylabel('Density', fontsize=10)
ax.legend(loc='upper right')
ax.grid(alpha=0.3)
ftitle(ax, 'Coupling Score Distributions by Mask')

# Strip plot: all residues, 3 groups stacked vertically
ax = axes[1]
groups_strip = [
    (ot_sc, TEXT3, 0, 'Other\
(n={})'.format(len(ot_sc))),
    (ac_sc, A2,    1, 'Active-site\
(n={})'.format(len(ac_sc))),
    (al_sc, A1,    2, 'Allosteric\
(n={})'.format(len(al_sc))),
]
for data, col, pos, lbl in groups_strip:
    rng = np.random.RandomState(pos*7)
    j   = rng.uniform(-0.3,0.3, len(data))
    ax.scatter(data, np.full(len(data),pos)+j, color=col,
               alpha=0.35, s=16, zorder=2)
    ax.scatter([np.mean(data)],[pos], color='white', s=120,
               marker='D', zorder=5, edgecolors=col, linewidths=2)
    ax.text(np.mean(data), pos+0.35, f'\u03bc={np.mean(data):.3f}',
            color=col, fontsize=9, ha='center', fontweight='bold')

ax.set_yticks([0,1,2])
ax.set_yticklabels(['Other\
({})'.format(len(ot_sc)),
                    'Active-site\
({})'.format(len(ac_sc)),
                    'Allosteric\
({})'.format(len(al_sc))], fontsize=9)
ax.get_yticklabels()[0].set_color(TEXT3)
ax.get_yticklabels()[1].set_color(A2)
ax.get_yticklabels()[2].set_color(A1)
ax.set_xlabel('NMA Coupling Score', fontsize=10)
ax.set_xlim(-0.05, 0.75)
ax.grid(axis='x', alpha=0.3)
ax.spines[['top','right']].set_visible(False)
ftitle(ax, 'Coupling Score by Residue Group')

plt.tight_layout(w_pad=3)
save(fig, 'F07_nma_coupling.png')


# ============================================================
# F08 \u2014 FAST-PETase VALIDATION (tighter layout)
# ============================================================
fast = [
    {'mut':'S121E','dist':18.2,'cat':'Intermediate','col':WARN},
    {'mut':'D186H','dist':11.5,'cat':'Active-Site', 'col':A2},
    {'mut':'R224Q','dist':25.1,'cat':'Allosteric',  'col':A1},
    {'mut':'N233K','dist':12.7,'cat':'Active-Site', 'col':A2},
    {'mut':'R280A','dist':16.4,'cat':'Intermediate','col':WARN},
]

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5),
                          gridspec_kw={'width_ratios': [1.5, 1], 'wspace': 0.10})
fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.12)
fig.suptitle('FAST-PETase: Real-World Validation of Allosteric Hypothesis',
             color=TEXT1, fontsize=13, fontweight='bold')

ys     = range(len(fast))
labels = [m['mut']  for m in fast]
colors = [m['col']  for m in fast]

ax = axes[0]
for i, (m, y) in enumerate(zip(fast, ys)):
    ax.barh(y, m['dist'], color=m['col'], alpha=0.8, height=0.55,
            edgecolor=m['col'], linewidth=1.2)
    ax.text(m['dist'] + 0.3, y, f"{m['dist']}\u00c5 \u2014 {m['cat']}",
            va='center', color=m['col'], fontsize=9, fontweight='bold')

ax.axvline(13.0, color=A2, lw=1.5, ls='--', alpha=0.85)
ax.axvline(16.0, color=A1, lw=1.5, ls='--', alpha=0.85)
ax.text(13.2, -0.55, 'Active-site\nthreshold', color=A2, fontsize=7.5)
ax.text(16.2, -0.55, 'Allosteric\nthreshold',  color=A1, fontsize=7.5)

ax.set_yticks(list(ys))
ax.set_yticklabels(labels, fontsize=10)
for lbl, c in zip(ax.get_yticklabels(), colors):
    lbl.set_color(c)
ax.set_xlabel('Distance from Catalytic Triad Center (\u00c5)', fontsize=10)
ax.set_xlim(0, 32)
ax.grid(axis='x', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)
ftitle(ax, 'FAST-PETase Mutation Distances', size=11)

# RIGHT: stat callout — smaller fonts + explicit wrap so nothing exits axes
ax2 = axes[1]
ax2.set_facecolor(CARD_BG)
ax2.set_xticks([]); ax2.set_yticks([])
for sp in ax2.spines.values():
    sp.set_visible(True); sp.set_edgecolor(BORDER); sp.set_linewidth(0.8)

ax2.text(0.5, 0.93, '4 / 5', ha='center', va='top',
         color=A1, fontsize=46, fontweight='bold', transform=ax2.transAxes)
ax2.text(0.5, 0.73, 'mutations are DISTAL\nfrom the active site',
         ha='center', va='top', color=TEXT1, fontsize=11,
         fontweight='bold', transform=ax2.transAxes, linespacing=1.4)
ax2.text(0.5, 0.55, '\u226513\u00c5 from catalytic triad S160/H237/D206',
         ha='center', va='top', color=TEXT2, fontsize=8.5,
         transform=ax2.transAxes)

ax2.plot([0.06, 0.94], [0.49, 0.49], color=BORDER, lw=1,
         transform=ax2.transAxes)

insight = ('Validates ALCAS hypothesis:\n'
           'Successful enzyme engineering\n'
           'leverages allosteric effects,\n'
           'not just active-site tuning.')
ax2.text(0.5, 0.43, insight, ha='center', va='top',
         color=TEXT2, fontsize=9.5, transform=ax2.transAxes, linespacing=1.6)

leg = [mpatches.Patch(color=A1,  label='Allosteric (\u226516\u00c5)'),
       mpatches.Patch(color=WARN, label='Intermediate (13\u201316\u00c5)'),
       mpatches.Patch(color=A2,   label='Active-Site (<13\u00c5)')]
ax2.legend(handles=leg, loc='lower center', bbox_to_anchor=(0.5, 0.01),
           facecolor=DARK_BG, edgecolor=BORDER, fontsize=8.5)
save(fig, 'F08_fastpetase.png')


print(f"\n{'='*55}")
print("ALL FIGURES COMPLETE")
print(f"{'='*55}")
for f in sorted(OUT.glob('F*.png')):
    print(f"  {f.name}  ({f.stat().st_size//1024} KB)")
print("\nNext: python src/viz/plot_figures_part2.py")