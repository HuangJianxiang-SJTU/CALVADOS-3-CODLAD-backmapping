"""Radar plot for CG-to-AA backmapping benchmark — three-condition comparison.

Data source: logs/figure3_step100/three_condition_decomposition.csv
Helix: 5-system BMRB-validated subset (PED00003/16/184/229/536)
Cβ and Pro φ normalization: ref/obs (allows >1.0 for improvement over reference)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

metrics = [
    "Rama favored\n(general, 1−JS)",
    r"C$\beta$ deviation",
    r"Pro $\varphi$ mean",
    "Rotamer favored\n(χ₁)",
    "Clash density",
    "Helix fraction",
    r"$R_g$",
]

# Canonical raw data from three_condition_decomposition.csv (2026-05-18)
# Rama: 1−JS(general); Cβ: ref/obs; Pro φ: |ref|/|obs|
# Rotamer: obs/ref; Clash: 1−|Δ|/ref; Helix: obs/ref; Rg: 1−|Δ|/ref
ped_reference = [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000]
ped_codlad    = [0.948, 1.032, 1.017, 0.863, 0.899, 1.008, 0.999]
cg_codlad     = [0.840, 0.496, 0.815, 0.852, 0.941, 0.005, 0.983]

n = len(metrics)
angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
angles += angles[:1]

ped_ref_closed   = ped_reference + [ped_reference[0]]
ped_codlad_closed = ped_codlad + [ped_codlad[0]]
cg_codlad_closed  = cg_codlad + [cg_codlad[0]]

colors = {
    'ped_ref':    '#2c7bb6',
    'ped_codlad': '#fdae61',
    'cg_codlad':  '#d7191c',
    'ring':       '#888888',
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
})

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
fig.patch.set_facecolor('white')
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Radial grid
ax.set_ylim(0, 1.10)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=13, color='#666666')
ax.set_rlabel_position(30)

# Reference ring at 1.0
ax.plot(np.linspace(0, 2 * np.pi, 200), np.full(200, 1.0),
        color=colors['ring'], linewidth=0.8, linestyle='--', alpha=0.5, zorder=0)

# Traces
ax.fill(angles, ped_ref_closed, alpha=0.06, color=colors['ped_ref'])
ax.plot(angles, ped_ref_closed, color=colors['ped_ref'], lw=2.2,
        label='PED reference')

ax.fill(angles, ped_codlad_closed, alpha=0.08, color=colors['ped_codlad'])
ax.plot(angles, ped_codlad_closed, color=colors['ped_codlad'], lw=2.2,
        label='PED + CODLAD')

ax.fill(angles, cg_codlad_closed, alpha=0.10, color=colors['cg_codlad'])
ax.plot(angles, cg_codlad_closed, color=colors['cg_codlad'], lw=2.2,
        label='CG + CODLAD')

# Axis labels — larger font, pushed away from center
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=13, fontweight='medium')
ax.tick_params(axis='x', pad=18)

# Legend (no frame)
ax.legend(loc='upper left', bbox_to_anchor=(-0.15, 1.05),
          fontsize=13, frameon=False)

# Grid
ax.grid(True, alpha=0.25, linewidth=0.5)
ax.spines['polar'].set_visible(False)

plt.tight_layout()
fig.savefig('logs/figure5/figure5_radar.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
fig.savefig('logs/figure5/figure5_radar.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close(fig)
print('Saved: logs/figure5/figure5_radar.png, logs/figure5/figure5_radar.pdf')
