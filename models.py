#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sb
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
import os


# settings
out_fname =  os.path.join(os.getcwd(), 'data.csv')
save_output = True
resultsdir = os.path.join(os.getcwd(), 'results')
if not os.path.exists(resultsdir):
    os.makedirs(resultsdir)
sb.set_style('white')
sb.set_context('paper')
sb.set(rc={"lines.linewidth": 1.5})
sb.set(font_scale=1.2)


def random_filename(base=''):
    import uuid
    return base + '' + uuid.uuid4().hex[:8]
    # import datetime
    # return base + str(datetime.datetime.now().date()) + '' + str(datetime.datetime.now().time()).replace(':', '.')


def save_figure(fig, figdir, fname_base, file_ext='.pdf', overwrite_figures=True, lowres=True):
    """Convenience function to save figures."""
    if figdir is not None:
        if not overwrite_figures:
            rng = '_' + random_filename(base='')
        else:
            rng = ''
        fname = fname_base + rng + file_ext
        fname_lowres = fname_base + rng + '_LOWRES' + file_ext
        fig.savefig(os.path.join(figdir, fname), transparent=False, dpi=300, bbox_inches='tight')
        if lowres:
            fig.savefig(os.path.join(figdir, fname_lowres), transparent=False, dpi=100, bbox_inches='tight')


def label_panel(ax, label='A', x=-0.15, y=1.1, fontweight='normal', fontsize='normal'):
    ax.text(x, y, label, transform=ax.transAxes, fontweight=fontweight, fontsize=fontsize, va='top', ha='right')


# %% retrieve data
df = pd.read_csv(out_fname)

# construct rt_z variable
df['rt_z'] = np.exp(df['rt_log_z'])

# median-split of entropy variable
df['H_median'] = df.H_z.median() # group-level
df.loc[df.H_z <= df.H_median, 'H_level'] = 'low'
df.loc[df.H_z > df.H_median, 'H_level'] = 'high'

# median-split of SR variable
df['N_median'] = df.N_dis_avg_z.median() # group-level
df.loc[df.N_dis_avg_z <= df.N_median, 'N_level'] = 'low'
df.loc[df.N_dis_avg_z > df.N_median, 'N_level'] = 'high'

# median-split of TH variable
df['TH_median'] = df.TH_dis_z.median() # group-level
df.loc[df.TH_dis_z <= df.TH_median, 'TH_level'] = 'low'
df.loc[df.TH_dis_z > df.TH_median, 'TH_level'] = 'high'

# median-split of IPT
df['ipt_median'] = df.ipt_log_z.median() # group-level
df.loc[df.ipt_log_z <= df.ipt_median, 'ipt_level'] = 'short'
df.loc[df.ipt_log_z > df.ipt_median, 'ipt_level'] = 'long'

# median split step number to define "planning phase"
df['step_median'] = df.step.median()  # group-level
df.loc[df.step <= df.step_median, 'planning_phase'] = 'early'
df.loc[df.step > df.step_median, 'planning_phase'] = 'late'

df.rename(columns={'planning_phase':'Online planning phase',
                   'H_level':'LTE level',
                   'ipt_level': 'IPT length',
                   'rt_log':'Logarithm of call-out times (seconds)',
                   'N_level':'SR level'},
          inplace=True)

AIC = []

# %%
print('\n\n\n ------------- Linear Model 0: ONLINE (no theoretic variables) -------------')
formula = "rt_log_z ~ ipt_log_z + step + steps + ed_z + pd_z + segment_pdist_z + pdist_to_dest_z + edist_to_dest_z + sum_deviation_z + sum_target_deviation_z"
res_full = sm.ols(formula=formula, data=df).fit()
print(res_full.summary())
AIC.append(res_full.aic)


# %%
print('\n\n\n ------------- Linear Model 1: ONLINE (no interactions) -------------')
formula = "rt_log_z ~ ipt_log_z + step + steps + ed_z + pd_z + segment_pdist_z + pdist_to_dest_z + edist_to_dest_z + N_dis_avg_z + H_z + sum_deviation_z + sum_target_deviation_z"
res_full = sm.ols(formula=formula, data=df).fit()
print(res_full.summary())
AIC.append(res_full.aic)


# %%
print('\n\n\n ------------- Linear Model 2: ONLINE (excluding IPT interactions) -------------')
formula = "rt_log_z ~ ipt_log_z + step + steps + ed_z + pd_z + segment_pdist_z + pdist_to_dest_z + edist_to_dest_z + N_dis_avg_z + H_z + sum_deviation_z + sum_target_deviation_z + N_dis_avg_z*H_z"
res_full = sm.ols(formula=formula, data=df).fit()
print(res_full.summary())
AIC.append(res_full.aic)


# %%
print('\n\n\n ------------- Linear Model 3: OFFLINE (including IPTxSR and IPTxLTE interactions) -------------')
formula = "rt_log_z ~ ipt_log_z + step + steps + ed_z + pd_z + segment_pdist_z + pdist_to_dest_z + edist_to_dest_z + sum_deviation_z + sum_target_deviation_z + ipt_log_z*N_dis_avg_z +  ipt_log_z*H_z"
res_ipt = sm.ols(formula=formula, data=df).fit()
print(res_ipt.summary())
AIC.append(res_ipt.aic)



# %%
print('\n\n\n ------------- Linear Model 4: OFFLINE (including IPTxSR and IPTxLTE and SRxLTE interactions) -------------')
formula = "rt_log_z ~ ipt_log_z + step + steps + ed_z + pd_z + segment_pdist_z + pdist_to_dest_z + edist_to_dest_z + sum_deviation_z + sum_target_deviation_z + ipt_log_z*N_dis_avg_z +  ipt_log_z*H_z + N_dis_avg_z*H_z"
res_ipt = sm.ols(formula=formula, data=df).fit()
print(res_ipt.summary())
AIC.append(res_ipt.aic)



# %%
print('\n\n\n ------------- Linear Model 5: OFFLINE (including three-way SRxLTExIPT interactions) -------------')
formula = "rt_log_z ~ ipt_log_z + step + steps + ed_z + pd_z + segment_pdist_z + pdist_to_dest_z + edist_to_dest_z + sum_deviation_z + sum_target_deviation_z + N_dis_avg_z*ipt_log_z*H_z"
res_ipt = sm.ols(formula=formula, data=df).fit()
print(res_ipt.summary())
AIC.append(res_ipt.aic)


#%% model comparison plot
sb.set_style('white')
fig3d = plt.figure(figsize=(5, 4))
ax = plt.gca()
sb.barplot(x=AIC, y=np.arange(len(AIC)), color='black', orient='h')
ax.set_yticklabels(['GLM0', 'GLM1', 'GLM2', 'GLM3', 'GLM4', 'GLM5'])
ax.set_xlim([5750, 5820])
ax.set_ylabel('Model')
ax.set_xlabel('Akaike Information Criterion')
ax.grid(which='major', axis='x', color='lightgrey', linestyle='-')
sb.despine(ax=ax, top=True, right=True, left=False, bottom=False)
plt.tight_layout()
if save_output:
    save_figure(fig=fig3d, file_ext='.pdf', fname_base='fig3d', figdir=resultsdir, lowres=False)
    save_figure(fig=fig3d, file_ext='.png', fname_base='fig3d', figdir=resultsdir, lowres=False)
else:
    plt.show()


