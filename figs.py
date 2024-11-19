#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from statannot import add_stat_annotation
from scipy.stats import zscore, spearmanr


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
color_SR = 'blue'
color_SR_high = 'darkblue' # darkblue
color_SR_low = 'cornflowerblue'
color_entropy = 'red'
color_entropy_high = 'firebrick' # darkred
color_entropy_low = 'lightcoral'
color_pp_early = 'grey'
color_pp_late = 'lightgrey'
color_int_high = 'darkviolet'
color_int_low = 'thistle'
font_weight = 'heavy'


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

# median-split of OTT
df['ipt_median'] = df.ipt_log_z.median() # group-level
df.loc[df.ipt_log_z <= df.ipt_median, 'ipt_level'] = 'short'
df.loc[df.ipt_log_z > df.ipt_median, 'ipt_level'] = 'long'

# median split step number to define "planning phase"
df['step_median'] = df.step.median()  # group-level
df.loc[df.step <= df.step_median, 'planning_phase'] = 'early'
df.loc[df.step > df.step_median, 'planning_phase'] = 'late'


df.rename(columns={'planning_phase':'Online phase',
                   'H_level':'LTE level',
                   'ipt_level': 'OTT length',
                   'rt_log':'Logarithm of call-out times (seconds)',
                   'N_level':'SR level'},
          inplace=True)



# %% FIGURE 2B
print('\n\n\n ------------- FIGURE 2B: Off-TT vs On-TT scatter -------------')
sb.set_style('white')
fig2b = plt.figure(figsize=(6, 5))

grouped = df.groupby(['study', 'subj', 'run'])
on_tt = grouped['cot_log'].sum() / grouped['pdist_to_dest'].max()
off_tt = grouped['ipt_log'].max() / grouped['pdist_to_dest'].max()

on_tt = on_tt.groupby(['subj']).transform(zscore)
off_tt = off_tt.groupby(['subj']).transform(zscore)
dftt = pd.DataFrame(data={'on-tt':on_tt, 'off-tt':off_tt}).dropna()
corr, p_value = spearmanr(dftt['on-tt'], dftt['off-tt'])
sb.regplot(data=dftt, x='off-tt', y='on-tt', color='k', ci=None, scatter_kws={'alpha':1., 's':4})
ax = plt.gca()
plt.annotate(r'$\rho = {:.3f}$' '\n' r'$p = {:.3g}$'.format(corr, p_value),
             xy=(0.05, 1), xycoords='axes fraction',
             horizontalalignment='left', verticalalignment='top')
sb.despine(ax=ax, top=True, right=True, left=False, bottom=False)
ax.set_ylabel('Online thinking time [distance normalized]')
ax.set_xlabel('Offline thinking time [distance normalized]')

if save_output:
    save_figure(fig=fig2b, file_ext='.pdf', fname_base='fig2b', figdir=resultsdir, lowres=False)
    save_figure(fig=fig2b, file_ext='.png', fname_base='fig2b', figdir=resultsdir, lowres=False)
else:
    plt.show()


# %% FIGURE 3C: variable correlation heatmap
print('\n\n\n ------------- FIGURE 3C: variable correlation heatmap -------------')
sb.set_style('white')
fig3c, axis = plt.subplots(1, 1, figsize=(9, 8))

dflim = df[['step', 'steps', 'segment_pdist_z', 'ed_z', 'pd_z', 'edist_to_dest_z', 'pdist_to_dest_z', 'sum_deviation_z',
            'sum_target_deviation_z', 'N_dis_avg_z', 'H_z']]
dflim.rename(columns={'step': 'plan step number',
                      'steps': 'No. of planning steps [route total]',
                      'ed_z': 'direct distance [route total]',
                      'pd_z': '**streetwise distance [route total]',
                      'segment_pdist_z': '***street segment length',
                      'pdist_to_dest_z': '**streetwise distance to dest.',
                      'edist_to_dest_z': 'direct distance to dest.',
                      'sum_deviation_z': 'segment angular deviation',
                      'sum_target_deviation_z': '*destination angular deviation',
                      'N_dis_avg_z': '**SR[start, state]',
                      'H_z': 'LTE[state]'}, inplace=True)
rho = dflim.corr()
heatmap = sb.heatmap(ax=axis, data=rho, annot=False, cbar=True, cmap="plasma",  cbar_kws={'shrink': 0.7}, robust=True, square=True, linewidths=1, linecolor='black')
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=30, ha='right')
axis.xaxis.tick_bottom()
axis.yaxis.tick_left()
axis.set_title('Correlation matrix for spatial and theoretic predictors', pad=20)
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.tight_layout()

if save_output:
    save_figure(fig=fig3c, file_ext='.pdf', fname_base='fig3c', figdir=resultsdir, lowres=False)
    save_figure(fig=fig3c, file_ext='.png', fname_base='fig3c', figdir=resultsdir, lowres=False)
else:
    plt.show()




# %% FIGURE 4A: OTT x SR interaction influences RT
print('\n\n\n ------------- FIGURE 4A: OTT x SR interaction influences RT -------------')
sb.set_style('white')
fig4a = plt.figure(figsize=(5, 4))
sb.barplot(data=df, x='OTT length', y='rt', hue='SR level', errorbar='se',
           order=['short', 'long'], hue_order=['low', 'high'], palette=[color_SR_low, color_SR_high])
box_pairs = [
    (("short", "low"), ("long", "low")),
    (("short", "high"), ("long", "high")),
]
ax = plt.gca()
test_results = add_stat_annotation(ax, data=df, x='OTT length', y='rt', hue='SR level',
                                   order=['short', 'long'], hue_order=['low', 'high'],
                                   box_pairs=box_pairs,
                                   test='t-test_ind', perform_stat_test=True,
                                   text_format='star',
                                   loc='outside', verbose=2)
sb.despine(ax=ax, top=True, right=True, left=False, bottom=False)
ax.set_title('Offline thinking time x successor representation \n accounts for variability in online thinking time', pad=20)
ax.set_ylabel('Call-out time (seconds) [logarithmic scale]')
ax.set_xlabel('Offline thinking time')
ax.set_yscale('log')
ax.set_ylim([1., 4])
ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
L = ax.get_legend()
L.get_texts()[0].set_text('LOW')
L.get_texts()[1].set_text('HIGH')
L.set_bbox_to_anchor((1., 1.1))

if save_output:
    save_figure(fig=fig4a, file_ext='.pdf', fname_base='fig4a', figdir=resultsdir, lowres=False)
    save_figure(fig=fig4a, file_ext='.png', fname_base='fig4a', figdir=resultsdir, lowres=False)
else:
    plt.show()



# %% FIGURE 4B: OTT x LTE interaction influences COT
print('\n\n\n ------------- FIGURE 4B: OTT x LTE interaction influences COT -------------')
sb.set_style('white')
fig4b = plt.figure(figsize=(5, 4))
sb.barplot(data=df, x='OTT length', y='rt', hue='LTE level', errorbar='se',
           order=['short', 'long'], hue_order=['low', 'high'], palette=[color_entropy_low, color_entropy_high])
box_pairs = [
    (("short", "low"), ("long", "low")),
    (("short", "high"), ("long", "high")),
]
ax = plt.gca()
test_results = add_stat_annotation(ax, data=df, x='OTT length', y='rt', hue='LTE level',
                                   order=['short', 'long'], hue_order=['low', 'high'],
                                   box_pairs=box_pairs,
                                   test='t-test_ind', perform_stat_test=True,
                                   text_format='star',
                                   loc='outside', verbose=2)
sb.despine(ax=ax, top=True, right=True, left=False, bottom=False)
ax.set_title('Offline thinking time x local transition entropy \n accounts for variability in online thinking time', pad=20)
ax.set_ylabel('Call-out time (seconds) [logarithmic scale]')
ax.set_xlabel('Offline thinking time')
ax.set_yscale('log')
ax.set_ylim([1., 4])
ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
L = ax.get_legend()
L.get_texts()[0].set_text('LOW')
L.get_texts()[1].set_text('HIGH')
L.set_bbox_to_anchor((1., 1.1))

if save_output:
    save_figure(fig=fig4b, file_ext='.pdf', fname_base='fig4b', figdir=resultsdir, lowres=False)
    save_figure(fig=fig4b, file_ext='.png', fname_base='fig4b', figdir=resultsdir, lowres=False)
else:
    plt.show()






# %% SM FIGURE 1A: SR x entropy interaction
print('\n\n\n ------------- SM FIGURE 1A: SR x entropy interaction -------------')
sb.set_style('white')
smfig1a, axis = plt.subplots(1, 1, figsize=(7, 8))

sb.barplot(ax=axis, data=df, x='SR level', y='rt', hue='LTE level', errorbar='se', order=['low', 'high'],
           hue_order=['low', 'high'], palette=[color_entropy_low, color_entropy_high])
box_pairs = [
    (("low", "low"), ("low", "high")),
    (("high", "low"), ("high", "high")),
    (("low", "low"), ("high", "low")),
    (("low", "high"), ("high", "high"))
]
test_results = add_stat_annotation(axis, data=df, x='SR level', y='rt_log_z', hue='LTE level',
                                   box_pairs=box_pairs,
                                   test='t-test_ind', perform_stat_test=True,
                                   text_format='star', text_offset=-1,
                                   use_fixed_offset=False,
                                   loc='outside', verbose=2)

axis.set_title('Successor representation x local transition entropy \n accounts for variability in online thinking time', pad=15)
axis.set_ylabel('Call-out time (seconds) \n [logarithmic scale]')
axis.set_xlabel('Successor representation value at London junction')
L = axis.get_legend()
L.get_texts()[0].set_text('LOW')
L.get_texts()[0].set_color(color_entropy_low)
L.get_texts()[0].set_weight('heavy')
L.get_texts()[1].set_text('HIGH')
L.get_texts()[1].set_color(color_entropy_high)
L.get_texts()[1].set_weight('heavy')
L.get_title().set_text('Local transition entropy (LTE)')
L._set_loc(loc=(0.15, -0.4))
sb.despine(ax=axis, top=True, right=True, left=False, bottom=False)
axis.set_yscale('log')
axis.set_ylim([1., 3.])
axis.yaxis.set_minor_formatter(mticker.ScalarFormatter())
axis.set_yticklabels(['1.0', '1.0']) # workaround
axis. set_xticklabels(['LOW', 'HIGH'])
axis.get_xticklabels()[0].set_color(color_SR_low)
axis.get_xticklabels()[0].set_weight('heavy')
axis.get_xticklabels()[1].set_color(color_SR_high)
axis.get_xticklabels()[1].set_weight('heavy')

# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.tight_layout()

if save_output:
    save_figure(fig=smfig1a, file_ext='.pdf', fname_base='smfig1a', figdir=resultsdir, lowres=False)
    save_figure(fig=smfig1a, file_ext='.png', fname_base='smfig1a', figdir=resultsdir, lowres=False)
else:
    plt.show()



# %% SM FIGURE 1B: LTE x planning phase interaction
print('\n\n\n ------------- SM FIGURE 1B: LTE x planning phase interaction -------------')
sb.set_style('white')
smfig1b = plt.figure(figsize=(5,4))
sb.barplot(data=df[df['SR level'] == 'high'], x='LTE level', y='rt', hue='Online phase', errorbar='se',
           order=['low', 'high'], hue_order=['early', 'late'], palette=[color_pp_early, color_pp_late])
box_pairs = [
    (("low", "early"), ("low", "late")),
    (("high", "early"), ("high", "late")),
]
ax = plt.gca()
test_results = add_stat_annotation(ax, data=df[df['SR level'] == 'high'], x='LTE level', y='rt_log_z', hue='Online phase',
                                   box_pairs=box_pairs,
                                   order=['low', 'high'], hue_order=['early', 'late'],
                                   test='t-test_ind', perform_stat_test=True,
                                   text_format='star',
                                   loc='outside', verbose=2)
sb.despine(ax=ax, top=True, right=True, left=False, bottom=False)
ax.set_title('Online phase x local transition entropy \n accounts for variability in online thinking time', pad=20)
ax.set_ylabel('Call-out time (seconds) [logarithmic scale]')
ax.set_xlabel('Local transition entropy at London junction')
ax.set_yscale('log')
ax.set_ylim([1., 4])
ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
L = ax.get_legend()
L.get_texts()[0].set_text('early')
L.get_texts()[1].set_text('late')
L.get_title().set_text('Online phase')
# L._set_loc(loc=(0.25, -0.5))
L.set_bbox_to_anchor((1., 1.1))
ax.set_ylim([1., 4])

if save_output:
    save_figure(fig=smfig1b, file_ext='.pdf', fname_base='smfig1b', figdir=resultsdir, lowres=False)
    save_figure(fig=smfig1b, file_ext='.png', fname_base='smfig1b', figdir=resultsdir, lowres=False)
else:
    plt.show()
