from pathlib import Path
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import pandas as pd

font_size=21

def save_confusion_matrix(df, main_path):
    for provider, group in df.groupby('provider'):
        for noise, group in group.groupby('noise_algorithm'):
            dir = main_path + '/' + provider
            Path(dir).mkdir(parents=True, exist_ok=True)

            for noise_level, group in group.groupby('noise_level'):
                cm = group['confusion_matrix'].iloc[0]
                cm = np.array(cm)

                df_cm = pd.DataFrame(cm, index=['Negative', 'Neutral', 'Positive'],
                                            columns=['Negative', 'Neutral', 'Positive'])
                ax = sn.heatmap(df_cm, cmap='Oranges', annot=True, fmt='d')
                fig_title = provider + ' '+noise + ' ' + str(noise_level)

                plt.title(fig_title)
                plt.xlabel("Predicted Values")
                plt.ylabel("Real Values")
                fig = ax.get_figure()
                Path(dir+'/confusion_matrix/'+noise).mkdir(parents=True, exist_ok=True)

                fig.savefig(dir+'/confusion_matrix/'+noise+'/'+fig_title+'.jpg', transparent=False, dpi=250)
                plt.clf()

def save_results_plot(df,main_path):
    for provider, group in df.groupby('provider'):
        for noise, group in group.groupby('noise_algorithm'):
            dir = main_path + '/' + provider
            Path(dir).mkdir(parents=True, exist_ok=True)

            group['noise_level']= df['noise_level'].map(float)
            group = group.sort_values(by=['noise_level'], ascending=True)

            fig2 = group.plot(x='noise_level', title=noise).get_figure()
            fig2.savefig(dir+'/'+noise+'.jpg', transparent=False, dpi=250)
            plt.clf()
        plt.close('all')

# todo: add fig_title parameter to better fit RQ3?
# OBS: used in RQ1 and RQ3
def gen_results_plots_RQ1(data,main_path, noise_levels):
    gen_result_plot_RQ1_raw_fmeasure(data, main_path, noise_levels)
    gen_result_plot_RQ1_fmeasure_drop(data, main_path, noise_levels)
    gen_result_plot_RQ1_median_fmeasure(data, main_path, noise_levels)

def gen_result_plot_RQ1_median_fmeasure(data, main_path: str, noise_levels):
    f_size = font_size - 4
    # select and arrange data
    df = pd.DataFrame(data)
    df = df[['provider', 'noise_level', 'fmeasure', 'noise_algorithm']]
    df['provider'] = df['provider'].str.capitalize()

    medians = df.groupby(['provider', 'noise_level'])['fmeasure'] \
                .median() \
                .reset_index()
    
    medians = medians.pivot(index='noise_level', columns='provider', values='fmeasure')
    # plot
    ax = medians.plot(xlabel='noise level (%)',
                      ylabel="f-measure")
    ax.xaxis.label.set_fontsize(f_size)
    ax.yaxis.label.set_fontsize(f_size)
    ax.legend(prop={'size': f_size-2})
    ax.tick_params(axis='both', which='major', labelsize=f_size)
    ax.set_title(label="Providers f-measure median",
                fontdict={'fontsize': f_size})
    ax.set_xticks(noise_levels)
    ax.set_xticklabels([str(x).replace('0.', '.')
                        for x in np.round(ax.get_xticks(), 3)])
    ax.set_xlim(0, max(noise_levels))
    ax.set_ylim(0, 1)

    plt.tight_layout()

    # save to disk
    Path(main_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(main_path+'/rq1_medians.pdf')
    plt.savefig(main_path+'/rq1_medians.jpg', transparent=False, dpi=250)
    plt.close('all')

    return ax

# OBS: used in RQ1 and RQ3
def gen_result_plot_RQ1_raw_fmeasure(data, main_path, noise_levels):
    '''
    RQ1 
    3 plots, 1 per provider: 
    axis x: noise level
    axis y: f-measure of each noise
    '''
    df = pd.DataFrame(data)
    df_rq1 = df[['provider', 'noise_level', 'fmeasure', 'noise_algorithm']]

    fig, axes_list = plt.subplots(ncols=len(df_rq1.groupby('provider')))
    axes = iter(axes_list)

    for provider, group in df_rq1.groupby('provider'):
        group = group[group['noise_algorithm'] != 'no_noise']

        worst_line = group[['provider', 'noise_level', 'fmeasure']] \
            .groupby('noise_level') \
            .min() \
            .reset_index()

        mean_line = group[['provider', 'noise_level', 'fmeasure']] \
            .groupby('noise_level') \
            .mean(numeric_only=True) \
            .reset_index()

        ax = next(axes)
        # setup axis
        plt.xlabel("noise level (%)")
        ax.set_ylabel("f-measure")

        ax.xaxis.label.set_fontsize(font_size)
        ax.yaxis.label.set_fontsize(font_size)
        ax.tick_params(axis='both', which='major', labelsize=font_size)
        ax.set_title(label=provider.capitalize(),
                     fontdict={'fontsize': font_size})

        ax.set_xticks(noise_levels)
        ax.set_xticklabels([str(x).replace('0.', '.')
                           for x in np.round(ax.get_xticks(), 3)])
        ax.set_xlim(0, max(noise_levels))
        ax.set_ylim(0, 1)
        markers = itertools.cycle(['>', '+', '.', 'o', '*', 's'])

        dir = main_path
        Path(dir).mkdir(parents=True, exist_ok=True)
        for algorithm in group['noise_algorithm'].unique():
            sample: pd.DataFrame = group[group['noise_algorithm'] == algorithm]
            sample = sample.sort_values(by=['noise_level'], ascending=True)

            fig2 = sample.plot(legend=None, ax=ax, marker=next(markers), markersize=7,
                               xlabel='noise level', x='noise_level', y='fmeasure',
                               label=algorithm,
                               figsize=(15, 7.5),
                               )
        ax.plot('noise_level', 'fmeasure',
                label='Worst', data=worst_line,
                color='red', linewidth=2,
                linestyle='dashed')
        ax.plot('noise_level', 'fmeasure',
                label='Mean', data=mean_line,
                color='darkorange', linewidth=2,
                linestyle='dashed')

    lines_labels = axes_list[0].get_legend_handles_labels()
    lines, labels = [sum(lol, []) for lol in zip(lines_labels)]
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.38, wspace=0.3, hspace=0)
    plt.figlegend(lines, labels, loc='lower center',
                  ncol=3, labelspacing=0.2, bbox_to_anchor=(0.5, 0),
                  bbox_transform=plt.gcf().transFigure, fontsize=font_size)

    plt.savefig(dir+'/rq1.pdf')

    for axe in axes_list:
        axe.legend(loc='upper left', framealpha=0.5)

    fig.savefig(dir+'/rq1_full.jpg', transparent=False, dpi=250)

    plt.close('all')

    return fig

'''
RQ1 
3 plots, 1 per provider: 
axis x: noise level
axis y: f-measure of each noise
'''
# OBS: used in RQ1 and RQ3
def gen_result_plot_RQ1_fmeasure_drop(data, main_path, noise_levels):
    df = pd.DataFrame(data)
    df_rq1 = df[['provider', 'noise_level', 'fmeasure', 'noise_algorithm']]

    fig, axes_list = plt.subplots(ncols=len(df_rq1.groupby('provider')))
    axes = iter(axes_list)

    for provider, group in df_rq1.groupby('provider'):
        group = group[group['noise_algorithm'] != 'no_noise']

        ax = next(axes)
        # setup axis
        plt.xlabel("noise level (%)")
        ax.set_ylabel("f-measure drop")

        ax.xaxis.label.set_fontsize(font_size)
        ax.yaxis.label.set_fontsize(font_size)
        ax.tick_params(axis='both', which='major', labelsize=font_size)
        ax.set_title(label=provider.capitalize(),
                     fontdict={'fontsize': font_size})

        ax.set_xticks(noise_levels)
        ax.set_xticklabels([str(x).replace('0.', '.')
                           for x in np.round(ax.get_xticks(), 3)])
        ax.set_xlim(0, max(noise_levels))
        ax.set_ylim(0, 1)
        markers = itertools.cycle(['>', '+', '.', 'o', '*', 's'])

        dir = main_path
        Path(dir).mkdir(parents=True, exist_ok=True)
        for algorithm in group['noise_algorithm'].unique():
            sample: pd.DataFrame = group[group['noise_algorithm'] == algorithm]

            no_noise_value = sample[sample['noise_level'] == 0]
            sample['fmeasure drop'] = no_noise_value.iloc[0]['fmeasure'] - sample['fmeasure'] 

            sample = sample.drop(columns=['fmeasure'])
            sample = sample.sort_values(by=['noise_level'], ascending=True)

            fig2 = sample.plot(legend=None, ax=ax, marker=next(markers), markersize=7,
                               xlabel='noise level', x='noise_level', y='fmeasure drop',
                               label=algorithm,
                               figsize=(15, 7.5),
                               )

    lines_labels = axes_list[0].get_legend_handles_labels()
    lines, labels = [sum(lol, []) for lol in zip(lines_labels)]
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.33, wspace=0.3, hspace=0)
    plt.figlegend(lines, labels, loc='lower center',
                  ncol=3, labelspacing=0.2, bbox_to_anchor=(0.5, 0),
                  bbox_transform=plt.gcf().transFigure, fontsize=font_size)

    plt.savefig(dir+'/rq1_drop.pdf')

    for axe in axes_list:
        axe.legend(loc='upper left', framealpha=0.5)

    fig.savefig(dir+'/rq1_drop_full.jpg', transparent=False, dpi=250)

    plt.close('all')

    return fig

'''
RQ2 
1 grafico para cada noise 
eixo x nivel de noise 
eixo y: f-measure para provedor 
'''
# OBS: used in RQ2 and RQ3
def gen_result_plot_RQ2_raw(data,main_path, noise_levels):
    font_size_rq2 = font_size - 5
    df = pd.DataFrame(data)
    df_rq2 = df[['provider', 'noise_level', 'fmeasure','noise_algorithm']]
    group = df_rq2[df_rq2['noise_algorithm'] != 'no_noise']
    for noise, group in df_rq2.groupby('noise_algorithm'):
        # setup axis
        fig, ax = plt.subplots()
        plt.xlabel("noise level (%)")
        plt.ylabel("f-measure")

        ax.xaxis.label.set_fontsize(font_size_rq2)
        ax.yaxis.label.set_fontsize(font_size_rq2)
        ax.tick_params(axis='both', which='major', labelsize=font_size_rq2)
        
        ax.set_xticks(noise_levels)
        ax.set_xticklabels([str(x).replace('0.', '.') for x in np.round(ax.get_xticks(), 3)])
        ax.set_xlim(0.1, max(noise_levels))
        ax.set_ylim(0, 1)

        dir = main_path
        Path(dir).mkdir(parents=True, exist_ok=True)
        for provider in group['provider'].unique():
            ax.set_title(label=noise.capitalize(), fontdict={'fontsize':font_size_rq2})

            sample = group[group['provider'] == provider]
            sample = sample.sort_values(by=['noise_level'], ascending=True)
            fig2 = sample.plot(ax=ax, xlabel='noise level', x='noise_level', y='fmeasure', label=provider.capitalize()).get_figure()
            plt.legend(prop={'size': font_size_rq2})
            fig.tight_layout() 
        fig2.savefig(dir+'/'+noise+'.jpg', transparent=False, dpi=250)
        fig2.savefig(dir+'/'+noise+'.pdf')
        
        plt.clf()
    plt.close('all')