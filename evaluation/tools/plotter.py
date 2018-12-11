#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt
from pylab import setp
from matplotlib.ticker import FuncFormatter

def find_step_of_base(x, base):
    return base * 10**np.floor(np.log10(np.abs(float(x))))

def _set_boxplot_colors(boxplot_object, color):
    setp(boxplot_object['boxes'][0], color=color)
    setp(boxplot_object['caps'][0], color=color)
    setp(boxplot_object['caps'][1], color=color)
    setp(boxplot_object['whiskers'][0], color=color)
    setp(boxplot_object['whiskers'][1], color=color)
    #setp(boxplot_object['fliers'], color=color)
    setp(boxplot_object['medians'][0], color=color)

def draw_boxplot(axis, stats, position, idx_experiment):
    """
        bxpstats : list of dicts
          A list of dictionaries containing stats for each boxplot.
          Required keys are:

          - ``med``: The median (scalar float).

          - ``q1``: The first quartile (25th percentile) (scalar
            float).

          - ``q3``: The third quartile (75th percentile) (scalar
            float).

          - ``whislo``: Lower bound of the lower whisker (scalar
            float).

          - ``whishi``: Upper bound of the upper whisker (scalar
            float).

          Optional keys are:

          - ``mean``: The mean (scalar float). Needed if
            ``showmeans=True``.

          - ``fliers``: Data beyond the whiskers (sequence of floats).
            Needed if ``showfliers=True``.

          - ``cilo`` & ``cihi``: Lower and upper confidence intervals
            about the median. Needed if ``shownotches=True``.

          - ``label``: Name of the dataset (string). If available,
            this will be used a tick label for the boxplot

        positions : array-like, default = [1, 2, ..., n]
          Sets the positions of the boxes. The ticks and limits
          are automatically set to match the positions.
    """
    colors = ['blue', 'black', 'green', 'red', 'mangenta', 'cyan', 'orange']
    bxpstats = []
    bxpstats_a = dict()
    bxpstats_a['med'] = stats['median']
    bxpstats_a['q1'] = stats['q1']
    bxpstats_a['q3'] = stats['q3']
    bxpstats_a['whislo'] = stats['min']
    bxpstats_a['whishi'] = stats['max']
    bxpstats.append(bxpstats_a)
    pb = axis.bxp(bxpstats,
                  positions=position,
                  widths=0.8, vert=True,
                  showcaps=True, showbox=True, showfliers=False, )
    _set_boxplot_colors(pb, colors[idx_experiment])

def draw_rpe_boxplots(output_dir, stats, n_segments):
    """ Draw boxplots from stats:
        which is a list that contains:
            - pipeline type (string) (like S, SP or SPR):
                - "relative_errors":
                    - segment distance (float) (like 10 or 20 etc):
                        - "max"
                        - "min"
                        - "mean"
                        - "median"
                        - "q1"
                        - "q3"
                        - "rmse"
        This function iterates over the pipeline types, and for each pipeline type, it plots
        the metrics achieved for each segment length. So the boxplot has in x-axis
        the number of segments considered, and in y-axis one boxplot per pipeline.
                        """

    colors = ['blue', 'black', 'green', 'red', 'mangenta', 'cyan', 'orange']
    if isinstance(stats, dict):
        n_experiment = len(stats)
        spacing = 1

        # Precompute position of boxplots in plot.
        pos = np.arange(0, n_segments * (n_experiment + spacing), (n_experiment + spacing))

        # Init axes
        # Use different plotting config.
        plt.style.use('default')
        import matplotlib as mpl
        from matplotlib import rc
        import seaborn as sns
        sns.reset_orig()
        mpl.rcParams.update(mpl.rcParamsDefault)
        rc('font',**{'family':'serif','serif':['Cardo'],'size':16})
        rc('text', usetex=False)

        fig = plt.figure(figsize=(6,6))
        ax_pos = fig.add_subplot(211, ylabel='RPE translation [m]')
        ax_yaw = fig.add_subplot(212, ylabel='RPE rotation [deg]', xlabel='Distance travelled [m]')
        dummy_plots_pos = []
        dummy_plots_yaw = []

        idx_experiment = 0
        x_labels = []
        final_max_e_pos=0.0
        final_max_e_yaw=0.0
        segment_lengths = []
        for pipeline_key, errors in sorted(stats.items()):
            # The dummy plots are used to create the legends.
            dummy_plot_pos = ax_pos.plot([1,1], '-', color=colors[idx_experiment])
            dummy_plots_pos.append(dummy_plot_pos[0])
            dummy_plot_yaw = ax_yaw.plot([1,1], '-', color=colors[idx_experiment])
            dummy_plots_yaw.append(dummy_plot_yaw[0])
            x_labels.append(pipeline_key)
            if isinstance(errors, dict):
                assert("relative_errors" in errors)
                # Check that we have the expected number of segments
                assert(n_segments == len(errors['relative_errors']))
                idx_segment = 0
                for segment_length, stats in sorted(errors["relative_errors"].items(), key = lambda item: int(item[0])):
                    segment_lengths.append(segment_length)
                    # Find max value overall, to set max in y-axis
                    max_e_pos = stats["rpe_trans"]["max"]
                    max_e_yaw = stats["rpe_rot"]["max"]
                    #max_e_pos = 10.2+0.02
                    #max_e_yaw = 30.0
                    if max_e_pos > final_max_e_pos:
                        final_max_e_pos = max_e_pos
                    if max_e_yaw > final_max_e_yaw:
                        final_max_e_yaw = max_e_yaw
                    # Draw boxplot
                    draw_boxplot(ax_pos, stats["rpe_trans"], [idx_experiment + pos[idx_segment]],
                                 idx_experiment)
                    draw_boxplot(ax_yaw, stats["rpe_rot"], [idx_experiment + pos[idx_segment]],
                                idx_experiment)
                    idx_segment = idx_segment + 1
            else:
                raise Exception("\033[91mValue in stats should be a dict: " + errors + "\033[99m")
            idx_experiment = idx_experiment + 1

        # Create legend.
        ax_pos.legend(dummy_plots_yaw, x_labels, bbox_to_anchor=(0., 1.02, 1., .102),
                      loc=3, ncol=3, mode='expand', borderaxespad=0.)

        def _ax_formatting(ax, dummy_plots, final_max_e):
            ax.yaxis.grid(ls='--', color='0.7')
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: '%.2f'%y))
            # ax.xaxis.grid(which='major', visible=True, ls=' ')
            # ax.xaxis.grid(which='minor', visible=False)
            #ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.set_xticks(pos + 0.5*n_experiment - 0.5)
            ax.set_xticklabels(segment_lengths)
            ax.set_xlim(xmin=pos[0] - 1, xmax=pos[-1] + n_experiment + 0.2)
            ax.set_ylim(ymin=0, ymax=final_max_e)
            # Set yticks every multiple of 5 or 1. (so a tick 0.03 is now 0.00, 0.05,
            # and if 1 then a tick 0.034 is 0.03, 0.04)
            yticks = np.arange(0, final_max_e, find_step_of_base(final_max_e/5, 5))
            if len(yticks) < 4: # 4 is the minimum of yticks that we want.
                ax.set_yticks(np.arange(0, final_max_e, find_step_of_base(final_max_e/5, 1)))
            else:
                ax.set_yticks(yticks)
            for p in dummy_plots:
                p.set_visible(False)

        # give some extra space for the plot...
        final_max_e_pos += 0.05*final_max_e_pos
        final_max_e_yaw += 0.05*final_max_e_yaw
        _ax_formatting(ax_pos, dummy_plots_pos, final_max_e_pos)
        _ax_formatting(ax_yaw, dummy_plots_yaw, final_max_e_yaw)

        fig.savefig(os.path.join(output_dir, 'traj_relative_errors_boxplots.eps'),
                    bbox_inches="tight", format="eps", dpi=1200)
    else:
        raise Exception("\033[91mStats should be a dict: " + stats + "\033[99m")

    # Restore plotting config.
    from evo.tools.settings import SETTINGS
    plt.style.use('seaborn')
    # configure matplotlib and seaborn according to package settings
    sns.set(style=SETTINGS.plot_seaborn_style,
            palette=SETTINGS.plot_seaborn_palette,
            font=SETTINGS.plot_fontfamily,
            font_scale=SETTINGS.plot_fontscale
           )

    rc_params = {
        "lines.linewidth": SETTINGS.plot_linewidth,
        "text.usetex": SETTINGS.plot_usetex,
        "font.family": SETTINGS.plot_fontfamily,
        "font.serif": ['Cardo'],
        "pgf.texsystem": SETTINGS.plot_texsystem
    }
    mpl.rcParams.update(rc_params)

def draw_ape_boxplots(stats, output_dir):
    """ Draw boxplots from stats:
        which is a list that contains:
            - dataset name (string) (like V1_01_easy, MH_01_easy etc):
                - pipeline type (string) (like S, SP or SPR):
                    - "absolute_errors":
                        - "max"
                        - "min"
                        - "mean"
                        - "median"
                        - "q1"
                        - "q3"
                        - "rmse"
        This function iterates over the pipeline types, and for each pipeline type, it plots
        the metrics achieved, as a boxplot. So the boxplot has in x-axis the dataset name,
        and in y-axis one boxplot per pipeline."""
    colors = ['blue', 'black', 'green', 'red', 'mangenta', 'cyan', 'orange']
    if isinstance(stats, dict):
        n_param_values = len(stats)
        n_pipeline_types = len(stats.values()[0])
        spacing = 1

        # Precompute position of boxplots in plot.
        pos = np.arange(0, n_param_values * (n_pipeline_types + spacing),
                        (n_pipeline_types + spacing))

        # Use different plotting config.
        plt.style.use('default')
        import matplotlib as mpl
        from matplotlib import rc
        import seaborn as sns
        sns.reset_orig()
        mpl.rcParams.update(mpl.rcParamsDefault)
        rc('font',**{'family':'serif','serif':['Cardo'],'size':16})
        rc('text', usetex=False)

        # Init axis
        fig = plt.figure(figsize=(14, 6))
        ax_pos = fig.add_subplot(111, ylabel='APE translation error [m]', xlabel="Dataset")
        legend_labels = []
        legend_handles = []
        # Draw legend.
        color_id = 0
        for pipeline_type, pipeline_stats in sorted(stats.values()[0].items()):
            # The dummy plots are used to create the legends.
            dummy_plot_pos = ax_pos.plot([1,1], '-', color=colors[color_id])
            legend_labels.append(pipeline_type)
            legend_handles.append(dummy_plot_pos[0])
            color_id = color_id + 1

        idx_param_value = 0
        final_max_e_pos=0.50
        xtick_labels=[]
        pipelines_failed = dict()
        for dataset_name, pipeline_types in sorted(stats.items()):
            xtick_labels.append(dataset_name.replace('_', '\_'))
            if isinstance(pipeline_types, dict):
                idx_pipeline_type = 0
                for pipeline_type, pipeline_stats in sorted(pipeline_types.items()):
                    if isinstance(pipeline_stats, dict):
                        # Find max value overall, to set max in y-axis
                        max_e_pos = pipeline_stats["absolute_errors"]["max"]
                        # if max_e_pos > final_max_e_pos:
                           # final_max_e_pos = max_e_pos
                        # Draw boxplot
                        draw_boxplot(ax_pos, pipeline_stats["absolute_errors"],
                                     [idx_pipeline_type + pos[idx_param_value]], idx_pipeline_type)
                    else:
                        # If pipeline_stats is not a dict, then it means the pipeline failed...
                        # Just plot a cross...
                        pipelines_failed[idx_pipeline_type] = [pipeline_type, idx_param_value]
                    idx_pipeline_type = idx_pipeline_type + 1
            else:
                raise Exception("\033[91mValue in stats should be a dict: " + errors + "\033[99m")
            idx_param_value = idx_param_value + 1

        # Draw crosses instead of boxplots for pipelines that failed.
        for idx_pipeline, pipeline_type_idx_param_pair in pipelines_failed.items():
            x_middle = idx_pipeline + pos[pipeline_type_idx_param_pair[1]]
            x_1 = [x_middle - 0.5*spacing, x_middle + 0.5*spacing]
            y_1 = [0, final_max_e_pos]
            x_2 = [x_middle - 0.5*spacing, x_middle + 0.5*spacing]
            y_2 = [final_max_e_pos, 0]
            red_cross_plot = ax_pos.plot([1,1], 'xr')
            pipeline_type = pipeline_type_idx_param_pair[0]
            legend_labels.append("{} failure".format(pipeline_type))
            legend_handles.append(red_cross_plot[0])
            ax_pos.plot(x_1, y_1, '-r')
            ax_pos.plot(x_2, y_2, '-r')

        # Create legend.
        ax_pos.legend(legend_handles, legend_labels, bbox_to_anchor=(0., 1.02, 1., .102),
                      loc=3, ncol=3, mode='expand', borderaxespad=0.)

        def _ax_formatting(ax, dummy_plots, final_max_e):
            ax.yaxis.grid(ls='--', color='0.7')
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: '%.2f'%y))
            # ax.xaxis.grid(which='major', visible=True, ls=' ')
            # ax.xaxis.grid(which='minor', visible=False)
            #ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.set_xticks(pos + 0.5*n_pipeline_types - 0.5)
            ax.set_xticklabels(xtick_labels, rotation=-40, ha='left')
            ax.set_xlim(xmin=pos[0] - 1, xmax=pos[-1] + n_pipeline_types + 0.2)
            ax.set_ylim(ymin=0, ymax= final_max_e)
            yticks = np.arange(0, final_max_e, find_step_of_base(final_max_e/5, 5))
            if len(yticks) < 4:
                ax.set_yticks(np.arange(0, final_max_e, find_step_of_base(final_max_e/5, 1)))
            else:
                ax.set_yticks(yticks)
            for p in dummy_plots:
                p.set_visible(False)

        # give some extra space for the plot...
        final_max_e_pos += 0.02
        _ax_formatting(ax_pos, legend_handles, final_max_e_pos)

        fig.savefig(os.path.join(output_dir, 'datasets_ape_boxplots.eps'),
                    bbox_inches="tight", format="eps", dpi=1200)
    else:
        raise Exception("\033[91mStats should be a dict: " + stats + "\033[99m")

    # Restore plotting config.
    from evo.tools.settings import SETTINGS
    plt.style.use('seaborn')
    # configure matplotlib and seaborn according to package settings
    sns.set(style=SETTINGS.plot_seaborn_style,
            palette=SETTINGS.plot_seaborn_palette,
            font=SETTINGS.plot_fontfamily,
            font_scale=SETTINGS.plot_fontscale
           )

    rc_params = {
        "lines.linewidth": SETTINGS.plot_linewidth,
        "text.usetex": SETTINGS.plot_usetex,
        "font.family": SETTINGS.plot_fontfamily,
        "font.serif": ['Cardo'],
        "pgf.texsystem": SETTINGS.plot_texsystem
    }
    mpl.rcParams.update(rc_params)

def draw_regression_simple_boxplot_APE(param_names, stats, output_dir, max_y = -1):
    """ Draw boxpots where x-axis are the values of the parameters in param_names, and the y-axis has boxplots with APE
    performance of the pipelines in stats.
    Stats is organized as follows:
        - param_value_dir (path to directory containing results for the parameter with given value)
            - pipeline (pipeline type e.g. S, SP or SPR)
                - results (which is actually -max, -min etc !OR! False if there are no results if the pipeline failed."""
    colors = ['blue', 'black', 'green', 'red', 'mangenta', 'cyan', 'orange']
    if isinstance(stats, dict):
        n_param_values = len(stats)
        assert(n_param_values > 0)
        n_pipeline_types = len(stats.values()[0])
        spacing = 1

        # Precompute position of boxplots in plot.
        pos = np.arange(0, n_param_values * (n_pipeline_types + spacing),
                        (n_pipeline_types + spacing))

        # Use different plotting config.
        plt.style.use('default')
        import matplotlib as mpl
        from matplotlib import rc
        import seaborn as sns
        sns.reset_orig()
        mpl.rcParams.update(mpl.rcParamsDefault)
        rc('font',**{'family':'serif','serif':['Cardo'],'size':16})
        rc('text', usetex=False)

        # Init axis
        fig = plt.figure(figsize=(6,2))
        param_names_dir = ""
        for i in param_names:
            param_names_dir += str(i) + "-"
            param_names_dir = param_names_dir[:-1]
        ax_pos = fig.add_subplot(111, ylabel='APE translation error [m]', xlabel="Values of parameter: {}".format(param_names_dir))
        legend_labels = []
        legend_handles = []
        # Draw legend.
        color_id = 0
        for pipeline_type, pipeline_stats in sorted(stats.values()[0].items()): #inefficient in py2 but compatible py2 & 3
            # The dummy plots are used to create the legends.
            dummy_plot_pos = ax_pos.plot([1,1], '-', color=colors[color_id])
            legend_labels.append(pipeline_type)
            legend_handles.append(dummy_plot_pos[0])
            color_id = color_id + 1

        idx_param_value = 0
        auto_scale = False
        final_max_e_pos = 0
        if max_y < 0:
            auto_scale = True
        else:
            final_max_e_pos = max_y
        param_values_boxplots=[]
        pipelines_failed = dict()
        for param_value_boxplots, pipeline_types in sorted(stats.items()):
            param_values_boxplots.append(param_value_boxplots)
            if isinstance(pipeline_types, dict):
                idx_pipeline_type = 0
                for pipeline_type, pipeline_stats in sorted(pipeline_types.items()):
                    if isinstance(pipeline_stats, dict):
                        # Find max value overall, to set max in y-axis
                        max_e_pos = pipeline_stats["absolute_errors"]["max"]
                        if auto_scale:
                            if max_e_pos > final_max_e_pos:
                               final_max_e_pos = max_e_pos
                        # Draw boxplot
                        draw_boxplot(ax_pos, pipeline_stats["absolute_errors"],
                                     [idx_pipeline_type + pos[idx_param_value]], idx_pipeline_type)
                    else:
                        # If pipeline_stats is not a dict, then it means the pipeline failed...
                        # Just plot a cross...
                        pipelines_failed[idx_pipeline_type] = [pipeline_type, idx_param_value]
                    idx_pipeline_type = idx_pipeline_type + 1
            else:
                raise Exception("\033[91mValue in stats should be a dict: " + errors + "\033[99m")
            idx_param_value = idx_param_value + 1

        # Draw crosses instead of boxplots for pipelines that failed.
        for idx_pipeline, pipeline_type_idx_param_pair in pipelines_failed.items():
            x_middle = idx_pipeline + pos[pipeline_type_idx_param_pair[1]]
            x_1 = [x_middle - 0.5*spacing, x_middle + 0.5*spacing]
            y_1 = [0, final_max_e_pos]
            x_2 = [x_middle - 0.5*spacing, x_middle + 0.5*spacing]
            y_2 = [final_max_e_pos, 0]
            red_cross_plot = ax_pos.plot([1,1], 'xr')
            pipeline_type = pipeline_type_idx_param_pair[0]
            legend_labels.append("{} failure".format(pipeline_type))
            legend_handles.append(red_cross_plot[0])
            ax_pos.plot(x_1, y_1, '-r')
            ax_pos.plot(x_2, y_2, '-r')

        # Create legend.
        ax_pos.legend(legend_handles, legend_labels, bbox_to_anchor=(0., 1.02, 1., .102),
                      loc=3, ncol=3, mode='expand', borderaxespad=0.)

        def _ax_formatting(ax, dummy_plots, final_max_e):
            ax.yaxis.grid(ls='--', color='0.7')
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: '%.2f'%y))
            # ax.xaxis.grid(which='major', visible=True, ls=' ')
            # ax.xaxis.grid(which='minor', visible=False)
            #ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.set_xticks(pos + 0.5*n_pipeline_types - 0.5)
            ax.set_xticklabels(param_values_boxplots)
            ax.set_xlim(xmin=pos[0] - 1, xmax=pos[-1] + n_pipeline_types + 0.2)
            ax.set_ylim(ymin=0, ymax= final_max_e)
            yticks = np.arange(0, final_max_e, find_step_of_base(final_max_e/5, 5))
            if len(yticks) < 4:
                ax.set_yticks(np.arange(0, final_max_e, find_step_of_base(final_max_e/5, 1)))
            else:
                ax.set_yticks(yticks)
            for p in dummy_plots:
                p.set_visible(False)

        # give some extra space for the plot...
        final_max_e_pos += 0.02
        _ax_formatting(ax_pos, legend_handles, final_max_e_pos)

        fig.savefig(os.path.join(output_dir, param_names_dir + '_absolute_errors_boxplots.eps'),
                    bbox_inches="tight", format="eps", dpi=1200)
    else:
        raise Exception("\033[91mStats should be a dict: " + stats + "\033[99m")

    # Restore plotting config.
    from evo.tools.settings import SETTINGS
    plt.style.use('seaborn')
    # configure matplotlib and seaborn according to package settings
    sns.set(style=SETTINGS.plot_seaborn_style,
            palette=SETTINGS.plot_seaborn_palette,
            font=SETTINGS.plot_fontfamily,
            font_scale=SETTINGS.plot_fontscale
           )

    rc_params = {
        "lines.linewidth": SETTINGS.plot_linewidth,
        "text.usetex": SETTINGS.plot_usetex,
        "font.family": SETTINGS.plot_fontfamily,
        "font.serif": ['Cardo'],
        "pgf.texsystem": SETTINGS.plot_texsystem
    }
    mpl.rcParams.update(rc_params)
