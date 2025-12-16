#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Synthetic Ocean AI']


# MIT License
#
# Copyright (c) 2025 Synthetic Ocean AI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


try:
    import sys
    import numpy

    import seaborn
    import logging

    from typing import List
    from pathlib import Path
    from typing import Optional

    import matplotlib.pyplot as plt
    from Tools.Plot.Plot import Plot
    import Tools.Plot.config as config

    from Tools.Plot.utils import create_directory

except ImportError as error:
    logging.error(error)
    sys.exit(-1)

DEFAULT_TITLE=''
DEFAULT_LEGEND_TITLE=''
DEFAULT_X_LABEL=''
DEFAULT_Y_LABEL=''
DEFAULT_TITLE_FONT_SIZE=16
DEFAULT_LABELS_FONT_SIZE=10
DEFAULT_TICKS_LABELS_FONT_SIZE=10
DEFAULT_LEGEND_FONT_SIZE=10
DEFAULT_WIDTH_BARS=0.2
DEFAULT_GAP=0.3
DEFAULT_GROUP = ["R-S", "R-R"]
DEFAULT_CLASSIFICATION_METRICS_DIRECTORY ='classification_metrics'
DEFAULT_AVAILABLE_PALETTES = ["pastel", "bright", "dark", "muted",
                              "colorblind", "deep", "Set1", "Set2",
                              "Set3", "tab10", "tab20"]
DEFAULT_METRICS = ["EuclideanDistance", "HellingerDistance",
                   "ManhattanDistance", "HammingDistance", "JaccardDistance"]

# Options
OPTIONS=['mean_std', 'mean', 'std']
DEFAULT_OPTION='mean_std'


class PlotDistanceMetrics(Plot):
    """
    A class for plotting distance metrics comparison across different datasets and groups.

    This class generates bar plots comparing various distance metrics across different
    datasets and groups, with options to display mean, standard deviation, or both.
    The plot can be customized with different colors, labels, and formatting options.

    Attributes:
        groups (list): List of group names to compare.
        metrics (list): List of metric names to plot.
        classifiers (list): List of classifier names (unused in current implementation).
        color_map (dict): Mapping of groups to color palettes.
        width_bars (float): Width of individual bars.
        option (str): Display option ('mean', 'std', or 'mean_std').
        gap (float): Gap between different dataset clusters.
        title (str): Main plot title.
        legend_title (str): Title for the legend.
        x_label (str): Label for x-axis.
        y_label (str): Label for y-axis.
        title_font_size (int): Font size for title.
        labels_font_size (int): Font size for axis labels.
        ticks_labels_font_size (int): Font size for tick labels.
        legend_font_size (int): Font size for legend text.
        save_path (Path): Directory to save the output plot.

    Example:
        >>> input_files = ["results/dataset1.json", "results/dataset2.json"]
        >>> plotter = PlotDistanceMetrics(
        ...     input_files=input_files,
        ...     groups=["GroupA", "GroupB"],
        ...     metrics=["euclidean", "manhattan"],
        ...     title="Metric Comparison",
        ...     option="mean_std"
        ... )
        >>> # This will generate and save a PDF plot comparing the metrics
    """

    def __init__(self,
                 input_files: List[str],
                 groups=None,
                 metrics=None,
                 classifiers=config.CLASSIFIERS,
                 color_map=None,
                 title=DEFAULT_TITLE,
                 legend_title=DEFAULT_LEGEND_TITLE,
                 x_label=DEFAULT_X_LABEL,
                 y_label=DEFAULT_Y_LABEL,
                 title_font_size=DEFAULT_TITLE_FONT_SIZE,
                 labels_font_size=DEFAULT_LABELS_FONT_SIZE,
                 ticks_labels_font_size=DEFAULT_TICKS_LABELS_FONT_SIZE,
                 legend_font_size=DEFAULT_LEGEND_FONT_SIZE,
                 width_bars=DEFAULT_WIDTH_BARS,
                 gap=DEFAULT_GAP,
                 option: Optional[str] = DEFAULT_OPTION):
        """
        Initialize the PlotDistanceMetrics instance.

        Args:
            input_files: List of paths to JSON files containing metric data.
            groups: List of group names to include in comparison.
            metrics: List of metric names to plot.
            classifiers: List of classifier names (unused in current implementation).
            color_map: Custom color mapping dictionary. If None, default palettes will be used.
            title: Title for the plot.
            legend_title: Title for the legend.
            x_label: Label for x-axis.
            y_label: Label for y-axis.
            title_font_size: Font size for title.
            labels_font_size: Font size for axis labels.
            ticks_labels_font_size: Font size for tick labels.
            legend_font_size: Font size for legend text.
            width_bars: Width of individual bars.
            gap: Gap between different dataset clusters.
            option: Display option ('mean', 'std', or 'mean_std').

        Raises:
            ValueError: If number of groups exceeds available color palettes or invalid option.

        Example:
            >>> plotter = PlotDistanceMetrics(
            ...     input_files=["data1.json", "data2.json"],
            ...     groups=["Control", "Experimental"],
            ...     metrics=["accuracy", "precision"],
            ...     title="Performance metrics",
            ...     option="mean_std"
            ... )
        """
        super().__init__()

        if groups is None:
            groups = DEFAULT_GROUP

        if metrics is None:
            metrics = DEFAULT_METRICS

        self._validate_inputs(groups, option, color_map)

        self._initialize_attributes(groups=groups,
                                    metrics=metrics,
                                    classifiers=classifiers,
                                    color_map=color_map,
                                    width_bars=width_bars,
                                    option=option,
                                    gap=gap,
                                    title=title,
                                    legend_title=legend_title,
                                    x_label=x_label,
                                    y_label=y_label,
                                    title_font_size=title_font_size,
                                    labels_font_size=labels_font_size,
                                    ticks_labels_font_size=ticks_labels_font_size,
                                    legend_font_size=legend_font_size,
                                    input_files=input_files)

        data = self._read_data(input_files=input_files)
        self._plot_distance_metrics(data=data, datasets=input_files)

    @staticmethod
    def _validate_inputs(groups, option, color_map):
        """
        Validate input parameters.

        Args:
            groups: List of group names.
            option: Display option.
            color_map: Custom color mapping or None.

        Raises:
            ValueError: If invalid parameters are provided.
        """
        if color_map is None and len(groups) > len(DEFAULT_AVAILABLE_PALETTES):
            raise ValueError(
                f"Not enough color palettes for the number of groups ({len(groups)} groups, "
                f"but only {len(DEFAULT_AVAILABLE_PALETTES)} palettes available). "
                f"Consider reducing the number of groups or defining a custom color_map."
            )

        if option not in OPTIONS:
            raise ValueError(f"Invalid option: {option}. Must be one of {OPTIONS}.")

    def _initialize_attributes(self, **kwargs):
        """
        Initialize class attributes from keyword arguments.

        Args:
            **kwargs: Dictionary containing all initialization parameters.
        """
        self.groups = kwargs['groups']
        self.metrics = kwargs['metrics']
        self.classifiers = kwargs['classifiers']

        self.color_map = kwargs['color_map'] or self._create_default_color_map()
        self.width_bars = kwargs['width_bars']
        self.option = kwargs['option']
        self.gap = kwargs['gap']

        self.title = kwargs['title']
        self.legend_title = kwargs['legend_title']
        self.x_label = kwargs['x_label']
        self.y_label = kwargs['y_label']

        self.title_font_size = kwargs['title_font_size']
        self.labels_font_size = kwargs['labels_font_size']
        self.ticks_labels_font_size = kwargs['ticks_labels_font_size']
        self.legend_font_size = kwargs['legend_font_size']

        self.save_path = Path(kwargs['input_files'][0]).parent

    def _create_default_color_map(self):
        """
        Create default color map if none provided.

        Returns:
            dict: Mapping of group names to color palettes.

        Example:
            >>> self._create_default_color_map()
            {'Group1': [(0.1, 0.2, 0.3), ...], 'Group2': [(0.4, 0.5, 0.6), ...]}
        """

        return {group: seaborn.color_palette(DEFAULT_AVAILABLE_PALETTES[i], n_colors=len(self.metrics))
                for i, group in enumerate(self.groups)}

    def _plot_distance_metrics(self, data, datasets):
        """
        Main method to plot distance metrics.

        Args:
            data: Dictionary containing metric data for all datasets.
            datasets: List of dataset names/paths being compared.

        Note:
            This method coordinates the entire plotting process by calling helper methods.
        """
        fig, ax = self._setup_plot()
        colors_metrics = self._create_colors_metrics_mapping()
        x = self._calculate_x_positions(datasets)

        self._plot_all_bars(ax, data, datasets, x, colors_metrics)
        self._configure_plot_appearance(ax, x)
        self._save_plot()

    @staticmethod
    def _setup_plot():
        """
        Initialize matplotlib figure and axes.

        Returns:
            tuple: (fig, ax) matplotlib figure and axes objects.
        """
        return plt.subplots(figsize=(12, 6))

    def _create_colors_metrics_mapping(self):
        """
        Create mapping between metric-group combinations and colors.

        Returns:
            dict: Mapping of "metric group" strings to color values.

        Example:
            >>> self._create_colors_metrics_mapping()
            {'euclidean Group1': (0.1, 0.2, 0.3), ...}
        """
        colors_metrics = {}

        for i, metric in enumerate(self.metrics):

            for group in self.groups:
                colors_metrics[f"{metric} {group}"] = self.color_map[group][i]

        return colors_metrics

    def _calculate_x_positions(self, datasets):
        """
        Calculate x positions for all bars.

        Args:
            datasets: List of datasets being compared.

        Returns:
            ndarray: Array of x positions for bar clusters.
        """
        return numpy.arange(len(datasets)) * (len(self.metrics) * len(self.groups) * self.width_bars + self.gap)

    def _plot_all_bars(self, ax, data, datasets, x, colors_metrics):
        """
        Plot all bars for each metric-group combination.

        Args:
            ax: Matplotlib axes object.
            data: Dictionary containing metric data.
            datasets: List of dataset names/paths.
            x: Array of x positions for bar clusters.
            colors_metrics: Color mapping dictionary.
        """
        for i, (metric, group) in enumerate([(m, k) for m in self.metrics for k in self.groups]):
            mean, std = self._get_mean_std_values(data, datasets, group, metric)
            bars = self._plot_single_bar(ax, x, i, mean, std, colors_metrics, metric, group)
            self._add_bar_labels(ax, bars, std)

    @staticmethod
    def _get_mean_std_values(data, datasets, group, metric):
        """
        Extract mean and std values from data.

        Args:
            data: Dictionary containing metric data.
            datasets: List of dataset names/paths.
            group: Current group name.
            metric: Current metric name.

        Returns:
            tuple: (mean_values, std_values) for the specified group and metric.
        """
        mean = [data[dataset]["DistanceMetrics"][group][metric]['mean'] for dataset in datasets]
        std = [data[dataset]["DistanceMetrics"][group][metric]['std'] for dataset in datasets]
        return mean, std

    def _plot_single_bar(self, ax, x, i, mean, std, colors_metrics, metric, group):
        """
        Plot a single bar group.

        Args:
            ax: Matplotlib axes object.
            x: Array of x positions.
            i: Index of current metric-group combination.
            mean: List of mean values.
            std: List of std values.
            colors_metrics: Color mapping dictionary.
            metric: Current metric name.
            group: Current group name.

        Returns:
            BarContainer: Matplotlib bar container object.
        """
        error_kw = {'capsize': 2, 'ecolor': 'black'} if (self.option == 'std' or self.option == 'mean_std') else {}

        return ax.bar(x + i * self.width_bars, mean, self.width_bars,
                      yerr=std if (self.option == 'std' or self.option == 'mean_std') else None,
                      label=f"{metric} {group}", color=colors_metrics[f"{metric} {group}"],
                      error_kw=error_kw)

    def _add_bar_labels(self, ax, bars, std):
        """
        Add labels to bars based on selected option.

        Args:
            ax: Matplotlib axes object.
            bars: Bar container object.
            std: List of std values.
        """
        if self.option == 'mean_std':
            ax = self._draw_mean(ax, bars)
            ax = self._draw_std(ax, bars, std)

        elif self.option == 'mean':
            ax = self._draw_mean(ax, bars)

        elif self.option == 'std':
            ax = self._draw_std(ax, bars, std)

    def _configure_plot_appearance(self, ax, x):
        """
        Configure plot title, labels, ticks and legend.

        Args:
            ax: Matplotlib axes object.
            x: Array of x positions.
        """
        ax.set_title(f'distance {self.title} ', fontsize=self.title_font_size)
        ax.set_xlabel(self.x_label, fontsize=self.labels_font_size)
        ax.set_ylabel(self.y_label, fontsize=self.labels_font_size)

        ax.set_xticks(x + self.width_bars * len(self.metrics) * len(self.groups) / 2)
        ax.set_xticklabels("", fontsize=self.ticks_labels_font_size)

        ax.legend(title=self.legend_title, loc='upper left', bbox_to_anchor=(1, 1), fontsize=self.legend_font_size)

        plt.tight_layout()

    def _save_plot(self):
        """
        Save the plot to file in PDF format.

        Note:
            The plot will be saved in the same directory as the first input file,
            with a filename based on the title and option.
        """
        create_directory(self.save_path)
        save_figure = f"{self.save_path}/distance_{self.option} - {self.title}.pdf"
        plt.savefig(save_figure, format="pdf")
        print(f"New figure created: {save_figure}")
        plt.close()

    @staticmethod
    def _draw_std(ax, bars, std):
        """
        Add the std values on the bars.

        Args:
            ax: Matplotlib axes object.
            bars: Bar container object.
            std: List of std values.

        Returns:
            Matplotlib axes object.

        Example:
            >>> ax = self._draw_std(ax, bars, [0.1, 0.2])
        """
        ax.bar_label(bars, labels=[f'{e:.2f}' for e in std],
                     padding=0, color='black', fontsize='medium', fontweight='600')
        return ax

    def _draw_mean(self, ax, bars):
        """
        Add the mean values on the bars.

        Args:
            ax: Matplotlib axes object.
            bars: Bar container object.

        Returns:
            Matplotlib axes object.

        Example:
            >>> ax = self._draw_mean(ax, bars)
        """
        for bar in bars:
            axis_y = bar.get_height()
            ax.text(bar.get_x() + (bar.get_width() / 2.0),
                    axis_y - 0.025 if self.option == 'mean' else 0.02,
                    f'{axis_y:.2f}', ha='center', va='center',
                    color='black', fontsize='medium', fontweight='600')

        return ax

    def _get_values(self, data):
        """
        Extract necessary values from the input data structure.

        Args:
            data: Raw input data dictionary.

        Returns:
            dict: Processed dictionary containing only required metrics.

        Example:
            >>> values = self._get_values(raw_data)
            >>> values.keys()
            dict_keys(['DistanceMetrics'])
        """
        values = {"DistanceMetrics": {}}

        for group in self.groups:

            if group in data["DistanceMetrics"]:

                values["DistanceMetrics"][group] = {}
                fold_metrics = data["DistanceMetrics"][group].get('Summary', {})

                if fold_metrics:

                    for metric in self.metrics:

                        if metric in fold_metrics:
                            values["DistanceMetrics"][group][metric] = {'mean': fold_metrics[metric].get('mean'),
                                                                        'std': fold_metrics[metric].get("std")}

        return values
