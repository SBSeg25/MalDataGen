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

    from typing import Any
    from typing import Dict
    from typing import List
    from pathlib import Path
    from typing import Optional

    import matplotlib.pyplot as plt
    from Tools.Plot import Plot
    import Tools.config as config

    from Tools.utils import create_directory

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
DEFAULT_OPTIONS=['mean_std', 'mean', 'std']
DEFAULT_OPTION='mean_std'
DEFAULT_CLASSIFICATION_METRICS_DIRECTORY ='classsification_metrics'
DEFAULT_AVAILABLE_PALETTES = ["pastel", "bright", "dark", "muted",
                              "colorblind", "deep", "Set1", "Set2",
                              "Set3", "tab10", "tab20"]

class PlotClassificationMetrics(Plot):
    """
    Visualizes classification metrics across multiple datasets, groups, and classifiers.

    The plot shows performance metrics as grouped bars, with options to display
    mean values, standard deviations, or both. Each classifier gets its own plot
    for clear comparison across different experimental conditions.

    Args:
        input_files (List[str]): Paths to JSON files containing evaluation metrics
        groups (List[str], optional): List of group names to include. Defaults to config.GROUPS.
        metrics (List[str], optional): metrics to visualize. Defaults to config.METRICS.
        classifiers (List[str], optional): classifiers to include. Defaults to config.CLASSIFIERS.
        color_map (Dict[str, List], optional): Color palette for each group. If None, uses default palettes.
        title (str, optional): Main plot title. Defaults to DEFAULT_TITLE.
        legend_title (str, optional): Title for the legend. Defaults to DEFAULT_LEGEND_TITLE.
        x_label (str, optional): Label for x-axis. Defaults to DEFAULT_X_LABEL.
        y_label (str, optional): Label for y-axis. Defaults to DEFAULT_Y_LABEL.
        title_font_size (int, optional): Font size for title. Defaults to DEFAULT_TITLE_FONT_SIZE.
        labels_font_size (int, optional): Font size for axis labels. Defaults to DEFAULT_LABELS_FONT_SIZE.
        ticks_labels_font_size (int, optional): Font size for tick labels. Defaults to DEFAULT_TICKS_LABELS_FONT_SIZE.
        legend_font_size (int, optional): Font size for legend text. Defaults to DEFAULT_LEGEND_FONT_SIZE.
        width_bars (float, optional): Width of individual bars. Defaults to DEFAULT_WIDTH_BARS.
        gap (float, optional): Gap between dataset groups. Defaults to DEFAULT_GAP.
        option (str, optional): Display option ('mean', 'std', or 'mean_std'). Defaults to DEFAULT_OPTION.

    Raises:
        ValueError: If there aren't enough color palettes for the number of groups
        ValueError: If an invalid option is provided
    """

    def __init__(self,
                 input_files: List[str],
                 groups=config.GROUPS,
                 metrics=config.METRICS,
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
                 option: Optional[str]=DEFAULT_OPTION, ):

        super().__init__()
        """
        Initialize the PlotClassificationMetrics instance and generate plots.

        Example:
            # With custom color mapping
            custom_colors = {
                "Group1": sns.color_palette("Blues", n_colors=3),
                "Group2": sns.color_palette("Greens", n_colors=3)
            }
            plotter = PlotClassificationMetrics(
                input_files=["exp1/results.json"],
                color_map=custom_colors,
                option="mean_std"
            )
        """
        if color_map is None:

            if len(groups) > len(DEFAULT_AVAILABLE_PALETTES):
                raise ValueError(f"Not enough color palettes for the number of groups ({len(groups)} groups, "
                                 f"but only {len(DEFAULT_AVAILABLE_PALETTES)} palettes available). "
                                 f"Consider reducing the number of groups or defining a custom color_map.")

            color_map = {group: seaborn.color_palette(DEFAULT_AVAILABLE_PALETTES[i], n_colors=len(metrics))
                         for i, group in enumerate(groups)}
        if option not in DEFAULT_OPTIONS:
            raise ValueError(f"Invalid option: {option}. Must be one of {DEFAULT_OPTIONS}.")

        self.groups = groups
        self.metrics = metrics
        self.classifiers = classifiers

        self.color_map = color_map
        self.width_bars = width_bars
        self.option = option
        self.gap = gap

        self.title = title
        self.legend_title = legend_title
        self.x_label = x_label
        self.y_label = y_label

        self.title_font_size = title_font_size
        self.labels_font_size = labels_font_size
        self.ticks_labels_font_size = ticks_labels_font_size
        self.legend_font_size = legend_font_size

        self.save_path = Path(input_files[0]).parent
        data = self._read_data(input_files=input_files)

        self._plot_classification_metrics(data=data, datasets=input_files)


    def _plot_classification_metrics(self, data, datasets):
        """
        Generate the classification metrics plots for all classifiers.

        Args:
            data (Dict): Processed metrics data
            datasets (List[str]): List of dataset names/paths

        Example:
            This method is called automatically during initialization.
            For manual plotting with custom data:

            processed_data = {
                "dataset1": {
                    "GroupA": {
                        "RandomForest": {
                            "accuracy": {"mean": 0.85, "std": 0.03},
                            "f1_score": {"mean": 0.82, "std": 0.04}
                        }
                    }
                }
            }
            self._plot_classification_metrics(data=processed_data, datasets=["dataset1"])
        """
        colors_metrics = self._generate_colors_by_metric_group()
        classifiers = self._extract_present_classifiers(data)

        for clf in classifiers:
            fig, ax = self._create_figure()
            x = self._calculate_x_positions(datasets)

            self._plot_bars_for_classifier(ax, clf, data, datasets, colors_metrics, x)
            self._format_axes(ax, clf, x)

            self._save_figure(fig, clf)

    def _generate_colors_by_metric_group(self):
        """
        Generate color mapping for each metric-group combination.

        Returns:
            Dict[str, Tuple]: Mapping of "metric group" strings to RGB color values

        Example:
            colors = self._generate_colors_by_metric_group()
            # Returns {"accuracy GroupA": (0.1, 0.2, 0.5), ...}
        """
        colors_metrics = {}

        for i, metric in enumerate(self.metrics):

            for group in self.groups:
                colors_metrics[f"{metric} {group}"] = self.color_map[group][i]

        return colors_metrics

    @staticmethod
    def _extract_present_classifiers(data):
        """
        Extract unique classifier names present in the data.

        Args:
            data (Dict): Processed metrics data

        Returns:
            Set[str]: Unique classifier names

        Example:
            data = {
                "dataset1": {
                    "GroupA": {"SVM": {...}, "RF": {...}},
                    "GroupB": {"SVM": {...}}
                }
            }
            classifiers = self._extract_present_classifiers(data)
            # Returns {"SVM", "RF"}
        """
        return {clf for values in data.values() for values in values.values() for clf in values.keys()}

    @staticmethod
    def _create_figure():
        """
        Create a new matplotlib figure and axes.

        Returns:
            Tuple[Figure, Axes]: Figure and axes objects

        Example:
            fig, ax = self._create_figure()
            # Returns a new 12x6 inch figure with empty axes
        """
        return plt.subplots(figsize=(12, 6))

    def _calculate_x_positions(self, datasets):
        """
        Calculate x-axis positions for dataset groups.

        Args:
            datasets (List[str]): List of dataset names

        Returns:
            ndarray: Array of x positions for each dataset group

        Example:
            x_positions = self._calculate_x_positions(["ds1", "ds2"])
            # Returns array([0., 1.5]) (depending on width_bars and gap)
        """
        return numpy.arange(len(datasets)) * (len(self.metrics) * len(self.groups) * self.width_bars + self.gap)

    def _plot_bars_for_classifier(self,
                                  ax: plt.Axes,
                                  clf: str,
                                  data: Dict,
                                  datasets: List[str],
                                  colors_metrics: Dict,
                                  x: numpy.ndarray) -> None:
        """
        Plot all bars for a single classifier across metrics and groups.

        Parameters:
            ax : Matplotlib axes object
            clf : Classifier name being plotted
            data : metrics data structure
            datasets : Dataset names
            colors_metrics : Color mapping dictionary
            x : Base x-positions for dataset groups
        """
        # Iterate through all metric-group combinations
        for i, (metric, group) in enumerate([(m, k) for m in self.metrics for k in self.groups]):
            # Extract mean values (handle missing data with 0)
            mean = [
                data[dataset][group][clf][metric]['mean']
                if clf in data[dataset][group] else 0
                for dataset in datasets
            ]

            # Extract standard deviations
            std = [
                data[dataset][group][clf][metric]['std']
                if clf in data[dataset][group] else 0
                for dataset in datasets
            ]

            # Plot bars with optional error bars
            bars = ax.bar(
                x + i * self.width_bars,  # Position along x-axis
                mean,  # Bar heights
                self.width_bars,  # Bar width
                yerr=std if self.option in ['std', 'mean_std'] else None,  # Error bars
                label=f"{metric} {group}",  # Legend label
                color=colors_metrics[f"{metric} {group}"],  # Bar color
                error_kw={
                    'capsize': 2,  # Error bar cap size
                    'ecolor': 'black'  # Error bar color
                } if self.option in ['std', 'mean_std'] else {}
            )

            # Add value annotations based on visualization option
            if self.option in ['mean', 'mean_std']:
                self._draw_mean(ax, bars)

            if self.option in ['std', 'mean_std']:
                self._draw_std(ax, bars, std)

    def _format_axes(self, ax: plt.Axes, clf: str, x: numpy.ndarray) -> None:
        """
        Apply consistent formatting to plot axes.

        Parameters:
            ax : Matplotlib axes object
            clf : Classifier name (for title)
            x : X-axis positions of dataset groups
        """
        # Set title with classifier name
        ax.set_title(f'{self.title} - {clf}', fontsize=self.title_font_size)

        # Axis labels
        ax.set_xlabel(self.x_label, fontsize=self.labels_font_size)
        ax.set_ylabel(self.y_label, fontsize=self.labels_font_size)

        # Position x-ticks at center of each dataset group
        ax.set_xticks(x + self.width_bars * len(self.metrics) * len(self.groups) / 2)

        # Use empty labels (datasets typically identified by other means)
        ax.set_xticklabels([""] * len(x), fontsize=self.ticks_labels_font_size)

        # Standardize y-axis for metric comparison
        ax.set_ylim(0, 1.01)  # Slightly above 1 for annotation space

        # Configure legend (placed outside plot area)
        ax.legend(
            title=self.legend_title,
            loc='upper left',
            bbox_to_anchor=(1, 1),  # Position outside plot
            fontsize=self.legend_font_size
        )

        # Adjust layout to prevent clipping
        plt.tight_layout()

    def _save_figure(self, fig: plt.Figure, clf: str) -> None:
        """
        Save plot to PDF file with standardized naming.

        Parameters:
            fig : Matplotlib figure object
            clf : Classifier name (for filename)
        """
        # Ensure output directory exists
        create_directory(self.save_path)

        # Generate filename with classifier and plot title
        filename = f"{self.save_path}/mean - {clf} - {self.title}.pdf"

        # Save as vector PDF for publication quality
        fig.savefig(filename, format="pdf")
        print(f"Saved new figure: {filename}")

        # Close figure to free memory
        plt.close(fig)

    @staticmethod
    def _draw_std(ax, bars, std):
        """
        Add standard deviation labels to bars.

        Args:
            ax (Axes): Matplotlib axes object
            bars (BarContainer): Bar objects
            std (List[float]): Standard deviation values

        Returns:
            Axes: Modified axes object

        Example:
            # Internal use - called during bar plotting
            ax = self._draw_std(ax, bars, [0.1, 0.2])
        """
        ax.bar_label(bars, labels=[f'{e:.2f}' for e in std], padding=0, color='black', fontsize='medium', fontweight='600')
        return ax


    def _draw_mean(self, ax, bars):
        """
        Add mean value labels to bars.

        Args:
            ax (Axes): Matplotlib axes object
            bars (BarContainer): Bar objects

        Returns:
            Axes: Modified axes object

        Example:
            # Internal use - called during bar plotting
            ax = self._draw_mean(ax, bars)
        """
        for bar in bars:
            y_axis_evaluation = bar.get_height()
            ax.text(
                bar.get_x() + (bar.get_width() / 2.0),
                y_axis_evaluation - 0.025 if self.option == 'mean' else 0.02,
                f'{y_axis_evaluation:.2f}',
                ha='center',
                va='center',
                color='black',
                fontsize='medium',
                fontweight='600'
            )
        return ax

    def _get_values(self, data: Dict) -> Dict:
        """
        Extract and structure metric values from raw evaluation data.

        Parameters:
            data : Raw input data from evaluation files

        Returns:
            Nested dictionary with structured metrics:
            {
                group: {
                    classifier: {
                        metric: {
                            'mean': value,
                            'std': value
                        }
                    }
                }
            }
        """
        values = {}

        # Process each configured group
        for group in self.groups:
            if group in data:
                values[group] = {}

                # Process each configured classifier
                for clf in self.classifiers:
                    if clf in data[group]:
                        values[group][clf] = {}

                        # Extract summary statistics
                        fold_metrics = data[group][clf].get('Summary', {})

                        if fold_metrics:
                            # Extract each configured metric
                            for metric in self.metrics:
                                if metric in fold_metrics:
                                    values[group][clf][metric] = {
                                        'mean': fold_metrics[metric].get('mean'),
                                        'std': fold_metrics[metric].get("std")
                                    }
        return values


    # Property getters and setters with validation
    @property
    def groups(self) -> List[str]:
        """Get the list of experimental groups being compared."""
        return self._groups

    @groups.setter
    def groups(self, value: List[str]) -> None:
        """Set the experimental groups with validation."""
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            raise ValueError("Groups must be a list of strings")
        self._groups = value

    @property
    def metrics(self) -> List[str]:
        """Get the list of performance metrics being visualized."""
        return self._metrics

    @metrics.setter
    def metrics(self, value: List[str]) -> None:
        """Set the performance metrics with validation."""
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            raise ValueError("metrics must be a list of strings")
        self._metrics = value

    @property
    def classifiers(self) -> List[str]:
        """Get the list of classifiers being compared."""
        return self._classifiers

    @classifiers.setter
    def classifiers(self, value: List[str]) -> None:
        """Set the classifiers with validation."""
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            raise ValueError("classifiers must be a list of strings")
        self._classifiers = value

    @property
    def color_map(self) -> Dict[str, Any]:
        """Get the color mapping dictionary for groups/metrics."""
        return self._color_map

    @color_map.setter
    def color_map(self, value: Optional[Dict[str, Any]]) -> None:
        """Set or generate the color mapping with validation."""
        if value is None:
            if len(self.groups) > len(DEFAULT_AVAILABLE_PALETTES):
                raise ValueError(
                    f"Not enough color palettes for {len(self.groups)} groups. "
                    f"Only {len(DEFAULT_AVAILABLE_PALETTES)} available. "
                    "Reduce groups or provide custom color_map."
                )
            value = {
                group: seaborn.color_palette(DEFAULT_AVAILABLE_PALETTES[i], n_colors=len(self.metrics))
                for i, group in enumerate(self.groups)
            }
        elif not isinstance(value, dict):
            raise ValueError("color_map must be a dictionary")
        self._color_map = value

    @property
    def width_bars(self) -> float:
        """Get the width of individual bars in plot units."""
        return self._width_bars

    @width_bars.setter
    def width_bars(self, value: float) -> None:
        """Set bar width with validation."""
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("width_bars must be a positive number")
        self._width_bars = float(value)

    @property
    def option(self) -> str:
        """Get the current visualization option ('mean', 'std', or 'mean_std')."""
        return self._option

    @option.setter
    def option(self, value: str) -> None:
        """Set visualization option with validation."""
        if value not in DEFAULT_OPTIONS:
            raise ValueError(f"Option must be one of {DEFAULT_OPTIONS}")
        self._option = value

    @property
    def gap(self) -> float:
        """Get the gap size between dataset groups in plot units."""
        return self._gap

    @gap.setter
    def gap(self, value: float) -> None:
        """Set gap size with validation."""
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError("gap must be a non-negative number")
        self._gap = float(value)

    @property
    def title(self) -> str:
        """Get the main plot title."""
        return self._title

    @title.setter
    def title(self, value: str) -> None:
        """Set plot title with validation."""
        if not isinstance(value, str):
            raise ValueError("title must be a string")
        self._title = value

    @property
    def legend_title(self) -> str:
        """Get the legend title."""
        return self._legend_title

    @legend_title.setter
    def legend_title(self, value: str) -> None:
        """Set legend title with validation."""
        if not isinstance(value, str):
            raise ValueError("legend_title must be a string")
        self._legend_title = value

    @property
    def x_label(self) -> str:
        """Get the x-axis label."""
        return self._x_label

    @x_label.setter
    def x_label(self, value: str) -> None:
        """Set x-axis label with validation."""
        if not isinstance(value, str):
            raise ValueError("x_label must be a string")
        self._x_label = value

    @property
    def y_label(self) -> str:
        """Get the y-axis label."""
        return self._y_label

    @y_label.setter
    def y_label(self, value: str) -> None:
        """Set y-axis label with validation."""
        if not isinstance(value, str):
            raise ValueError("y_label must be a string")
        self._y_label = value

    @property
    def title_font_size(self) -> int:
        """Get the title font size."""
        return self._title_font_size

    @title_font_size.setter
    def title_font_size(self, value: int) -> None:
        """Set title font size with validation."""
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("title_font_size must be a positive number")
        self._title_font_size = int(value)

    @property
    def labels_font_size(self) -> int:
        """Get the axis labels font size."""
        return self._labels_font_size

    @labels_font_size.setter
    def labels_font_size(self, value: int) -> None:
        """Set axis labels font size with validation."""
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("labels_font_size must be a positive number")
        self._labels_font_size = int(value)

    @property
    def ticks_labels_font_size(self) -> int:
        """Get the tick labels font size."""
        return self._ticks_labels_font_size

    @ticks_labels_font_size.setter
    def ticks_labels_font_size(self, value: int) -> None:
        """Set tick labels font size with validation."""
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("ticks_labels_font_size must be a positive number")
        self._ticks_labels_font_size = int(value)

    @property
    def legend_font_size(self) -> int:
        """Get the legend font size."""
        return self._legend_font_size

    @legend_font_size.setter
    def legend_font_size(self, value: int) -> None:
        """Set legend font size with validation."""
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("legend_font_size must be a positive number")
        self._legend_font_size = int(value)

    @property
    def save_path(self) -> Path:
        """Get the path where plots will be saved."""
        return self._save_path

    @save_path.setter
    def save_path(self, value: Path) -> None:
        """Set the save path with validation."""
        if not isinstance(value, Path):
            raise ValueError("save_path must be a Path object")
        self._save_path = value
