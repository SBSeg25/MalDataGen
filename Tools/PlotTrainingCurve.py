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

    import os
    import sys

    import math
    import logging

    import matplotlib.pyplot as plt
    from Tools.Plot import Plot
    import Tools.config as config

    from Tools.utils import create_directory
    from matplotlib.backends.backend_pdf import PdfPages

except ImportError as error:
    logging.error(error)
    sys.exit(-1)

DEFAULT_CONFUSION_MATRIX_DIRECTORY ='confusion_matrices'

DEFAULT_TITLE=''
DEFAULT_X_LABEL = 'Predicted Label'
DEFAULT_Y_LABEL = 'True Label'
DEFAULT_TICKS_LABELS = ['True', 'False']
DEFAULT_SUB_TITLE_FONT_SIZE=16
DEFAULT_TITLE_FONT_SIZE=12
DEFAULT_LABELS_FONT_SIZE=10
DEFAULT_TICKS_LABELS_FONT_SIZE=10
DEFAULT_TICKS_SCALE_FONT_SIZE=8

class PlotTrainingCurve(Plot):
    """
    A class for visualizing machine learning training curves across multiple folds.

    This class generates PDF reports containing training curves for each fold,
    showing the evolution of metrics across epochs. The visualization includes
    multiple subplots (one per fold) with configurable styling options.

    Attributes:
        groups (list): List of data groups to analyze
        metrics (list): metrics to plot (e.g., ['accuracy', 'loss'])
        classifiers (list): classifiers used in the experiment
        x_label (str): Label for x-axis
        y_label (str): Label for y-axis
        title (str): Main title for the plot
        save_path (str): Directory where plots will be saved

    Example Usage:
        >>> # Basic usage with single input file
        >>> plotter = PlotTrainingCurve(
        ...     input_file="path/to/training_data.json",
        ...     title="Model Training Progress"
        ... )
        >>>
        >>> # Advanced usage with custom configuration
        >>> plotter = PlotTrainingCurve(
        ...     input_file=["file1.json", "file2.json"],
        ...     metrics=['accuracy', 'precision', 'recall'],
        ...     title="Custom Training Analysis",
        ...     title_font_size=14,
        ...     labels_font_size=12
        ... )
    """
    def __init__(self,
                 input_file: str,
                 groups=config.GROUPS,
                 metrics=config.METRICS,
                 classifiers=config.CLASSIFIERS,
                 title=DEFAULT_TITLE,
                 x_label=DEFAULT_X_LABEL,
                 y_label=DEFAULT_Y_LABEL,
                 ticks_labels=DEFAULT_TICKS_LABELS,
                 sub_title_font_size=DEFAULT_SUB_TITLE_FONT_SIZE,
                 title_font_size=DEFAULT_TITLE_FONT_SIZE,
                 labels_font_size=DEFAULT_LABELS_FONT_SIZE,
                 ticks_labels_font_size=DEFAULT_TICKS_LABELS_FONT_SIZE,
                 ticks_scale_font_size=DEFAULT_TICKS_SCALE_FONT_SIZE):
        super().__init__()
        """
        Initialize the PlotTrainingCurve instance.

        Args:
            input_file (str or list): Path to training data JSON file(s)
            groups (list, optional): Data groups to include. Defaults to config.GROUPS
            metrics (list, optional): metrics to plot. Defaults to config.METRICS
            classifiers (list, optional): classifiers to include. Defaults to config.CLASSIFIERS
            title (str, optional): Plot title. Defaults to DEFAULT_TITLE
            x_label (str, optional): X-axis label. Defaults to DEFAULT_X_LABEL
            y_label (str, optional): Y-axis label. Defaults to DEFAULT_Y_LABEL
            ticks_labels (list, optional): Tick labels. Defaults to DEFAULT_TICKS_LABELS
            sub_title_font_size (int, optional): Subtitle font size. Defaults to DEFAULT_SUB_TITLE_FONT_SIZE
            title_font_size (int, optional): Title font size. Defaults to DEFAULT_TITLE_FONT_SIZE
            labels_font_size (int, optional): Axis labels font size. Defaults to DEFAULT_LABELS_FONT_SIZE
            ticks_labels_font_size (int, optional): Tick labels font size. Defaults to DEFAULT_TICKS_LABELS_FONT_SIZE
            ticks_scale_font_size (int, optional): Tick scale font size. Defaults to DEFAULT_TICKS_SCALE_FONT_SIZE
        """
        """
        Initialize all class attributes with provided values.

        Args:
            input_file (str or list): Input file path(s)
            groups (list): Data groups
            metrics (list): metrics to track
            classifiers (list): classifiers used
            title (str): Plot title
            x_label (str): X-axis label
            y_label (str): Y-axis label
            ticks_labels (list): Tick labels
            sub_title_font_size (int): Subtitle font size
            title_font_size (int): Title font size
            labels_font_size (int): Axis labels font size
            ticks_labels_font_size (int): Tick labels font size
            ticks_scale_font_size (int): Tick scale font size
        """
        self.groups = groups
        self.metrics = metrics
        self.classifiers = classifiers
        self.x_label = x_label
        self.y_label = y_label
        self.ticks_labels = ticks_labels
        self.title = title
        self.sub_title_font_size = sub_title_font_size
        self.title_font_size = title_font_size
        self.labels_font_size = labels_font_size
        self.ticks_labels_font_size = ticks_labels_font_size
        self.ticks_scale_font_size = ticks_scale_font_size
        self.save_path = os.path.dirname(input_file[0])

        data = self._read_data(input_files=input_file)
        self.plot_training_curves(data)

    def plot_training_curves(self, json_data):
        """
        Generate and save training curves visualization.

        Creates a multi-page PDF document with training curves for each fold,
        showing metric evolution across epochs.

        Args:
            json_data (dict): Training data in dictionary format
        """
        create_directory(self.save_path)
        save_path = self._get_save_path()

        num_plots = len(json_data)
        rows, cols = self._calculate_grid_dimensions(num_plots)

        with PdfPages(save_path) as pdf:
            fig, axes = self._create_figure_and_axes(rows, cols)
            self._plot_all_folds(fig, axes, json_data)
            self._save_figure(pdf, fig, save_path)

    def _get_save_path(self):
        """
        Generate the output file path for saving the plot.

        Returns:
            str: Full path to output PDF file
        """
        return f"{self.save_path}/training_curves - {self.title}.pdf"

    @staticmethod
    def _calculate_grid_dimensions(num_plots):
        """
        Calculate optimal grid dimensions for subplots.

        Args:
            num_plots (int): Number of plots to arrange

        Returns:
            tuple: (rows, cols) for subplot grid
        """
        cols = 2
        rows = math.ceil(num_plots / cols)
        return rows, cols

    @staticmethod
    def _create_figure_and_axes(rows, cols):
        """
        Create matplotlib figure and axes with appropriate sizing.

        Args:
            rows (int): Number of rows in subplot grid
            cols (int): Number of columns in subplot grid

        Returns:
            tuple: (figure, flattened axes array)
        """
        fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))
        return fig, axes.flatten()

    def _plot_all_folds(self, fig, axes, json_data):
        """
        Plot training data for all folds on provided axes.

        Args:
            fig (matplotlib.figure.Figure): Main figure object
            axes (numpy.ndarray): Array of axes objects
            json_data (dict): Training data dictionary
        """
        for idx, (f, ax) in enumerate(zip(json_data, axes)):
            f_data = json_data[f]
            metrics_data = self._process_epoch_data(f_data)
            self._plot_metrics(ax, metrics_data)
            self._configure_axis(ax, idx)

        fig.suptitle(f"{self.title}", fontsize=self.sub_title_font_size)
        plt.tight_layout()

    @staticmethod
    def _process_epoch_data(fold_data):
        """
        Process epoch-level training data for a single fold.

        Args:
            fold_data (dict): Training data for a single fold

        Returns:
            dict: Processed metrics data organized by metric
        """
        metrics_data = {}

        # Process each epoch's data
        for epoch_data in fold_data.get('epochs', []):
            epoch = epoch_data.get('epoch')
            metrics = epoch_data.get('metrics', {})

            # Organize data by metric
            for metric, value in metrics.items():
                if metric not in metrics_data:
                    metrics_data[metric] = {'epochs': [], 'values': []}

                metrics_data[metric]['epochs'].append(epoch)
                metrics_data[metric]['values'].append(value)

        return metrics_data

    @staticmethod
    def _plot_metrics(ax, metrics_data):
        """
        Plot metrics data on a single axis.

        Args:
            ax (matplotlib.axes.Axes): Axis to plot on
            metrics_data (dict): metrics data to plot
        """
        for metric, data in metrics_data.items():
            ax.plot(data['epochs'], data['values'], label=metric, linestyle='-')

    @staticmethod
    def _configure_axis(ax, fold_idx):
        """
        Save figure to PDF and clean up resources.

        Args:
            pdf (PdfPages): PDF pages object
            fig (matplotlib.figure.Figure): Figure to save
            save_path (str): Path where PDF is saved
        """
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Metric Value')
        ax.set_title(f'Fold {fold_idx + 1}')
        ax.legend()
        ax.grid(True)

    @staticmethod
    def _save_figure(pdf, fig, save_path):
        """Save the figure and close it."""
        pdf.savefig(fig)
        plt.close(fig)
        print(f"New figure created: {save_path}")

    def _get_values(self, data):
        """Utility method to safely extract values from data structure."""
        return data

    @property
    def groups(self):
        """Get the list of data groups being analyzed.

        Returns:
            list: Current groups configuration
        """
        return self._groups

    @groups.setter
    def groups(self, value):
        """Set the list of data groups to analyze.

        Args:
            value (list): New groups configuration

        Raises:
            TypeError: If value is not a list
        """
        if not isinstance(value, list):
            raise TypeError("groups must be a list")
        self._groups = value

    @property
    def metrics(self):
        """Get the metrics being tracked.

        Returns:
            list: Current metrics configuration
        """
        return self._metrics

    @metrics.setter
    def metrics(self, value):
        """Set the metrics to track.

        Args:
            value (list): New metrics configuration

        Raises:
            TypeError: If value is not a list
        """
        if not isinstance(value, list):
            raise TypeError("metrics must be a list")
        self._metrics = value

    @property
    def classifiers(self):
        """Get the classifiers being used.

        Returns:
            list: Current classifiers configuration
        """
        return self._classifiers

    @classifiers.setter
    def classifiers(self, value):
        """Set the classifiers to use.

        Args:
            value (list): New classifiers configuration

        Raises:
            TypeError: If value is not a list
        """
        if not isinstance(value, list):
            raise TypeError("classifiers must be a list")
        self._classifiers = value

    @property
    def x_label(self):
        """Get the x-axis label.

        Returns:
            str: Current x-axis label
        """
        return self._x_label

    @x_label.setter
    def x_label(self, value):
        """Set the x-axis label.

        Args:
            value (str): New x-axis label

        Raises:
            TypeError: If value is not a string
        """
        if not isinstance(value, str):
            raise TypeError("x_label must be a string")
        self._x_label = value

    @property
    def y_label(self):
        """Get the y-axis label.

        Returns:
            str: Current y-axis label
        """
        return self._y_label

    @y_label.setter
    def y_label(self, value):
        """Set the y-axis label.

        Args:
            value (str): New y-axis label

        Raises:
            TypeError: If value is not a string
        """
        if not isinstance(value, str):
            raise TypeError("y_label must be a string")
        self._y_label = value

    @property
    def ticks_labels(self):
        """Get the tick labels configuration.

        Returns:
            list: Current ticks labels
        """
        return self._ticks_labels

    @ticks_labels.setter
    def ticks_labels(self, value):
        """Set the tick labels.

        Args:
            value (list): New ticks labels configuration

        Raises:
            TypeError: If value is not a list
        """
        if not isinstance(value, list):
            raise TypeError("ticks_labels must be a list")
        self._ticks_labels = value

    @property
    def title(self):
        """Get the plot title.

        Returns:
            str: Current title
        """
        return self._title

    @title.setter
    def title(self, value):
        """Set the plot title.

        Args:
            value (str): New title

        Raises:
            TypeError: If value is not a string
        """
        if not isinstance(value, str):
            raise TypeError("title must be a string")
        self._title = value

    @property
    def sub_title_font_size(self):
        """Get the subtitle font size.

        Returns:
            int: Current subtitle font size
        """
        return self._sub_title_font_size

    @sub_title_font_size.setter
    def sub_title_font_size(self, value):
        """Set the subtitle font size.

        Args:
            value (int): New font size in points

        Raises:
            TypeError: If value is not an integer
            ValueError: If value is not positive
        """
        if not isinstance(value, int):
            raise TypeError("sub_title_font_size must be an integer")
        if value <= 0:
            raise ValueError("sub_title_font_size must be positive")
        self._sub_title_font_size = value

    @property
    def title_font_size(self):
        """Get the title font size.

        Returns:
            int: Current title font size
        """
        return self._title_font_size

    @title_font_size.setter
    def title_font_size(self, value):
        """Set the title font size.

        Args:
            value (int): New font size in points

        Raises:
            TypeError: If value is not an integer
            ValueError: If value is not positive
        """
        if not isinstance(value, int):
            raise TypeError("title_font_size must be an integer")
        if value <= 0:
            raise ValueError("title_font_size must be positive")
        self._title_font_size = value

    @property
    def labels_font_size(self):
        """Get the axis labels font size.

        Returns:
            int: Current labels font size
        """
        return self._labels_font_size

    @labels_font_size.setter
    def labels_font_size(self, value):
        """Set the axis labels font size.

        Args:
            value (int): New font size in points

        Raises:
            TypeError: If value is not an integer
            ValueError: If value is not positive
        """
        if not isinstance(value, int):
            raise TypeError("labels_font_size must be an integer")
        if value <= 0:
            raise ValueError("labels_font_size must be positive")
        self._labels_font_size = value

    @property
    def ticks_labels_font_size(self):
        """Get the tick labels font size.

        Returns:
            int: Current tick labels font size
        """
        return self._ticks_labels_font_size

    @ticks_labels_font_size.setter
    def ticks_labels_font_size(self, value):
        """Set the tick labels font size.

        Args:
            value (int): New font size in points

        Raises:
            TypeError: If value is not an integer
            ValueError: If value is not positive
        """
        if not isinstance(value, int):
            raise TypeError("ticks_labels_font_size must be an integer")
        if value <= 0:
            raise ValueError("ticks_labels_font_size must be positive")
        self._ticks_labels_font_size = value

    @property
    def ticks_scale_font_size(self):
        """Get the tick scale font size.

        Returns:
            int: Current tick scale font size
        """
        return self._ticks_scale_font_size

    @ticks_scale_font_size.setter
    def ticks_scale_font_size(self, value):
        """Set the tick scale font size.

        Args:
            value (int): New font size in points

        Raises:
            TypeError: If value is not an integer
            ValueError: If value is not positive
        """
        if not isinstance(value, int):
            raise TypeError("ticks_scale_font_size must be an integer")
        if value <= 0:
            raise ValueError("ticks_scale_font_size must be positive")
        self._ticks_scale_font_size = value

    @property
    def save_path(self):
        """Get the current save path for output files.

        Returns:
            str: Current save directory path
        """
        return self._save_path

    @save_path.setter
    def save_path(self, value):
        """Set the save path for output files.

        Args:
            value (str): New directory path

        Raises:
            TypeError: If value is not a string
            ValueError: If path does not exist
        """
        if not isinstance(value, str):
            raise TypeError("save_path must be a string")
        if not os.path.exists(value):
            raise ValueError(f"Path does not exist: {value}")
        self._save_path = value