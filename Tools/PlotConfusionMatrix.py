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

    import numpy
    import seaborn
    import logging

    from typing import List
    import matplotlib.pyplot as plt
    from Tools.Plot import Plot
    import Tools.config as config

    from Tools.utils import create_directory

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


class PlotConfusionMatrix(Plot):
    """
    A class for generating and plotting confusion matrices from evaluation data.

    This class visualizes confusion matrices for multiple classifiers across different
    groups and folds, with customizable display options. It creates a grid of subplots
    for each classifier, showing the confusion matrices for all groups and folds.

    Attributes:
        groups (list):
            List of group names to include in the plots.
        metrics (list):
            List of metrics to consider (unused in current implementation).
        classifiers (list):
            List of classifier names to plot.
        x_label (str):
            Label for the x-axis of confusion matrices.
        y_label (str):
            Label for the y-axis of confusion matrices.
        ticks_labels (list):
            Labels for the ticks on confusion matrix axes.
        title (str):
            Main title for the plots.
        sub_title_font_size (int):
            Font size for sub-titles.
        title_font_size (int):
            Font size for main titles.
        labels_font_size (int):
            Font size for axis labels.
        ticks_labels_font_size (int):
            Font size for tick labels.
        ticks_scale_font_size (int):
            Font size for colorbar tick labels.
        save_path (str):
            Directory path where plots will be saved.

    # Example Usage

    # Example of how to use the PlotConfusionMatrix class

        from config import GROUPS, METRICS, CLASSIFIERS

        # Initialize and run the confusion matrix plotter
        plotter = PlotConfusionMatrix(
            input_file="path/to/evaluation_data.json",
            groups=GROUPS,
            metrics=METRICS,
            classifiers=CLASSIFIERS,
            x_label="Predicted Label",
            y_label="True Label",
            title="Model Performance Comparison",
            ticks_labels=["Positive", "Negative"],
            sub_title_font_size=12,
            title_font_size=14,
            labels_font_size=10,
            ticks_labels_font_size=9,
            ticks_scale_font_size=8
        )
    """

    def __init__(self,
                 input_file: str,
                 groups=config.GROUPS,
                 metrics=config.METRICS,
                 classifiers=config.CLASSIFIERS,
                 x_label=DEFAULT_X_LABEL,
                 y_label=DEFAULT_Y_LABEL,
                 title=DEFAULT_TITLE,
                 ticks_labels=DEFAULT_TICKS_LABELS,
                 sub_title_font_size=DEFAULT_SUB_TITLE_FONT_SIZE,
                 title_font_size=DEFAULT_TITLE_FONT_SIZE,
                 labels_font_size=DEFAULT_LABELS_FONT_SIZE,
                 ticks_labels_font_size=DEFAULT_TICKS_LABELS_FONT_SIZE,
                 ticks_scale_font_size=DEFAULT_TICKS_SCALE_FONT_SIZE):

        """
        Initialize the PlotConfusionMatrix instance.

        Args:
            input_file (str):
                Path to the input file containing evaluation data.
            groups (list, optional):
                List of group names to include. Defaults to config.GROUPS.
            metrics (list, optional):
                List of metrics to consider. Defaults to config.METRICS.
            classifiers (list, optional):
                List of classifier names. Defaults to config.CLASSIFIERS.
            x_label (str, optional):
                X-axis label. Defaults to DEFAULT_X_LABEL.
            y_label (str, optional):
                Y-axis label. Defaults to DEFAULT_Y_LABEL.
            title (str, optional):
                Main plot title. Defaults to DEFAULT_TITLE.
            ticks_labels (list, optional):
                Tick labels for confusion matrix. Defaults to DEFAULT_TICKS_LABELS.
            sub_title_font_size (int, optional):
                Subtitle font size. Defaults to DEFAULT_SUB_TITLE_FONT_SIZE.
            title_font_size (int, optional):
                Title font size. Defaults to DEFAULT_TITLE_FONT_SIZE.
            labels_font_size (int, optional):
                Axis labels font size. Defaults to DEFAULT_LABELS_FONT_SIZE.
            ticks_labels_font_size (int, optional):
                Tick labels font size. Defaults to DEFAULT_TICKS_LABELS_FONT_SIZE.
            ticks_scale_font_size (int, optional):
                Colorbar tick labels font size. Defaults to DEFAULT_TICKS_SCALE_FONT_SIZE.
        """

        super().__init__()

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

        self.save_path = os.path.dirname(input_file)

        data = self._read_data(input_files=[input_file])

        self._plot_confusion_matrix(data=data[input_file])

    def _plot_confusion_matrix(self, data):
        """
        Main method to plot confusion matrices for all classifiers and groups.

        Args:
            data (Dict): Nested dictionary containing evaluation data in the structure:
                        {
                            classifier_name: {
                                group_name: {
                                    fold_name: {
                                        'TruePositive': int,
                                        'TrueNegative': int,
                                        'FalsePositive': int,
                                        'FalseNegative': int
                                    }
                                }
                            }
                        }
        """
        max_number_folds = self._get_max_number_folds(data)
        colormaps = self._generate_colormaps(max_number_folds)

        for classifier_name, groups_data in data.items():
            logging.debug(f"{classifier_name}")
            max_values = self._calculate_max_values(groups_data)
            self._plot_classifier_matrices(classifier_name, groups_data, max_values, colormaps)

    @staticmethod
    def _get_max_number_folds(data):
        """
        Calculate the maximum number of folds across all classifiers and groups.

        Args:
            data (Dict): evaluation data dictionary.

        Returns:
            int: Maximum number of folds found in the data.
        """
        return max(len(folds) for classifier_data in data.values() for folds in classifier_data.values())

    @staticmethod
    def _generate_colormaps(num_colors):
        """
        Generate a list of colormaps for visualization.

        Args:
            num_colors (int): Number of distinct colormaps to generate.

        Returns:
            List: List of seaborn color palettes.
        """
        colors = seaborn.color_palette("bright", n_colors=num_colors)
        return [seaborn.light_palette(color, as_cmap=True) for color in colors]

    @staticmethod
    def _calculate_max_values(groups_data):
        """
        Calculate maximum values for normalization across folds.

        Args:
            groups_data (Dict): Dictionary containing group-wise fold data.

        Returns:
            Dict: Dictionary mapping fold names to their maximum values.
        """
        max_values = {}

        for group_name, fold_data in groups_data.items():

            logging.debug(f"\t{group_name}")

            for fold_name, values in fold_data.items():

                logging.debug(f"\t\t{fold_name}\t {values}")
                total = sum(values.values())
                max_values[fold_name] = max(max_values.get(fold_name, 0), total)

        return max_values

    def _plot_classifier_matrices(self, classifier_name, groups_data, max_values, colormaps):
        """
        Create subplots for all groups and folds of a single classifier.

        Args:
            classifier_name (str): Name of the classifier being plotted.
            groups_data (Dict): Dictionary containing group-wise fold data.
            max_values (Dict): Dictionary of maximum values for normalization.
            colormaps (List): List of colormaps to use for visualization.
        """
        rows, cols = self._determine_subplot_layout(groups_data)
        fig, axes = plt.subplots(rows, cols, figsize=(12, 6))

        for row_idx, (group_name, fold_data) in enumerate(groups_data.items()):
            self._plot_group_matrices(row_idx, group_name, fold_data, axes,
                                      max_values, colormaps, cols)

        self._finalize_classifier_plot(fig, classifier_name)

    @staticmethod
    def _determine_subplot_layout(groups_data):
        """
        Determine the subplot grid dimensions based on data structure.

        Args:
            groups_data (Dict): Dictionary containing group-wise fold data.

        Returns:
            Tuple[int, int]: Number of rows and columns needed for subplots.
        """
        rows = len(groups_data)
        cols = max(len(fold_data) for fold_data in groups_data.values())
        return rows, cols

    def _plot_group_matrices(self, row_idx, group_name, fold_data, axes,
                             max_values, colormaps, total_cols):
        """
        Plot all matrices for a single group.

        Args:
            row_idx (int): Row index for the current group in the subplot grid.
            group_name (str): Name of the group being plotted.
            fold_data (Dict): Dictionary containing fold data for the group.
            axes (plt.Axes): Matplotlib axes object for the subplots.
            max_values (Dict): Dictionary of maximum values for normalization.
            colormaps (List): List of colormaps to use for visualization.
            total_cols (int): Total number of columns in the subplot grid.
        """
        folds = list(fold_data.keys())

        for col_idx in range(total_cols):
            ax = axes[row_idx, col_idx]

            if col_idx < len(folds):
                self._plot_single_matrix(ax, group_name, folds[col_idx],
                                         fold_data[folds[col_idx]],
                                         colormaps[col_idx % len(colormaps)],
                                         max_values[folds[col_idx]])
            else:
                ax.axis('off')

    def _plot_single_matrix(self, ax, group_name, fold_name, values, colormap, max_value):
        """
        Plot a single confusion matrix in a subplot.

        Args:
            ax (plt.Axes): Matplotlib axes object for the subplot.
            group_name (str): Name of the group being plotted.
            fold_name (str): Name of the fold being plotted.
            values (Dict): Dictionary containing confusion matrix values.
            colormap: Seaborn color palette to use for the matrix.
            max_value (float): Maximum value for color normalization.
        """
        confusion_matrix = self._create_confusion_matrix(values)

        self._add_matrix_values(ax, confusion_matrix)
        self._configure_subplot(ax, group_name, fold_name)
        self._add_colorbar(ax, confusion_matrix, colormap, max_value)
        self._configure_axes(ax)

    @staticmethod
    def _create_confusion_matrix(values):
        """
        Create numpy array for the confusion matrix from evaluation values.

        Args:
            values (Dict): Dictionary containing:
                          {
                              'TruePositive': int,
                              'FalsePositive': int,
                              'FalseNegative': int,
                              'TrueNegative': int
                          }

        Returns:
            numpy.ndarray: 2x2 confusion matrix array.
        """
        return numpy.array([[values['TruePositive'], values['FalsePositive']],
                            [values['FalseNegative'], values['TrueNegative']]])

    def _add_matrix_values(self, ax, matrix):
        """
        Add text values to each cell of the confusion matrix.

        Args:
            ax (plt.Axes): Matplotlib axes object.
            matrix (numpy.ndarray): 2x2 confusion matrix.
        """
        for row in range(len(self.ticks_labels)):
            for col in range(len(self.ticks_labels)):
                ax.text(col, row, str(matrix[row, col]),
                        ha="center", va="center", color="black")

    def _configure_subplot(self, ax, group_name, fold_name):
        """
        Set subplot title and basic properties.

        Args:
            ax (plt.Axes): Matplotlib axes object.
            group_name (str): Name of the group.
            fold_name (str): Name of the fold.
        """
        ax.set_title(f"{group_name} {fold_name}", fontsize=self.title_font_size)

    def _add_colorbar(self, ax, matrix, colormap, max_value):
        """
        Add colorbar to the subplot.

        Args:
            ax (plt.Axes): Matplotlib axes object.
            matrix (numpy.ndarray): 2x2 confusion matrix.
            colormap: Seaborn color palette.
            max_value (float): Maximum value for color normalization.
        """
        cax = ax.matshow(matrix, cmap=colormap, vmin=0, vmax=max_value)
        cbar = plt.colorbar(cax, ax=ax, orientation='vertical', fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=self.ticks_scale_font_size)

    def _configure_axes(self, ax):
        """
        Configure axis labels and ticks for the confusion matrix.

        Args:
            ax (plt.Axes): Matplotlib axes object.
        """
        # Y-axis configuration
        ax.set_ylabel(self.y_label, fontsize=self.labels_font_size)
        ax.set_yticks(numpy.arange(len(self.ticks_labels)))
        ax.set_yticklabels(self.ticks_labels, fontsize=self.ticks_labels_font_size, rotation=90, va="center")

        # X-axis configuration
        ax.set_xlabel(self.x_label, fontsize=self.labels_font_size)
        ax.set_xticks(numpy.arange(len(self.ticks_labels)))
        ax.set_xticklabels(self.ticks_labels, fontsize=self.ticks_labels_font_size)
        ax.tick_params(axis='x', which='both', labelbottom=True, labeltop=False, bottom=True, top=False)

    def _finalize_classifier_plot(self, fig, classifier_name):
        """
        Finalize and save the plot for a classifier.

        Args:
            fig (plt.Figure): Matplotlib figure object.
            classifier_name (str): Name of the classifier.
        """
        fig.suptitle(f"{self.title} - {classifier_name}", fontsize=self.sub_title_font_size)
        plt.tight_layout()

        create_directory(self.save_path)
        save_figure = f"{self.save_path}/matrix_{classifier_name} - {self.title}.pdf"
        plt.savefig(save_figure, format="pdf")
        print(f"New figure created: {save_figure}")
        plt.close()

    def _get_values(self, data):
        """
        Extract and organize the necessary values from raw data.

        Args:
            data (Dict): Raw input data with nested structure.

        Returns:
            Dict: Reorganized data structure containing only the necessary values:
                  {
                      classifier_name: {
                          group_name: {
                              fold_name: {
                                  'TruePositive': int,
                                  'TrueNegative': int,
                                  'FalsePositive': int,
                                  'FalseNegative': int
                              }
                          }
                      }
                  }
        """

        values = {}

        for group in self.groups:

            if group in data:

                for clf in self.classifiers:
                    if clf in data[group]:

                        if not (clf in values):
                            values[clf] = {}
                        values[clf][group] = {}

                        for index_fold in range(1, len(data[group][clf])):
                            fold = f"{index_fold}-Fold"

                            if fold in data[group][clf]:

                                values[clf][group][fold] = {}
                                data_fold = data[group][clf].get(fold, {})
                                values[clf][group][fold]["TruePositive"] = data_fold.get("TruePositive")
                                values[clf][group][fold]["TrueNegative"] = data_fold.get("TrueNegative")
                                values[clf][group][fold]["FalsePositive"] = data_fold.get("FalsePositive")
                                values[clf][group][fold]["FalseNegative"] = data_fold.get("FalseNegative")
        return values

    @property
    def groups(self) -> List[str]:
        """Get the list of group names to include in the plots.

        Returns:
            List[str]: Current list of group names.
        """
        return self._groups

    @groups.setter
    def groups(self, value: List[str]) -> None:
        """Set the list of group names to include in the plots.

        Args:
            value (List[str]): New list of group names.
        """
        self._groups = value

    @property
    def metrics(self) -> List[str]:
        """Get the list of metrics being considered.

        Returns:
            List[str]: Current list of metrics.
        """
        return self._metrics

    @metrics.setter
    def metrics(self, value: List[str]) -> None:
        """Set the list of metrics to consider.

        Args:
            value (List[str]): New list of metrics.
        """
        self._metrics = value

    @property
    def classifiers(self) -> List[str]:
        """Get the list of classifier names being plotted.

        Returns:
            List[str]: Current list of classifier names.
        """
        return self._classifiers

    @classifiers.setter
    def classifiers(self, value: List[str]) -> None:
        """Set the list of classifier names to plot.

        Args:
            value (List[str]): New list of classifier names.
        """
        self._classifiers = value

    @property
    def x_label(self) -> str:
        """Get the current x-axis label for confusion matrices.

        Returns:
            str: Current x-axis label.
        """
        return self._x_label

    @x_label.setter
    def x_label(self, value: str) -> None:
        """Set the x-axis label for confusion matrices.

        Args:
            value (str): New x-axis label.
        """
        self._x_label = value

    @property
    def y_label(self) -> str:
        """Get the current y-axis label for confusion matrices.

        Returns:
            str: Current y-axis label.
        """
        return self._y_label

    @y_label.setter
    def y_label(self, value: str) -> None:
        """Set the y-axis label for confusion matrices.

        Args:
            value (str): New y-axis label.
        """
        self._y_label = value

    @property
    def ticks_labels(self) -> List[str]:
        """Get the current tick labels for confusion matrix axes.

        Returns:
            List[str]: Current tick labels.
        """
        return self._ticks_labels

    @ticks_labels.setter
    def ticks_labels(self, value: List[str]) -> None:
        """Set the tick labels for confusion matrix axes.

        Args:
            value (List[str]): New tick labels.
        """
        self._ticks_labels = value

    @property
    def title(self) -> str:
        """Get the current main title for the plots.

        Returns:
            str: Current main title.
        """
        return self._title

    @title.setter
    def title(self, value: str) -> None:
        """Set the main title for the plots.

        Args:
            value (str): New main title.
        """
        self._title = value

    @property
    def sub_title_font_size(self) -> int:
        """Get the current subtitle font size.

        Returns:
            int: Current subtitle font size in points.
        """
        return self._sub_title_font_size

    @sub_title_font_size.setter
    def sub_title_font_size(self, value: int) -> None:
        """Set the subtitle font size.

        Args:
            value (int): New subtitle font size in points.
        """
        self._sub_title_font_size = value

    @property
    def title_font_size(self) -> int:
        """Get the current title font size.

        Returns:
            int: Current title font size in points.
        """
        return self._title_font_size

    @title_font_size.setter
    def title_font_size(self, value: int) -> None:
        """Set the title font size.

        Args:
            value (int): New title font size in points.
        """
        self._title_font_size = value

    @property
    def labels_font_size(self) -> int:
        """Get the current axis labels font size.

        Returns:
            int: Current axis labels font size in points.
        """
        return self._labels_font_size

    @labels_font_size.setter
    def labels_font_size(self, value: int) -> None:
        """Set the axis labels font size.

        Args:
            value (int): New axis labels font size in points.
        """
        self._labels_font_size = value

    @property
    def ticks_labels_font_size(self) -> int:
        """Get the current tick labels font size.

        Returns:
            int: Current tick labels font size in points.
        """
        return self._ticks_labels_font_size

    @ticks_labels_font_size.setter
    def ticks_labels_font_size(self, value: int) -> None:
        """Set the tick labels font size.

        Args:
            value (int): New tick labels font size in points.
        """
        self._ticks_labels_font_size = value

    @property
    def ticks_scale_font_size(self) -> int:
        """Get the current colorbar tick labels font size.

        Returns:
            int: Current colorbar tick labels font size in points.
        """
        return self._ticks_scale_font_size

    @ticks_scale_font_size.setter
    def ticks_scale_font_size(self, value: int) -> None:
        """Set the colorbar tick labels font size.

        Args:
            value (int): New colorbar tick labels font size in points.
        """
        self._ticks_scale_font_size = value