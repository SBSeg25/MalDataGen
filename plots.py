#!/usr/bin/env python3
# -*- coding: utf-8 -*-

 
 
from Tools.ClusteringVisualizer import ClusteringVisualizer

# MIT License
#
# Copyright (c) 2025 2025 MalDataGen
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
    import argparse
    from typing import Dict
    from dataclasses import dataclass
    import json

    from Engine.data_io.CSVLoader import CSVDataProcessor
    from Tools.PlotHeatMap import HeatmapComparator

    from Tools.PlotTrainingCurve import PlotTrainingCurve

    from Tools.PlotDistanceMetrics import PlotDistanceMetrics
    from Tools.PlotConfusionMatrix import PlotConfusionMatrix
    import Tools.config as config
    from Tools.PlotClasssificationMetrics import PlotClassificationMetrics
    from plots_svm import plot_heatmap_svm
except ImportError as error:
    print(error)
    sys.exit(-1)

"""
SYNDATAGEN VISUALIZATION SUITE - Demonstration
============================================================

This script provides a complete demonstration of SynDataGen's integrated visualization capabilities
for machine learning workflows. It serves as both a production-ready tool and educational reference
implementation.

Key Features:
------------
1. Multi-modal Visualization:
   - Comparative analysis: Side-by-side heatmap comparisons
   - Model diagnostics: Training curves, confusion matrices
   - Statistical analysis: Distance metrics, classification reports

2. Advanced Configuration:
   - Dynamic parameterization via command line interface
   - Flexible data handling for various input formats
   - Customizable visualization aesthetics

3. Production-Grade Features:
   - Comprehensive error handling
   - Batch processing for multiple folds
   - Automated output organization

Implementation Notes:
---------------------
- Uses matplotlib/seaborn for backend rendering
- Implements proper figure sizing for publication-quality outputs
- Supports both interactive display and file export modes

"""
def list_of_strs(arg):
    return list(map(str, arg.split(',')))
@dataclass
class Arguments:
    data_load_label_column: str = 'class'
    data_load_max_samples: int = -1
    data_load_max_columns: int = -1
    data_load_start_column: int = 0
    data_load_end_column: int = 50
    data_load_path_file_input: str = ''
    data_load_path_file_output: str = ''
    data_load_exclude_columns: list = None
    number_samples_per_class: dict = None

def plot_heatmaps_from_dataset_comparison(dataset_path: str, synthetic_path: str, output_file: str):
    def load_data(path):
        processor = CSVDataProcessor(Arguments(data_load_path_file_input=path,
                                               data_load_path_file_output=''))  # Initialize the output path
        processor.load_csv()
        number_columns = processor.get_number_columns()
        return (processor.get_features_by_label(1)[:number_columns, :number_columns],
                processor.get_features_by_label(0)[:number_columns, :number_columns], number_columns)

    # Load data for both datasets
    position_1, negative_1, number_cols_1 = load_data(dataset_path)
    position_2, negative_2, number_cols_2 = load_data(synthetic_path)

    for label, (real, synth) in [(1, (position_1, position_2)), (0, (negative_1, negative_2))]:
        HeatmapComparator(
            figure_size=(16, 8),
            color_map='coolwarm',
            titles=(f'Real Dataset (label={label})', f'Synthetic Dataset (label={label})'),
            linewidths=0,
            linecolor='black',
            x_tick_labels=False,
            y_tick_labels=False,
            annotations=False,
            font_scale=1.2,
            color_bar=True,
            color_bar_label="Value Intensity",
            normalize=True,
            grid_shape=(1, 3),
            export_path=f'{output_file}_{"Positive" if label else "Negative"}_class.pdf',
            tight_layout=True,
            style="whitegrid",
            show_difference_matrix=True
        ).plot(real, synth)


def plot_clusters_from_dataset(dataset_path: str, output_file: str,
                               cluster_algo: str = 'agglo',
                               reduction_algo: str = 'pca',
                               n_clusters: int = 8,
                               n_components: int = 2):
    """
    Plots clusters for samples with label 1 from a dataset.

    Args:
        dataset_path: Path to the input CSV file
        output_file: Path to save the output visualization
        cluster_algo: Clustering algorithm ('kmeans', 'dbscan', 'agglo', etc.)
        reduction_algo: Dimensionality reduction method ('pca', 'tsne', 'umap')
        n_clusters: Number of clusters to identify
        n_components: 2 for 2D visualization or 3 for 3D
    """
    # Load and process the data
    processor = CSVDataProcessor(Arguments(data_load_path_file_input=dataset_path,
                                           data_load_path_file_output=''))

    processor.load_csv()

    # Get only samples with label 1
    X = processor.get_features_by_label(1)
    number_cols = processor.get_number_columns()
    X = X[:number_cols, :number_cols]  # Use the same size limitation as in heatmap function

    # Initialize the visualizer
    visualizer = ClusteringVisualizer(
        cluster_algo=cluster_algo,
        reduction_algo=reduction_algo,
        number_clusters=n_clusters,
        number_components=n_components,
        random_state=42,
        point_size=50,
        color_map='viridis',
        title_font_size=14,
        show_legend=True,
        export_path=output_file  # Added export path parameter
    )

    # Preprocess, cluster and visualize
    X_scaled = visualizer.preprocess(X)
    labels = visualizer.cluster(X_scaled)
    visualizer.plot_clusters()



def plot_heatmaps_from_dataset(dataset_path: str, output_file: str):
    processor = CSVDataProcessor(Arguments(data_load_path_file_input=dataset_path,
                                           data_load_path_file_output=''))  # Initialize the output path
    processor.load_csv()
    number_cols = processor.get_number_columns()

    HeatmapComparator(
        figure_size=(16, 8),
        color_map='coolwarm',
        titles=('Positive Sample (label=1)', 'Negative Sample (label=0)'),
        linewidths=0,
        linecolor='black',
        x_tick_labels=False,
        y_tick_labels=False,
        font_scale=1.2,
        color_bar=True,
        color_bar_label="Value Intensity",
        normalize=True,
        grid_shape=(1, 3),
        export_path=output_file,
        tight_layout=True,
        style="whitegrid",
        show_difference_matrix=True).plot(processor.get_features_by_label(1)[:number_cols, :number_cols],
                                          processor.get_features_by_label(0)[:number_cols, :number_cols])


 
def main():

    parser = argparse.ArgumentParser(description='Plots SynDataGen')

    parser.add_argument("--results", "-r", nargs="+", type=list_of_strs, required=True)
    parser.add_argument("--training", "-t", nargs='+', type=str, required=False)
    parser.add_argument("--title", "-i", nargs='+', type=str, default=[""])
    parser.add_argument("--folds", "-k", nargs='+', type=int, default=5)
    parser.add_argument("--dataset", "-d", nargs='+', type=str, required=True)
    parser.add_argument("--output_dir", "-o", nargs='+', type=str, required=True)
    parser.add_argument("--model", "-m", nargs='+', type=str, default="none")
    parser.add_argument("--f_plot", "-f_pl",action='store_true')
    args = parser.parse_args()
    args.results = args.results[0]
 
    if (args.f_plot):
        PlotClassificationMetrics(input_files=[args.results[-1]], title=" - ".join(args.title))
        PlotDistanceMetrics(input_files=[args.results[-1]], title=" - ".join(args.title))
        PlotConfusionMatrix(input_file=args.results[-1], title=" - ".join(args.title))
        output=args.output_dir
        plot_heatmap_svm(input_files=args.results,title= "/".join(output[0].split("/")[:2]))
    else: 
            PlotClassificationMetrics(input_files=args.results, title=" - ".join(args.title))
            PlotDistanceMetrics(input_files=args.results, title=" - ".join(args.title))
            PlotConfusionMatrix(input_file=args.results[0], title=" - ".join(args.title))
    if args.training:
        PlotTrainingCurve(input_file=args.training, title=" - ".join(args.title))

    plot_heatmaps_from_dataset(args.dataset[0],
                               f'{args.output_dir[0]}/EvaluationResults/heat_map_original_data.pdf')

    for k in range(args.folds[0]):
        path = f'{args.output_dir[0]}/DataGenerated/DataOutput_K_fold_{k}_{args.model[0]}.txt'
        plot_heatmaps_from_dataset(path, f'{args.output_dir[0]}/EvaluationResults/heat_map_k_{k}.pdf')
        plot_heatmaps_from_dataset_comparison(args.dataset[0], path,
                                            f'{args.output_dir[0]}/EvaluationResults/Comparison_heat_map_k_{k}')

    plot_clusters_from_dataset(
        dataset_path=args.dataset[0],
        output_file=f'{args.output_dir[0]}/EvaluationResults/ClusterMapOriginalDataPositiveMalware.pdf',
        cluster_algo='agglo',
        reduction_algo='pca',
        n_clusters=5,
        n_components=2
    )


if __name__ == '__main__':
    sys.exit(main())