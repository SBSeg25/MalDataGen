#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Synthetic Ocean AI']

from Tools.ClusteringVisualizer import ClusteringVisualizer

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
    import argparse
    import os
    from typing import List
    from typing import Dict
    from dataclasses import dataclass
    import json
    import seaborn as sns
    from Engine.data_io.CSVLoader import CSVDataProcessor
    from Tools.PlotHeatMap import HeatmapComparator

    from Tools.PlotTrainingCurve import PlotTrainingCurve
    import pandas as pd
    from Tools.PlotDistanceMetrics import PlotDistanceMetrics
    from Tools.PlotConfusionMatrix import PlotConfusionMatrix
    import Tools.config as config

    import matplotlib.pyplot as plt
    from Tools.Plot import Plot
    from Tools.PlotClasssificationMetrics import PlotClassificationMetrics

except ImportError as error:
    print(error)
    sys.exit(-1)
def list_of_strs(arg):
    return list(map(str, arg.split(',')))
def _get_values(data: Dict) -> Dict:
    values = {}

    # Process each configured group
    for group in config.GROUPS:
        if group in data:
            values[group] = {}

            # Only process SVM classifier
            clf =   "SupportVectorMachine"
            if clf in data[group]:
                values[group][clf] = {}

                # Extract summary statistics
                fold_metrics = data[group][clf].get('Summary', {})

                if fold_metrics:
                    # Extract each configured metric
                    for metric in config.METRICS:
                        if metric in fold_metrics:
                            values[group][clf][metric] = {
                                'mean': fold_metrics[metric].get('mean'),
                                'std': fold_metrics[metric].get("std")
                            }
    return values



def plot_heatmap_svm(input_files: list,title:str):
    # Model name mapping from directory to display names
    MODEL_MAPPING = {
        'adversarial': 'cGAN',
        'adversarial_demo': 'cGAN',
        'autoencoder': 'AE',
        'variational': 'VAE',
        'variational_demo': 'VAE',
        'quantized': 'VQ-VAE',
        'wasserstein': 'WGAN',
        'wasserstein_gp': 'WGAN-GP',
        'latent_diffusion': 'LDM',
        'copula': 'Copula',
        'tvae': 'TVAE',
        'ctgan': 'CTGAN'
    }

    base_metrics = [
        "Accuracy",
        "Precision", 
        "Recall",
        "F1Score",
        "Specificity",
        "MeanSquareError",
        "MeanAbsoluteError"
    ]
    
    # Create metric order with TSTR and TRTS variants
    metric_order = []
    for metric in base_metrics:
        metric_order.append(f"{metric} TSTR")
        metric_order.append(f"{metric} TRTS")
    
    # Initialize model data structure with all possible models
    model_data = {display_name: [] for display_name in MODEL_MAPPING.values()}
    
    # Collect data from files
    for file_path in input_files:
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                
                # Extract model name from path (4th segment in path)
                path_parts = file_path.split('/')
                dir_name = path_parts[3] if len(path_parts) > 3 else ''
                display_name = MODEL_MAPPING.get(dir_name, None)
                
                if not display_name:
                    print(f"Skipping unknown model type from path: {file_path}")
                    continue
                
                # Initialize metrics collection
                current_metrics = []
                valid_metrics = True
                
                # Process both evaluation groups
                for group_name in ['TS-TR', 'TR-TS']:
                    if group_name not in data:
                        print(f"Missing {group_name} group in {file_path}")
                        valid_metrics = False
                        continue
                    
                    if 'SupportVectorMachine' not in data[group_name]:
                        print(f"Missing SVM classifier in {group_name} group for {file_path}")
                        valid_metrics = False
                        continue
                    
                    svm_data = data[group_name]['SupportVectorMachine']
                    if 'Summary' not in svm_data:
                        print(f"Missing Summary in SVM classifier for {group_name} in {file_path}")
                        valid_metrics = False
                        continue
                    
                    # Collect all metrics for this group
                    summary = svm_data['Summary']
                    for metric in base_metrics:
                        if metric in summary:
                            current_metrics.append(summary[metric]['mean'])
                        else:
                            print(f"Missing {metric} in {group_name} for {display_name}")
                            valid_metrics = False
                
                # Only add if we got all expected metrics
                if valid_metrics and len(current_metrics) == len(metric_order):
                    model_data[display_name] = current_metrics
                else:
                    print(f"Incomplete metrics for {display_name}, expected {len(metric_order)} got {len(current_metrics)}")
                    
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    # Prepare DataFrame - only include models with data
    metrics = []
    models = []
    values = []
    
    for model_name, model_values in model_data.items():
        if model_values:  # Only include models with complete data
            metrics.extend(metric_order)
            models.extend([model_name] * len(metric_order))
            values.extend(model_values)
    
    if not values:
        print("No valid model data found to plot")
        return
    
    df = pd.DataFrame({
        "Metric": metrics,
        "Model": models,
        "Value": values
    })
    
    # Create heatmap data with consistent ordering
    heatmap_data = df.pivot(index="Metric", columns="Model", values="Value")
    heatmap_data = heatmap_data.loc[metric_order]
    
    # Define and apply column order (only including models that have data)
    column_order = [name for name in MODEL_MAPPING.values() if name in heatmap_data.columns]
    heatmap_data = heatmap_data[column_order]
    
    # Plotting
    plt.figure(figsize=(max(10, len(column_order) * 1.5), 10))
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        cmap="YlGnBu",
        fmt=".2f",
        linewidths=0.5,
        vmin=0.0,
        vmax=1.0,
        annot_kws={"size": 8}
    )
    
    # Add separator line between TSTR and TRTS metrics
    ax.axhline(y=len(base_metrics), color='black', linewidth=2)
    
    plt.title("SVM  performance utility_across_models", pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Save and show
    plt.savefig(title+"/"+'svm_utility_across_models.pdf', format='pdf', bbox_inches='tight')
    plt.show()