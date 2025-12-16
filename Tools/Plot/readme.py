import json
import os
from datetime import datetime


def generate_readme(json_data, output_dir="."):
    # Extract dataset information
    dataset_info = {
        "Name": os.path.basename(json_data["arguments"]["data_load_path_file_input"]),
        "Size": "N/A",  # Not provided in JSON
        "Samples per class": json_data["arguments"]["number_samples_per_class"],
        "Number of columns": "N/A",  # Not provided in JSON
        "Format": os.path.splitext(json_data["arguments"]["data_load_path_file_input"])[1][1:].upper(),
        "Model type": json_data["arguments"]["model_type"]
    }

    # Model information
    model_type = json_data["arguments"]["model_type"]
    model_info = {
        "Type": model_type,
        "Latent dimension": json_data["arguments"].get("autoencoder_latent_dimension", "N/A"),
        "Epochs": json_data["arguments"].get("autoencoder_number_epochs", "N/A"),
        "Batch size": json_data["arguments"].get("autoencoder_batch_size", "N/A"),
        "Learning rate": json_data["arguments"].get("adam_optimizer_learning_rate", "N/A"),
        "Activation": json_data["arguments"].get("autoencoder_activation_function", "N/A")
    }

    # Get current date
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Create README content
    readme_content = f"""# Synthetic Data Generation Report - {dataset_info['Name']}

**Date:** {current_date}  
**Model Type:** {model_type.upper()}  

---

## 1. Dataset Information
| **Characteristic**       | **Value**                          |
|--------------------------|------------------------------------|
| **Name**                 | {dataset_info['Name']}             |
| **Format**               | {dataset_info['Format']}           |
| **Samples per class**    | Class 0: {dataset_info['Samples per class']['classes']['0']}, Class 1: {dataset_info['Samples per class']['classes']['1']} |
| **Number of classes**    | {dataset_info['Samples per class']['number_classes']} |
| **Model type**           | {dataset_info['Model type']}       |

---

## 2. Model Configuration
| **Parameter**            | **Value**                          |
|--------------------------|------------------------------------|
| Latent dimension         | {model_info['Latent dimension']}   |
| Training epochs          | {model_info['Epochs']}             |
| Batch size               | {model_info['Batch size']}         |
| Learning rate            | {model_info['Learning rate']}      |
| Activation function      | {model_info['Activation']}         |

---

## 3. Performance Metrics Summary

### 3.1 Training on Synthetic, Testing on Real (TS-TR)
"""

    # Add TS-TR results
    classifiers = list(json_data["TS-TR"].keys())
    for classifier in classifiers:
        summary = json_data["TS-TR"][classifier]["Summary"]
        readme_content += f"""
#### {classifier}
- **Accuracy:** {summary['Accuracy']['mean']:.4f} ± {summary['Accuracy']['std']:.4f}
- **Precision:** {summary['Precision']['mean']:.4f} ± {summary['Precision']['std']:.4f}
- **Recall:** {summary['Recall']['mean']:.4f} ± {summary['Recall']['std']:.4f}
- **F1 Score:** {summary['F1Score']['mean']:.4f} ± {summary['F1Score']['std']:.4f}
"""

    readme_content += """
### 3.2 Training on Real, Testing on Synthetic (TR-TS)
"""

    # Add TR-TS results
    for classifier in classifiers:
        summary = json_data["TR-TS"][classifier]["Summary"]
        readme_content += f"""
#### {classifier}
- **Accuracy:** {summary['Accuracy']['mean']:.4f} ± {summary['Accuracy']['std']:.4f}
- **Precision:** {summary['Precision']['mean']:.4f} ± {summary['Precision']['std']:.4f}
- **Recall:** {summary['Recall']['mean']:.4f} ± {summary['Recall']['std']:.4f}
- **F1 Score:** {summary['F1Score']['mean']:.4f} ± {summary['F1Score']['std']:.4f}
"""

    # Add distance metrics
    distance_metrics = json_data["DistanceMetrics"]["R-S"]["Summary"]
    readme_content += f"""
---

## 4. Distance Metrics (Real vs Synthetic)
| **Metric**              | **Value**                          |
|--------------------------|------------------------------------|
| Euclidean Distance       | {distance_metrics['EuclideanDistance']['mean']:.4f} ± {distance_metrics['EuclideanDistance']['std']:.4f} |
| Hellinger Distance       | {distance_metrics['HellingerDistance']['mean']:.4f} ± {distance_metrics['HellingerDistance']['std']:.4f} |
| Manhattan Distance       | {distance_metrics['ManhattanDistance']['mean']:.4f} ± {distance_metrics['ManhattanDistance']['std']:.4f} |
| Hamming Distance         | {distance_metrics['HammingDistance']['mean']:.4f} ± {distance_metrics['HammingDistance']['std']:.4f} |
| Jaccard Distance         | {distance_metrics['JaccardDistance']['mean']:.4f} ± {distance_metrics['JaccardDistance']['std']:.4f} |

---

## 5. Efficiency Metrics
"""

    # Add efficiency metrics
    eff_metrics = json_data["EfficiencyMetrics"]["Summary"]
    readme_content += f"""
| **Metric**              | **Value**                          |
|--------------------------|------------------------------------|
| Training Time (ms)       | {eff_metrics['Time_training_ms']['mean']:.2f} ± {eff_metrics['Time_training_ms']['std']:.2f} |
| Generation Time (ms)     | {eff_metrics['Time_generating_ms']['mean']:.2f} ± {eff_metrics['Time_generating_ms']['std']:.2f} |
| Process CPU Usage (%)    | {eff_metrics['Process_CPU_%']['mean']:.2f} ± {eff_metrics['Process_CPU_%']['std']:.2f} |
| Process Memory (MB)      | {eff_metrics['Process_Memory_MB']['mean']:.2f} ± {eff_metrics['Process_Memory_MB']['std']:.2f} |

---

## 6. Visualizations
*(Note: In a complete implementation, this section would include paths to generated plots)*

- Confusion matrices for each classifier
- Training curves
- Feature distribution comparisons
- Distance metric visualizations

---

## 7. Conclusion
- **Best performing classifier:** [To be determined based on metrics]
- **Data similarity:** The distance metrics indicate [low/moderate/high] similarity between real and synthetic data
- **Efficiency:** The model shows [good/poor] computational efficiency with average training time of {eff_metrics['Time_training_ms']['mean']:.2f}ms

"""

    # Save to file
    output_path = os.path.join(output_dir, "README.md")
    with open(output_path, "w") as f:
        f.write(readme_content)

    return output_path


# Example usage:
if __name__ == "__main__":
    # Load your JSON data
    with open("results.json") as f:
        data = json.load(f)

    # Generate README
    readme_path = generate_readme(data)
    print(f"README generated at: {readme_path}")