[![ml-intro](Layout/layout.png)]()
<div align="center">
  <img src="https://img.shields.io/badge/status-active-success.svg" />
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" />
  <img src="https://img.shields.io/badge/GPU-Supported-brightgreen.svg" />
  <a href="https://github.com/SBSeg25/MalDataGen/tree/bf2ecc5858da8c2e057663852cace4235f226600/Docs">
    <img src="https://img.shields.io/badge/docs-API_Reference-blue.svg" />
  </a>
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" />
  <img src="https://img.shields.io/badge/open%20source-yes-green.svg" />
   <div>
    <img src="https://img.shields.io/github/stars/SBSeg25/MalDataGen?style=social" alt="GitHub Stars" />
    <img src="https://img.shields.io/github/forks/SBSeg25/MalDataGen?style=social" alt="GitHub Forks" />
  </div>
</div>
</div>


# 🌊 MalDataGen - v.1.0.0 (Jellyfish 🪼)

SynDataGen is an advanced Python framework for generating and evaluating synthetic tabular datasets using modern generative models, including diffusion and adversarial architectures. Designed for researchers and practitioners, it provides reproducible pipelines, fine-grained control over model configuration, and integrated evaluation metrics for realistic data synthesis.

[![SynDataGen](https://img.shields.io/badge/Webpage%20SynDataGen%20Library-red?style=for-the-badge&logo=github&logoColor=white)](https://kayua.github.io/SyntheticDataGen.github.io/)

## Citation

If you use **MalDataGen** in your research, whether for generating synthetic data, reproducing results, or as part of your malware detection pipeline, please cite our paper:

```bibtex
@inproceedings{sbseg25_maldatagen,
 author = {Kayuã Paim and Angelo Nogueira and Diego Kreutz and Weverton Cordeiro and Rodrigo Mansilha},
 title = { MalDataGen: A Modular Framework for Synthetic Tabular Data Generation in Malware Detection},
 booktitle = {Companion Proceedings of the 25th Brazilian Symposium on Cybersecurity},
 location = {Foz do Iguaçu/PR},
 year = {2025},
 keywords = {},
 issn = {0000-0000},
 pages = {38--47},
 publisher = {SBC},
 address = {Porto Alegre, RS, Brasil},
 doi = {10.5753/sbseg_estendido.2025.12113},
 url = {https://sol.sbc.org.br/index.php/sbseg_estendido/article/view/36739}
}
```


---

## 📚 Table of Contents (Readme.md)

 
- [📖 Overview (Informações básicas)](#overview)
- [Video](#overview)
- [Security worries](#securty_worries)
- [Awards Received](#stamps)
- [🚀 Getting Started](#getting-started)
- [⚙️ Installation (Instalação)](#installation)
- [🧠 Architectures](#architectures)
- [🛠 Features](#features)
- [📊 Evaluation Strategy](#evaluation)
- [📈 Metrics](#metrics)
- [📋 Architecture Diagrams](#architecture-diagrams)
- [🔧 Technologies Used (Dependências)](#technologies)
- [🔗 References](#references)
  

---

## 📖 Overview <a name="overview"></a>

MalDataGen is a modular and extensible synthetic data generation library for tabular data for malware dectition. It aims to:

- Support state-of-the-art generative models (GANs, VAEs, Diffusion, etc.)
- Improve model generalization by augmenting training data
- Enable fair benchmarking via reproducible evaluations (TS-TR and TR-TS)
- Provide publication-ready metrics and visualizations

It supports GPU acceleration, CSV/XLS ingestion, custom CLI scripts, and integration with academic pipelines.
---
### Model architecure overivew
WWe provide a visual overview of the internal architecture of each model's building blocks through five detailed figures, highlighting the main structural changes across the models. These diagrams are documented and explained in the Overview.md [Overview.md ] file.(https://github.com/SBSeg25/MalDataGen/blob/2dd9eaad74da7726c130e50dbc35f95a463cbd00/Docs/Overview.md)


### 📋 Architecture Documentation

We provide a comprehensive visual overview (8 diagrams)  at [Docs/Diagrams/](Docs/Diagrams/) of the MalDataGen framework, covering its architecture, design principles, data processing flow, and evaluation strategies. Developed using Mermaid notation, these diagrams support understanding of both the structural and functional aspects of the system. They include high-level system architecture, object-oriented class relationships, evaluation workflows, training pipelines, metric frameworks, and data flow. Together, they offer a detailed and cohesive view of how MalDataGen enables the generation and assessment of synthetic data in cybersecurity contexts.



---

## 📖 Video <a name="Video"></a>
The following link showcases a video of a demonstration of the tool: https://drive.google.com/file/d/1sbPZ1x5Np6zolhFvCBWoMzqNqrthlUe3/view?usp=sharing

if that doesn't work we have a backup on: https://youtu.be/t-AZtsLJUlQ

---

## 🚀 Getting Started <a name="getting-started"></a>

### Prerequisites

- Python 3.10+
- pip
- (Optional) CUDA 11+ for GPU acceleration

### Optional: Create a virtual environment

```bash
pip install virtualenv
python3 -m venv ~/Python3venv/MalDataGen
source ~/Python3venv/MalDataGen/bin/activate
```

---

## ⚙️ Installation <a name="installation"></a>

```bash
git clone https://github.com/SBSeg25/MalDataGen.git
cd MalDataGen
pip install --upgrade pip
pip install -r requirements.txt
# or
pip install .
```
## Security worries <a name="securty_worries"></a>

#### We declare that the local execution of experiments has no security worries, however the docker executing require sudo permissions being available to the docker engine.


## 🏆 Awards Received <a name="stamps"></a>

**Highlighted Artifact**  
Awarded for outstanding contributions in the artifacts category.  
[Details at SBSEG 25](https://doc-artefatos.github.io/sbseg2025/results.html#trabalhos-destaque-na-categoria-artefatos)

**Best Tool of SBSEG 2025**  
Recognized as the most innovative and impactful tool at the symposium.  
[Official award document](https://sbseg2025.ppgia.pucpr.br/wp-content/uploads/2025/09/PremiacaoSBSEG-2025.pdf)


 
### 🚀 Run Tests

#### Demo
In order to execute a demo of the tool, utilized the comand listed below. The execution of this reduced demo takes around 3 minutes on a AMD Ryzen 7 5800x, 8 cores, 64 GB RAM machine.
 
```bash
# Run the basic demo
python3 run_campaign_sbseg.py -c sf
```

Alternatively, you can use the a docker container to execute the demo, by using the following comand:

```bash
# Run the basic demo
./run_demo_docker.sh 
```


#### Reproduction
In order to reproduce the results from the paper execute the comand below, the experiments take around 7 hours on a AMD Ryzen 7 5800x, 8 cores, 64 GB RAM machine.

```bash
# Run all experiments from the paper
python3 run_campaign_sbseg.py 
```

Or to execute with docker:
```bash
# Run all experiments from the paper
./run_experiments_docker.sh  
```

---

##### Expected outputs:
After executing the experiments, you should observe the following structure within the outputs folder, with a separate folder for each model executed:
<img width="1600" height="507" alt="image" src="https://github.com/user-attachments/assets/fafb0516-c227-4aba-8596-679aeb1d68f9" />
A results folder is also present, containing the training curves for each model.

Within each model's folder, there will be five subfolders:

    - Data generated: Contains the synthetic dataset and the partitioned subsets of the real dataset used for training.

    - Evaluation results: Contains:

        - A clustering visualization of the dataset samples to assist in identifying malware families.

        - Heatmaps comparing the synthetic and real samples for each fold; these are intended to illustrate the variability of specific features, with a closer alignment indicating greater similarity.

        - Confusion matrices for each classifier on each fold.

        - A bar graph presenting the metrics for each classifier using the TSTR and TRTS evaluation methods.

    - Logs: Contains the generated logs.

    - Monitor: Contains the raw data collected during the monitoring of the experiment.

    - Models Saved: Contains the saved models for each fold, provided the option to save models was active.

Additionally, a file named "Binary classification metrics for SVM classifier.pdf" should be created in the project's root folder. This file provides a comparison of the SVM classifier's performance across the models, similar to Figure 3 in the article.

## 🧠 Architectures Supported <a name="architectures"></a>

### 🔨 Native Models
| Model                 | Description                                                                 | Use Case                                    |
|-----------------------|-----------------------------------------------------------------------------|---------------------------------------------|
| `CGAN`                | Conditional GANs conditioned on labels or attributes                        | Class balancing, controlled generation      |
| `WGAN`                | Wasserstein GAN with Earth-Mover distance for improved stability            | Imbalanced datasets, stable training        |
| `WGAN-GP`             | Wasserstein GAN with gradient penalty for stable training                   | Imbalanced datasets, complex distributions  |
| `Autoencoder`         | Latent-space learning through compression-reconstruction                    | Feature extraction, denoising               |
| `VAE`                 | Probabilistic Autoencoder with latent sampling                              | Probabilistic generation and imputation     |
| `Denoising Diffusion` | Progressive noise-based generative model                                    | Robust generation with high-quality samples |
| `Latent Diffusion`    | Diffusion model operating in compressed latent space                        | High-resolution generation, efficiency      |
| `VQ-VAE`              | Discrete latent-space via quantization                                      | Categorical and mixed-type data             |
| `SMOTE`               | Synthetic Minority Over-sampling Technique (interpolation-based)            | Class imbalance in tabular data             |


### 📦 Third-Party Supported (SDV)
| Model       | Description                                                                 | Use Case                              |
|-------------|-----------------------------------------------------------------------------|---------------------------------------|
| `TVAE`      | Variational Autoencoder optimized for tabular data                          | Structured/tabular data synthesis     |
| `Copula`    | Statistical model based on dependency (copula) functions                    | Synthetic data with correlations      |
| `CTGAN`     | GAN with mode-specific normalization for tabular data                       | Mixed-type/categorical synthesis      |

*Legenda*:  
- **SDV**: Integração com a biblioteca [Synthetic Data Vault](https://sdv.dev/).
---

## 🛠 Features <a name="features"></a>

- 📊 Cross-validation (stratified k-fold)
- ⚙️ Fully customizable model configuration
- 📈 Built-in metrics for data quality
- 🔁 Persistent models & experiment saving
- 📉 Graphing utilities for visual reports
- 📉 Clustering visualization of datasets
- 📉 Heat maps between the synthetic and real samples
- 🧪 Automated experiment pipelines
- 💾 Data export to CSV/XLS formats

---

## 📊 Evaluation Strategy <a name="evaluation"></a>

Two validation approaches are supported:

- **TS-TR (Train Synthetic – Test Real)**  
  Measures generalization ability by training on synthetic data and testing on real data.

- **TR-TS (Train Real – Test Synthetic)**  
  Assesses generative realism by training on real and testing on synthetic samples.

---

## 📈 Metrics Tracked <a name="metrics"></a>

### Primary

- Accuracy, Precision, Recall, F1-score, Specificity
- ROC-AUC, MSE, MAE, FNR, TNR

### Secondary

- Euclidean Distance, Hellinger Distance
- Log-Likelihood, Manhattan Distance

---


---

## 📋 Architecture Diagrams <a name="architecture-diagrams"></a>

Comprehensive architecture documentation is available in the [Docs/Diagrams/](Docs/Diagrams/) directory, including:

- **System Architecture**: High-level framework overview and component relationships
- **Core Class Hierarchy**: Object-oriented design and inheritance structure
- **Evaluation Strategy**: TS-TR and TR-TS evaluation flow diagrams
- **Model Training Pipeline**: Complete workflow sequence from data to results
- **Metrics Framework**: Comprehensive evaluation metrics overview
- **Data Flow Architecture**: End-to-end data processing pipeline
- **Generative Models Comparison**: Model categories and characteristics
- **Deployment Architecture**: Docker and execution mode options

All diagrams are created using Mermaid format for easy maintenance and version control. They can be viewed directly in GitHub or exported for academic publications.

---

## 🧰 Technologies Used <a name="technologies"></a>

| Tool          | Purpose                     |
|---------------|-----------------------------|
| Python 3.8+   | Core language               |
| NumPy, Pandas | Data processing             |
| TensorFlow    | Model building              |
| Matplotlib, Plotly | Visualization          |
| PyTorch (planned) | Future multi-backend support |
| Docker        | Containerization            |
| Git           | Version control             |

---

## 🔬 System Requirements

### Hardware

| Component | Minimum    | Recommended             |
|----------|------------|--------------------------|
| CPU      | Any x86_64 | Multi-core (i5/Ryzen 5+) |
| RAM      | 4 GB       | 8 GB+                    |
| Storage  | 10 GB      | 20 GB SSD                |
| GPU      | Optional   | NVIDIA with CUDA 11+     |

### Software

| Component | Version    | Notes                    |
|-----------|------------|--------------------------|
| OS        | Ubuntu 22.04+ | Linux preferred       |
| Python    | ≥ 3.8.10   | Virtualenv recommended   |
| Docker    | ≥ 27.2.1   | Optional but supported   |
| Git       | Latest     | Required                 |
| CUDA      | ≥ 11.0     | Optional for GPU execution |

---

## 🔗 References <a name="references"></a>
### How to cite this tool
```bibtex
@inproceedings{sbseg25_maldatagen,
 author = {Kayuã Paim and Angelo Nogueira and Diego Kreutz and Weverton Cordeiro and Rodrigo Mansilha},
 title = { MalDataGen: A Modular Framework for Synthetic Tabular Data Generation in Malware Detection},
 booktitle = {Companion Proceedings of the 25th Brazilian Symposium on Cybersecurity},
 location = {Foz do Iguaçu/PR},
 year = {2025},
 keywords = {},
 issn = {0000-0000},
 pages = {38--47},
 publisher = {SBC},
 address = {Porto Alegre, RS, Brasil},
 doi = {10.5753/sbseg_estendido.2025.12113},
 url = {https://sol.sbc.org.br/index.php/sbseg_estendido/article/view/36739}
}
```

### Core Papers
- [Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)  
- [Goodfellow, I. et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661)  
- [Ho, J. et al. (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)  
- [Oord, A. v. d. et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)  
- [Arjovsky, M. et al. (2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875)  

### SDV Ecosystem
- [Patki, N. et al. (2016). The Synthetic Data Vault](https://arxiv.org/abs/1811.11264)  
- [Xu, L. et al. (2019). Modeling Tabular Data using Conditional GAN](https://arxiv.org/abs/1907.00503)  

### Supplementary
- [Mirza, M. & Osindero, S. (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)  
- [Gulrajani, I. et al. (2017). Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)  


## 🧩 License

Distributed under the MIT License. See LICENSE for more information.
