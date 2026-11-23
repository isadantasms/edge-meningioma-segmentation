<div align="center">

# Semantic Segmentation of Meningiomas via Knowledge Distillation and Explainable AI for Computationally Constrained Environments

[![Language: English](https://img.shields.io/badge/Language-English-blue.svg)](#)
[![Versão em Português](https://img.shields.io/badge/Vers%C3%A3o-Portugu%C3%AAs-green.svg)](README.pt-BR.md)
[![Institution: UFU](https://img.shields.io/badge/UFU-FACOM-003366.svg)](https://facom.ufu.br/)
[![Program: PET Information Systems](https://img.shields.io/badge/PET-Information%20Systems-orange.svg)](https://www.petfacom.ufu.br/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-In%20Progress-brightgreen.svg)]()
[![Documentation](https://img.shields.io/badge/Official%20Documents-Attached-success.svg)](#-project-documentation-and-attachments)
[![Notion Workspace](https://img.shields.io/badge/Notion-Research%20Workspace-000000.svg?logo=notion&logoColor=white)](https://app.notion.com/p/Segmenta-o-Sem-ntica-de-Meningiomas-via-Destila-o-de-Conhecimento-e-Intelig-ncia-Artificial-Explic-4070fdd8eda580a9812afa1ed45d73f6)

**[🇺🇸 English](README.md) | [🇧🇷 Português](README.pt-BR.md)**

<p align="justify">
Individual scientific research project developed under the <b>Tutorial Education Program (PET - Information Systems)</b> at the <b>Faculty of Computing (FACOM)</b>, <b>Federal University of Uberlândia (UFU)</b>, Brazil.
</p>

</div>

---

## 📌 Project Identification

- **Student Researcher:** Isabela Dantas de Moraes ([isabeladantasms@ufu.br](mailto:isabeladantasms@ufu.br) / Student ID: `12521BSI221`)
- **Faculty Advisor:** Prof. Dr. Marcelo Zanchetta do Nascimento ([marcelo.nascimento@ufu.br](mailto:marcelo.nascimento@ufu.br) / SIAPE: `1546624`)
- **PET Group Tutor:** Prof. Dr. Thiago Pirola Ribeiro
- **Academic Unit:** Faculty of Computing (FACOM), Federal University of Uberlândia (UFU), Brazil
- **Research Duration:** 12 months (November 2026 – October 2027)
- **Interactive Lab Notes:** [Notion Research Workspace](https://app.notion.com/p/Segmenta-o-Sem-ntica-de-Meningiomas-via-Destila-o-de-Conhecimento-e-Intelig-ncia-Artificial-Explic-4070fdd8eda580a9812afa1ed45d73f6)

---

## 📖 Abstract & Overview

**Meningiomas** represent the most common type of primary central nervous system (CNS) tumors, accounting for over 40% of diagnosed cases. Accurate tumor boundary delineation on **Magnetic Resonance Imaging (MRI)** scans is essential for neurosurgical planning and longitudinal clinical monitoring. However, manual delineation is labor-intensive, time-consuming, and heavily reliant on specialized neuroradiologists.

While modern **deep learning (DL)** models achieve state-of-the-art segmentation performance, conventional high-capacity architectures demand substantial computational resources (high-end GPUs and large memory footprints) that are typically unavailable in low-resource clinics and public healthcare systems, such as Brazil's Unified Health System (**SUS**). Furthermore, clinical adoption of deep learning is often hindered by the "black-box" nature of neural networks.

This project introduces a **two-dimensional semantic segmentation approach for meningiomas** utilizing **Knowledge Distillation (KD)** and **Explainable Artificial Intelligence (XAI)** tailored for **computationally constrained environments (standard CPUs)**:
1. **Teacher-Student Distillation:** A high-capacity teacher network transfers structural and boundary representations to a lightweight, compact student network.
2. **CPU-Feasible Inference:** The student network maintains high segmentation precision while drastically cutting parameter count, RAM consumption, and execution latency on conventional commodity CPUs.
3. **Interpretability & Transparency:** Predictions are audited using local interpretability techniques (**LIME**) and layer activation feature maps, verifying anatomical plausibility and supporting specialist clinical review.

---

## 🎯 Objectives

### General Objective
To develop and evaluate a semantic segmentation framework for meningiomas on brain MRI scans using **Knowledge Distillation**, investigating the trade-offs between **segmentation accuracy**, **CPU computational efficiency**, and **prediction explainability**.

### Specific Objectives
1. **Dataset Characterization & Curation:** Curate and characterize the public **BRISC** benchmark dataset, filtering meningioma cases across axial, coronal, and sagittal planes, standardizing image dimensions and spatial intensity distributions.
2. **Robust Data Partitioning:** Establish training, validation, and testing partitions with rigorous patient-level separation to eliminate data leakage.
3. **Baseline Architecture Modeling:** Train a high-capacity **Teacher Network** and an uncompressed **Student Network** (trained from scratch without distillation) as baseline references.
4. **Knowledge Distillation Implementation:** Train the student network using multi-objective distillation losses:
   - Soft probabilistic output alignment with temperature scaling ($\mathcal{L}_{\text{KD}}$);
   - Feature-aligned intermediate representation distillation ($\mathcal{L}_{\text{feat}}$);
   - Supervised segmentation loss with ground-truth masks ($\text{Cross-Entropy} + \text{Dice Loss}$).
5. **CPU Efficiency & Latency Benchmark:** Benchmark memory footprint (RAM), parameter count, model disk size, and end-to-end CPU inference latency per slice under controlled hardware conditions.
6. **Explainability & Error Audit (XAI):** Adapt **LIME** for dense semantic predictions and analyze intermediate feature activation maps, evaluating alignment with annotated tumor boundaries and diagnosing failure cases.
7. **Open Science & Reproducibility:** Document network hyperparameters, library versions, random seeds, and pipeline scripts to ensure full scientific reproducibility.

---

## 🔬 Methodology & Workflow

```
+-------------------------------------------------------------+
|                     BRISC Dataset (2025)                    |
|   (T1-weighted contrast-enhanced MRI: Axial, Coronal, Sag.) |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|               Preprocessing & Standardization               |
|      (Intensity normalization, spatial resizing, train aug) |
+-------------------------------------------------------------+
                               |
            +------------------+------------------+
            |                                     |
            v                                     v
+------------------------+             +----------------------+
|    Teacher Network     |             |   Student Network    |
| (High Capacity Model)  |             | (Compact / Edge-CPU) |
+------------------------+             +----------------------+
            |                                     |
            +-------- Knowledge Distillation -----+
            |   - Soft output logits (Temp T)     |
            |   - Intermediate feature alignment  |
            |   - Total Loss: CE + Dice + KD      |
            |                                     |
            +-----------------> v <---------------+
                                |
             +------------------+------------------+
             |                                     |
             v                                     v
+-------------------------+           +-------------------------+
|  Segmentation Metrics   |           | Computational Efficiency|
|  - Dice Similarity (DSC)|           |  - CPU Latency (ms)     |
|  - IoU (Jaccard Index)  |           |  - Allocated RAM (MB)   |
|  - HD95 (95% Hausdorff) |           |  - Number of Parameters |
+-------------------------+           +-------------------------+
             \                                     /
              \                                   /
               +----------------+----------------+
                                |
                                v
               +---------------------------------+
               | Explainability & Auditability   |
               |  - LIME for Dense Predictions   |
               |  - Intermediate Activation Maps |
               +---------------------------------+
```

### Distillation Loss Formulation
The student network is optimized via a composite loss function:
$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{seg}}(y, \hat{y}_{\text{student}}) + (1 - \alpha) \mathcal{L}_{\text{KD}}(\hat{y}_{\text{student}}, \hat{y}_{\text{teacher}}; T) + \beta \mathcal{L}_{\text{feat}}(F_{\text{student}}, F_{\text{teacher}})$$

Where:
- $\mathcal{L}_{\text{seg}}$ is the supervised loss (Dice + Cross-Entropy) against ground-truth manual annotations $y$.
- $\mathcal{L}_{\text{KD}}$ represents the Kullback-Leibler divergence between temperature-softened logits ($T > 1$).
- $\mathcal{L}_{\text{feat}}$ enforces feature representation alignment across selected encoder/decoder layers.
- $\alpha, \beta$ are balancing hyperparameters determined on the validation split.

---

## 📓 Preliminary Experiments & Notebooks

The preliminary exploratory tasks and baseline pipelines conducted during the project initiation are preserved in the [`notebooks/`](./notebooks/) directory:

| Notebook | Focus | Description | Link |
| :--- | :---: | :--- | :---: |
| 🧪 **01 — Classification (ConvNeXt-Atto)** | Edge Classification | Preliminary baseline pipeline for multi-class brain tumor classification (*glioma, meningioma, pituitary, no tumor*) using ConvNeXt-Atto with reproducible splits. | [Open Notebook](./notebooks/01_classification_convnext_atto.ipynb) |
| 🎯 **02 — Segmentation (SegFormer Baseline)** | Semantic Segmentation | Binary semantic segmentation pipeline validating patch extraction, training loops, and initial Dice/IoU tracking for brain tumors. | [Open Notebook](./notebooks/02_segmentation_segformer.ipynb) |

---

## 📅 Project Timeline (12 Months)

| Period | Milestone / Planned Activities |
| :---: | :--- |
| **11/2026** | Literature review; BRISC dataset acquisition and licensing verification; computational environment setup; preliminary preprocessing pipeline implementation. |
| **01/2027** | Definition of data splits; architecture selection; baseline training of Teacher and Student (without distillation). |
| **03/2027** | Output distillation implementation; hyperparameter tuning on the validation set; logging of preliminary experiments. |
| **05/2027** | Investigation of intermediate feature alignment; initial integration of XAI (LIME); ablation studies and experiment repetition. |
| **07/2027** | Final evaluation on the unseen test set; CPU runtime latency and RAM profiling; error analysis and qualitative XAI assessment. |
| **10/2027** | Consolidation and discussion of results; preparation of reproducibility artifacts; final report writing and scientific paper submission. |

---

## 📂 Project Documentation and Attachments

Official institutional documents, regulatory records, and academic presentations are archived in the [`docs/`](./docs/) directory with English-standardized file names:

| Document | Description | Format / Link |
| :--- | :--- | :---: |
| 📄 **Research Work Plan** | Comprehensive PET individual research work plan approved by UFU | [View PDF (`docs/research_work_plan.pdf`)](./docs/research_work_plan.pdf) |
| 📝 **DIREN Research Registration** | Official UFU DIREN/PROGRAD registration record with digital Gov.br signatures | [View PDF (`docs/research_registration.pdf`)](./docs/research_registration.pdf) |
| 📊 **Academic Poster (WTDCC 2026)** | Scientific poster presented at the Workshop on Theses and Dissertations in Computer Science | [View PDF (`docs/scientific_poster_wtdcc_2026.pdf`)](./docs/scientific_poster_wtdcc_2026.pdf) |

---

## 🚀 Quick Start & Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/isadantasms/edge-meningioma-segmentation.git
cd edge-meningioma-segmentation
```

### 2. Environment Configuration
Create a virtual environment with Python 3.10+ and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Notebooks & Exploratory Pipeline
The exploratory experiments and pipeline stages are documented inside the [`notebooks/`](./notebooks/) directory:
* `01_classification_convnext_atto.ipynb`: Baseline classification with ConvNeXt-Atto ($224 \times 224$).
* `02_segmentation_segformer.ipynb`: Semantic segmentation baseline with SegFormer ($256 \times 256$).

---

## 📁 Repository Structure

```text
edge-meningioma-segmentation/
│
├── docs/                               # Official institutional and academic records
│   ├── research_work_plan.pdf         # Approved research plan (PET/UFU)
│   ├── research_registration.pdf      # Formal DIREN/UFU registration document
│   └── scientific_poster_wtdcc_2026.pdf # Academic poster (WTDCC 2026)
│
├── notebooks/                          # Experimental and exploratory Jupyter notebooks
│   ├── 01_classification_convnext_atto.ipynb # Edge classification task (ConvNeXt-Atto)
│   └── 02_segmentation_segformer.ipynb        # Semantic segmentation baseline (SegFormer)
│
├── data/                               # Dataset preparation scripts and split definitions
├── src/                                # Project source code
│   ├── data/                           # Dataset loaders and augmentations
│   ├── models/                         # Network definitions (Teacher, Student, Adapters)
│   ├── distillation/                   # Knowledge distillation loss functions & training loops
│   ├── explainability/                 # XAI implementations (LIME & activation maps)
│   └── utils/                          # Metrics (Dice, IoU, HD95) and CPU profiling tools
│
├── .gitignore                          # Git tracking exclusions
├── requirements.txt                    # Project dependencies
├── README.md                           # Main documentation (English)
├── README.pt-BR.md                     # Documentação completa (Português)
```

---

## 📚 References

1. **COLLI, B. O. et al.** Epidemiological features of meningiomas: a single Brazilian center’s experience with 993 cases. *Arquivos de Neuro-Psiquiatria*, v. 79, n. 8, 2021.
2. **FATEH, A. et al.** BRISC: annotated dataset for brain tumor segmentation and classification with Swin-HAFNet. *arXiv preprint*, arXiv:2506.14318, 2025. [arXiv:2506.14318](https://arxiv.org/abs/2506.14318).
3. **FREIRE, T.** SUS realiza seis em cada dez exames de imagem no Brasil, diz estudo. *Agência Brasil*, Report on the Atlas da Radiologia no Brasil 2025 (Colégio Brasileiro de Radiologia e Diagnóstico por Imagem), 2025.
4. **GOLDBRUNNER, R. et al.** EANO guideline on the diagnosis and management of meningiomas. *Neuro-Oncology*, v. 23, n. 11, p. 1821–1834, 2021.
5. **HINTON, G.; VINYALS, O.; DEAN, J.** Distilling the knowledge in a neural network. *arXiv preprint*, arXiv:1503.02531, 2015.
6. **ISLAM, M. S. et al.** Optimizing brain tumor classification for edge devices with knowledge distillation. *IEEE ICCIT*, 2025.
7. **LABELLA, D. et al.** A multi-institutional meningioma MRI dataset for automated multi-sequence image segmentation. *Scientific Data*, v. 11, art. 496, 2024.
8. **PINTO, L. M. et al.** Comparative analysis of mortality trends from meningioma across regions of Brazil from 2012 to 2022. *Brazilian Journal of Oncology*, v. 21, 2025.
9. **PRICE, M. et al.** CBTRUS statistical report: primary brain and other central nervous system tumors diagnosed in the United States in 2018–2022. *Neuro-Oncology*, v. 27, suppl. 4, 2025.
10. **RAHMAN, M. A. et al.** FAKD-XAI: feature-aligned knowledge distillation with explainable AI for efficient brain tumor classification. In: *ICDSAIA*, Springer, 2025.
11. **RIBEIRO, M. T.; SINGH, S.; GUESTRIN, C.** “Why should I trust you?”: explaining the predictions of any classifier. *ACM SIGKDD*, 2016.

---

## 🏛️ Institutional Support & Acknowledgments

This research is supported by:
- **PET Sistemas de Informação** — Universidade Federal de Uberlândia (UFU)
- **Faculdade de Computação (FACOM / UFU)**
- **Pró-Reitoria de Graduação (PROGRAD / UFU)**
- **Diretoria de Ensino (DIREN / UFU)**
- **Divisão de Formação Discente (DIFDI / UFU)**
