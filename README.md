# InTAct + Prompting: Interval-based Consolidation for Prompt-Based Continual Learning

**Authors**: Patryk Krukowski, Jan Miksa, Piotr Helm, Jacek Tabor, Paweł Wawrzyński, Przemysław Spurek  
*GMUM — Jagiellonian University*

![Teaser](./imgs/prompt_intact.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)

> **Note:** This repository will be merged to the [repository](https://github.com/pkrukowski1/InTAct) in the future.

## Overview

This repository contains the implementation of **InTAct (Interval-based Task Activation Consolidation)** integrated into state-of-the-art **prompt-based continual learning** methods, such as [L2P](https://arxiv.org/abs/2112.08654), [DualPrompt](https://arxiv.org/pdf/2204.04799), and [CODA-Prompt](https://arxiv.org/abs/2211.13218).

While prompt-based methods help reduce catastrophic forgetting, they still experience representation drift within shared parameters, such as the classifier weights. InTAct addresses this by applying functional regularization to these shared components, allowing them to adapt to new domains without overwriting the functional behavior needed by previously learned tasks.

### Supported Methods
We provide InTAct integration for the following baselines:
* **L2P** (Learning to Prompt) + InTAct
* **DualPrompt** + InTAct
* **CODA-Prompt** + InTAct

---

## 🛠️ Getting Started

### Prerequisites

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/pkrukowski1/PromptInTAct](https://github.com/pkrukowski1/PromptInTAct)
    cd PromptInTAct
    ```

2.  **Create and activate the conda environment**
    The environment is named `prompt_intact`.
    ```bash
    conda create -n prompt_intact python=3.8
    conda activate prompt_intact
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

---

## Running Experiments

All training scripts and configurations are located in the `experiments/` folder.

1.  **Configure Hyperparameters**
    To ensure reproducibility and match the results reported in the paper, you must first examine and configure the hyperparameters used in the scripts. Before running, check the relevant bash script (e.g., `experiments/cifar100_l2p.sh`) and any configuration files.

2.  **Execute Training Scripts**
    Once hyperparameters are correctly configured, you can reproduce the results by executing the corresponding bash scripts:

    ```bash
    # Example: Run L2P integration on Split CIFAR-100 in CIL scenario
    bash experiments/cifar-100.sh

    # Example: Run DualPrompt integration on DomainNet in DIL scenario
    bash experiments/dil_domainnet.sh
    ```

---

## Acknwoledgments
- Thanks to the authors of the [CODA-Prompt](https://arxiv.org/pdf/2211.13218) paper for providing the implementation of CODA-Prompt and related methods.
- Thanks to the authors of the [KAC](https://arxiv.org/abs/2503.21076) aper for providing scripts for the Domain-Incremental Learning (DIL) scenario.