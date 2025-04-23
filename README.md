# NS4Dynamics

Official code for the NS-4DPhysics model, introduced in the paper:

> **NS-4DPhysics: Neural-Symbolic VideoQA with Explicit 4D Scene Representation and Physical Reasoning**

This repository contains the implementation of the neural-symbolic model that performs question answering over dynamic scenes by integrating explicit 4D scene representations and physical priors.

[![arXiv](https://img.shields.io/badge/arXiv-2406.00622-b31b1b.svg)](https://arxiv.org/abs/2406.00622) ![License](https://img.shields.io/github/license/XingruiWang/SuperCLEVR-Physics)


---

## 🔍 Overview

NS-4DPhysics is designed to answer complex temporal and counterfactual questions about object dynamics in videos. The model leverages:
- A 3D neural mesh scene parser
- Differentiable physics simulation
- Symbolic program execution

### Pipeline

<img src="https://xingruiwang.github.io/projects/DynSuperCLEVR/static/images/model.png" width="100%"/>



## 🧠 Dataset

We use the [DynSuperCLEVR](https://github.com/XingruiWang/DynSuperCLEVR) dataset, which contains synthetic video questions focused on:
- Velocity, acceleration, and collisions
- Temporal prediction and counterfactual reasoning


## 🔧 Setup (coming soon)

Instructions for environment setup and training will be provided in the next update.


## 📄 Citation

If you use this code or dataset, please cite our work:

```bibtex
@inproceedings{wang2025ns4dphysics,
  title={NS-4DPhysics: Neural-Symbolic VideoQA with Explicit 4D Scene Representation and Physical Reasoning},
  author={Wang, Xingrui and others},
  booktitle={CVPR},
  year={2025}
}

