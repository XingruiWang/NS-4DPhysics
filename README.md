# NS4Dynamics

Official code for the model NS-4DPhysics: Neural-Symbolic VideoQA with Explicit 4D Scene Representation and Physical Reasoning


> **This repository contains the model part of the paper:**  
> **_Compositional 4D Dynamic Scenes Understanding with Physics Priors for Video Question Answering (ICLR 2025)_**  
> [[Paper link](https://arxiv.org/abs/2406.00622)]


[![arXiv](https://img.shields.io/badge/arXiv-2406.00622-b31b1b.svg)](https://arxiv.org/abs/2406.00622) [![Project Page](https://img.shields.io/badge/Project%20Page-DynSuperCLEVR-0a7aca?logo=globe&logoColor=white)](https://xingruiwang.github.io/projects/DynSuperCLEVR/) ![License](https://img.shields.io/github/license/XingruiWang/SuperCLEVR-Physics)


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

