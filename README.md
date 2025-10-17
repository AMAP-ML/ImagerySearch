# ImagerySearch: Adaptive Test-Time Search for Video Generation Beyond Semantic Dependency Constraints

<p align="center">
  <img src="./asset/logo-all2.png" width="80%"/>
</p>

<p align="center">
          🤗 <a href="https://huggingface.co/papers/2510.14847">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2503.10076">Paper</a> 
</p>

# 🔥 Updates
*   **[Coming Soon]** Paper will be released!
*   **[Coming Soon]** Code will be released!
    

# 📣 Overview

<p align="center">
  <img src="./asset/overview.png" width="100%"/>
</p>

Recent video generation models excel at producing realistic videos but struggle in imaginative scenarios, where prompts involve rarely co-occurring concepts with long-distance semantic dependencies beyond training distributions. Existing test-time scaling methods improve quality under typical prompts but rely on fixed search spaces and static rewards, limiting their adaptability to creative compositions.

We propose **ImagerySearch**, a prompt-guided adaptive test-time search strategy that dynamically adjusts both the inference search space and reward function according to semantic relationships in the prompt. This enables more coherent and visually plausible videos in challenging imaginative settings. To evaluate progress in this direction, we introduce **LDT-Bench**, the first dedicated benchmark for long-distance semantic prompts, consisting of 2,839 diverse concept pairs and an automated protocol for assessing creative generation capabilities.

Extensive experiments show that ImagerySearch consistently outperforms strong video generation baselines and existing test-time scaling approaches on LDT-Bench, and achieves competitive improvements on VBench, demonstrating its effectiveness across diverse prompt types. We will release LDT-Bench and code to facilitate future research on imaginative video generation

# 🧠 Method
As shown in the following Figure, ImagerySearch includes three components: the adaptive Imagery Reward, the constrained semantic scorer, and the semantic-aware dynamic search space. The prompt is scored by the Constrained Semantic Scorer (producing $\bar{\mathcal{D}}_{\text{sem}}$) and simultaneously fed to the T2V backbone (Wan2.1). At every step $t$ specified by the imagery scheduler, we sample a set of candidate clips, rank them with a reward function conditioned on $\bar{\mathcal{D}}_{\text{sem}}$, and retain only a $\bar{\mathcal{D}}_{\text{sem}}$-controlled subset. The loop repeats until generation completes. The figure shows the result after a single denoising step at $t = 5$.


<p align="center">
  <img src="./asset/method.png" width="100%"/>
</p>

# 📚 LDT-Bench
## Overview&Prompt Suite
We propose a novel benchmark **LDT-Bench**, designed to systematically analyze the generalization ability of video generation models in complex scenarios induced by prompts with **L**ong-**D**istance semantic **T**exts.
In the following sections, LDT-Bench is introduced from two perspectives: the construction of the prompt suite and the design of evaluation metrics.
The core components of LDT-Bench are illustrated in the following Figure.

<p align="center">
  <img src="./asset/LDT-Bench.png" width="100%"/>
</p>

More Analysis of Prompt Suite are illustrated in the following Figure.
<p align="center">
  <img src="./asset/ldt-bench2.png" width="100%"/>
</p>


## ImageryMetrics
ImageryQA is designed to evaluate the ability of video generation models to generate videos with high imagery quality.
<p align="center">
  <img src="./asset/ImageryEvolution.png" width="100%"/>
</p>


# 📊Evaluation Results

## Quantitative Results

    
### LDT-Bench Leaderboard

<div align="center">

|  Model | ElementQA | AlignQA | AnomalyQA | ImageryQA (All) |
|-------|:---------:|:-------:|:---------:|:----------------:|
| Wan2.1 | 1.66 | 31.62 | 15.00 | 48.28 |
| Video-T1 | 1.91 | **38.16** | 14.68 | 54.75 |
| Evosearch | 1.92 | 36.10 | 16.46 | 54.48 |
| **ImagerySearch (Ours)** | **2.01** | 36.82 | **18.28** | **57.11** |

Table: Quantitative comparison on LDT-Bench. ImagerySearch achieves the best average performance.

</div>

### VBench Leaderboard

<div align="center">

| Model | Aesthetic Quality | Background Consistency | Dynamic Degree | Imaging Quality | Motion Smoothness | Subject Consistency | Average |
|-------|:-----------------:|:----------------------:|:--------------:|:---------------:|:-----------------:|:-------------------:|:-------:|
| Wan2.1 | 50.50 | 91.80 | 82.85 | 58.25 | 97.50 | 90.25 | 78.53 |
| Opensora | 48.80 | 95.25 | 73.15 | 61.35 | 99.05 | 92.95 | 78.43 |
| CogvideoX | 48.80 | 95.30 | 47.20 | 65.05 | 98.55 | 94.65 | 74.93 |
| Hunyuan | 50.45 | 92.65 | 85.00 | 59.55 | 95.75 | 90.55 | 78.99 |
| Video-T1 | 57.20 | 95.65 | 54.05 | 60.25 | **99.30** | 94.80 | 76.88 |
| Evosearch | 55.55 | 94.80 | 80.95 | 68.90 | 97.70 | 94.55 | 82.08 |
| **ImagerySearch (Ours)** | **57.70** | **96.00** | **84.05** | **69.20** | 98.00 | **95.90** | **83.48** |

Table: Quantitative comparison of video generation models on VBench. **ImagerySearch** achieves the best average performance across multiple metrics, indicating better alignment and generation quality.

</div>

### NFE
<p align="center">
  <img src="./asset/nfe.png" width="100%"/>
</p>


# ❤️Acknowledgement
We would like to express our gratitude to the following open-source repositories that our work is based on: [Evosearch](https://github.com/tinnerhrhe/EvoSearch-codes), [Q-Align](https://github.com/Q-Future/Q-Align), [VideoAlign](https://github.com/KwaiVGI/VideoAlign), [Wan2.1](https://github.com/Wan-Video/Wan2.1), [Qwe2.5-VL](https://github.com/QwenLM/Qwen2.5-VL).
Their contributions have been invaluable to this project.

# 📜License
The ImagerySearch is licensed under [Apache-2.0 license](http://www.apache.org/licenses/LICENSE-2.0). You are free to use our codes for research purpose.

