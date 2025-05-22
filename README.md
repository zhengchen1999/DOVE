
# DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution

[Zheng Chen](https://zhengchen1999.github.io/), [Zichen Zou](https://github.com/zzctmd), [Kewei Zhang](), [Xiongfei Su](https://ieeexplore.ieee.org/author/37086348852), [Xin Yuan](https://en.westlake.edu.cn/faculty/xin-yuan.html), [Yong Guo](https://www.guoyongcs.com/), and [Yulun Zhang](http://yulunzhang.com/), "DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution", 2025

<div>
<a href="https://github.com/zhengchen1999/DOVE/releases" target='_blank' style="text-decoration: none;"><img src="https://img.shields.io/github/downloads/zhengchen1999/DOVE/total?color=green&style=flat"></a>
<a href="https://github.com/zhengchen1999/DOVE" target='_blank' style="text-decoration: none;"><img src="https://visitor-badge.laobi.icu/badge?page_id=zhengchen1999/DOVE"></a>
<a href="https://github.com/zhengchen1999/DOVE/stargazers" target='_blank' style="text-decoration: none;"><img src="https://img.shields.io/github/stars/zhengchen1999/DOVE?style=social"></a>
</div>

[[project](https://zhengchen1999.github.io/DOVE)] [[arXiv]()] [[supplementary material]()] [dataset] [pretrained models]



#### 🔥🔥🔥 News

- **2025-5-22:** This repo is released.

---

> **Abstract:** Diffusion models have demonstrated promising performance in real-world video super-resolution (VSR). However, the dozens of sampling steps they require, make inference extremely slow. Sampling acceleration techniques, particularly single-step, provide a potential solution. Nonetheless, achieving one step in VSR remains challenging, due to the high training overhead on video data and stringent fidelity demands. To tackle the above issues, we propose DOVE, an efficient one-step diffusion model for real-world VSR. DOVE is obtained by fine-tuning a pretrained video diffusion model (*i.e.*, CogVideoX). To effectively train DOVE, we introduce the latent–pixel training strategy. The strategy employs a two-stage scheme to gradually adapt the model to the video super-resolution task.
> Meanwhile, we design a video processing pipeline to construct a high-quality dataset tailored for VSR, termed HQ-VSR. Fine-tuning on this dataset further enhances the restoration capability of DOVE. Extensive experiments show that DOVE exhibits comparable or superior performance to multi-step diffusion-based VSR methods. It also offers outstanding inference efficiency, achieving up to a **28$\times$** speed-up over existing methods such as MGLD-VSR.

![](./figs/Compare.png)

---

### Training Strategy

![](./figs/Strategy.png)

---

### Video Processing Pipeline

![](./figs/Pipeline.png)

## 🔖 TODO

- [ ] Release testing and training code.
- [ ] Release pre-trained models.
- [ ] Release HQ-VSR dataset.
- [ ] Provide WebUI.
- [ ] Provide HuggingFace demo.

## 🔗 Contents

1. Models
1. Training
1. Testing
1. [Results](#results)
1. [Acknowledgements](#acknowledgements)

## <a name="results"></a>🔎 Results

We achieve state-of-the-art performance on real-world video super-resolution.

<details open>
<summary>Quantitative Results (click to expand)</summary>

- Results in Tab. 2 of the main paper

<p align="center">
  <img width="900" src="figs/Quantitative.png">
</p>
</details>

<details open>
<summary>Qualitative Results (click to expand)</summary>

- Results in Fig. 4 of the main paper

<p align="center">
  <img width="900" src="figs/Qualitative-1.png">
</p>
<details>
<summary>More Qualitative Results</summary>


- More results in Fig. 1 of the supplementary material

<p align="center">
  <img width="900" src="figs/Qualitative-2.png">
</p>

- More results in Fig. 2 of the supplementary material

<p align="center">
  <img width="900" src="figs/Qualitative-3.png">
</p>
- More results in Fig. 3 of the supplementary material

<p align="center">
  <img width="900" src="figs/Qualitative-4.png">
</p>

</details>

</details>

## <a name="acknowledgements"></a>💡 Acknowledgements

This project is based on [CogVideo](https://github.com/THUDM/CogVideo) and [Open-Sora](https://github.com/hpcaitech/Open-Sora).

