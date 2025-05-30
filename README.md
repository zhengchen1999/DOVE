# DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution

[Zheng Chen](https://zhengchen1999.github.io/), [Zichen Zou](https://github.com/zzctmd), [Kewei Zhang](), [Xiongfei Su](https://ieeexplore.ieee.org/author/37086348852), [Xin Yuan](https://en.westlake.edu.cn/faculty/xin-yuan.html), [Yong Guo](https://www.guoyongcs.com/), and [Yulun Zhang](http://yulunzhang.com/), "DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution", 2025

<div>
<a href="https://github.com/zhengchen1999/DOVE/releases" target='_blank' style="text-decoration: none;"><img src="https://img.shields.io/github/downloads/zhengchen1999/DOVE/total?color=green&style=flat"></a>
<a href="https://github.com/zhengchen1999/DOVE" target='_blank' style="text-decoration: none;"><img src="https://visitor-badge.laobi.icu/badge?page_id=zhengchen1999/DOVE"></a>
<a href="https://github.com/zhengchen1999/DOVE/stargazers" target='_blank' style="text-decoration: none;"><img src="https://img.shields.io/github/stars/zhengchen1999/DOVE?style=social"></a>
</div>
[[arXiv](https://arxiv.org/abs/2505.16239)] [[supplementary material](https://github.com/zhengchen1999/DOVE/releases/download/v1/Supplementary_Material.pdf)] [dataset] [pretrained models]



#### 🔥🔥🔥 News

- **2025-5-22:** This repo is released.

---

> **Abstract:** Diffusion models have demonstrated promising performance in real-world video super-resolution (VSR). However, the dozens of sampling steps they require, make inference extremely slow. Sampling acceleration techniques, particularly single-step, provide a potential solution. Nonetheless, achieving one step in VSR remains challenging, due to the high training overhead on video data and stringent fidelity demands. To tackle the above issues, we propose DOVE, an efficient one-step diffusion model for real-world VSR. DOVE is obtained by fine-tuning a pretrained video diffusion model (*i.e.*, CogVideoX). To effectively train DOVE, we introduce the latent–pixel training strategy. The strategy employs a two-stage scheme to gradually adapt the model to the video super-resolution task.
> Meanwhile, we design a video processing pipeline to construct a high-quality dataset tailored for VSR, termed HQ-VSR. Fine-tuning on this dataset further enhances the restoration capability of DOVE. Extensive experiments show that DOVE exhibits comparable or superior performance to multi-step diffusion-based VSR methods. It also offers outstanding inference efficiency, achieving up to a **28×** speed-up over existing methods such as MGLD-VSR.

![](./assets/Compare.png)

---

<table border="0" style="width: 100%; text-align: center; margin-top: 20px;">
  <tr>
    <td>
      <video src="https://github.com/user-attachments/assets/67dbd151-a5e4-45b6-af0b-41aea4bc8c5c" controls autoplay loop></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/a95d636c-2bdd-4716-a471-ded225d76d42" controls autoplay loop></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/13b0b173-7545-40c6-9d50-2df103856203" controls autoplay loop></video>
    </td>
  </tr>
  <tr>
    <td>
      <video src="https://github.com/user-attachments/assets/e5b5d247-28af-43fd-b32c-1f1b5896d9e7" controls autoplay loop></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/d2618711-252c-4cec-862b-822f053d40ad" controls autoplay loop></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/866edcf5-4741-46d3-83f2-148ac777eeb5" controls autoplay loop></video>
    </td>
  </tr>
  <tr>
    <td>
      <video src="https://github.com/user-attachments/assets/1dce503c-95b3-421d-a5ff-575fc33f16fc" controls autoplay loop></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/26c359bf-f464-4f47-b300-36a850bffe51" controls autoplay loop></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/b75fdc77-677d-400b-ae21-6cee1cd0f66e" controls autoplay loop></video>
    </td>
  </tr>
  <tr>
    <td>
      <video src="https://github.com/user-attachments/assets/4ad0ca78-6cca-48c0-95a5-5d5554093f7d" controls autoplay loop></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/55105c46-3dc1-4fa3-95a3-fdb571e58189" controls autoplay loop></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/b338953b-4779-42a3-a087-e51e53c96141" controls autoplay loop></video>
    </td>
  </tr>
  <tr>
    <td>
      <video src="https://github.com/user-attachments/assets/1314a199-baa4-4dc1-b2e0-e54557e920a2" controls autoplay loop></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/fc14a1e0-b5d6-46a6-b6a2-1a96f7342a0e" controls autoplay loop></video>
    </td>
  </tr>
</table>




---

### Training Strategy

![](./assets/Strategy.png)

---

### Video Processing Pipeline

![](./assets/Pipeline.png)

---




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
  <img width="900" src="assets/Quantitative.png">
</p>

</details>

<details open>
<summary>Qualitative Results (click to expand)</summary>

- Results in Fig. 4 of the main paper

<p align="center">
  <img width="900" src="assets/Qualitative-1.png">
</p>
<details>
<summary>More Qualitative Results</summary>




- More results in Fig. 3 of the supplementary material

<p align="center">
  <img width="900" src="assets/Qualitative-2-1.png">
</p>



- More results in Fig. 4 of the supplementary material

<p align="center">
  <img width="900" src="assets/Qualitative-2-2.png">
</p>


- More results in Fig. 5 of the supplementary material

<p align="center">
  <img width="900" src="assets/Qualitative-3-1.png">
  <img width="900" src="assets/Qualitative-3-2.png">
</p>


- More results in Fig. 6 of the supplementary material

<p align="center">
  <img width="900" src="assets/Qualitative-4-1.png">
  <img width="900" src="assets/Qualitative-4-2.png">
</p>


- More results in Fig. 7 of the supplementary material

<p align="center">
  <img width="900" src="assets/Qualitative-5-1.png">
  <img width="900" src="assets/Qualitative-5-2.png">
</p>

</details>

</details>

## <a name="citation"></a>📎 Citation

If you find the code helpful in your research or work, please cite the following paper(s).

```
@article{chen2025dove,
  title={DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution},
  author={Chen, Zheng and Zou, Zichen and Zhang, Kewei and Su, Xiongfei and Yuan, Xin and Guo, Yong and Zhang, Yulun},
  journal={arXiv preprint arXiv:2505.16239},
  year={2025}
}
```

## <a name="acknowledgements"></a>💡 Acknowledgements

This project is based on [CogVideo](https://github.com/THUDM/CogVideo) and [Open-Sora](https://github.com/hpcaitech/Open-Sora).

