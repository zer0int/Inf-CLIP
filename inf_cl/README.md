<p align="center">
    <img src="https://github.com/user-attachments/assets/53a09bd1-c8ac-43c0-80ae-03ba284c94ad" width="150" style="margin-bottom: 0.2;"/>
<p>

<h3 align="center"><a href="https://arxiv.org/abs/2406.07476">
Breaking the Memory Barrier: Near Infinite Batch Size Scaling for Contrastive Loss</a></h3>
<h5 align="center"> If our project helps you, please give us a star ⭐ on GitHub to support us. 🙏🙏 </h2>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2406.07476-AD1C18.svg?logo=arXiv)](https://arxiv.org/abs/2406.07476)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/DAMO-NLP-SG/Inf-CLIP/blob/main/LICENSE) 
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FDAMO-NLP-SG%2FInf-CLIP&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
[![GitHub issues](https://img.shields.io/github/issues/DAMO-NLP-SG/Inf-CLIP?color=critical&label=Issues)](https://github.com/DAMO-NLP-SG/Inf-CLIP/issues?q=is%3Aopen+is%3Aissue)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/DAMO-NLP-SG/Inf-CLIP?color=success&label=Issues)](https://github.com/DAMO-NLP-SG/Inf-CLIP/issues?q=is%3Aissue+is%3Aclosed)  <br>

</h5>

<div align="center"><img src="https://github.com/user-attachments/assets/2c19838b-43d8-4145-b28c-903f3d76f8ab" width="800" /></div>

<details open><summary>💡 Some other multimodal foundation model projects from our team may interest you ✨. </summary><p>
<!--  may -->

> [**Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding**](https://github.com/DAMO-NLP-SG/Video-LLaMA) <br>
> Hang Zhang, Xin Li, Lidong Bing <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/DAMO-NLP-SG/Video-LLaMA)  [![github](https://img.shields.io/github/stars/DAMO-NLP-SG/Video-LLaMA.svg?style=social)](https://github.com/DAMO-NLP-SG/Video-LLaMA) [![arXiv](https://img.shields.io/badge/Arxiv-2306.02858-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2306.02858) <br>

> [**VCD: Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding**](https://arxiv.org/abs/2311.16922) <br>
> Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, Lidong Bing <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/DAMO-NLP-SG/VCD)  [![github](https://img.shields.io/github/stars/DAMO-NLP-SG/VCD.svg?style=social)](https://github.com/DAMO-NLP-SG/VCD)  [![arXiv](https://img.shields.io/badge/Arxiv-2311.16922-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2311.16922) <br>

> [**VideoLLaMA 2: Advancing Spatial-Temporal Modeling and Audio Understanding in Video-LLMs**](https://github.com/DAMO-NLP-SG/VideoLLaMA2) <br>
> Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, Lidong Bing <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/DAMO-NLP-SG/VideoLLaMA2)  [![github](https://img.shields.io/github/stars/DAMO-NLP-SG/VideoLLaMA2.svg?style=social)](https://github.com/DAMO-NLP-SG/VideoLLaMA2) [![arXiv](https://img.shields.io/badge/Arxiv-2406.07476-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2406.07476) <br>

> [**The Curse of Multi-Modalities: Evaluating Hallucinations of Large Multimodal Models across Language, Visual, and Audio**](https://arxiv.org/abs/2410.12787) <br>
> Sicong Leng, Yun Xing, Zesen Cheng, Yang Zhou, Hang Zhang, Xin Li, Deli Zhao, Shijian Lu, Chunyan Miao, Lidong Bing <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/DAMO-NLP-SG/CMM)  [![github](https://img.shields.io/github/stars/DAMO-NLP-SG/CMM.svg?style=social)](https://github.com/DAMO-NLP-SG/CMM)  [![arXiv](https://img.shields.io/badge/Arxiv-2410.12787-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.12787) <br>

</p></details>

## 📰 News
* **[2024.10.18]**  Release training, evaluation codes of Inf-CLIP.

<div align="center"><img src="https://github.com/user-attachments/assets/11c5cc32-aac2-497d-bbc1-33e065a71be0" width="800" /></div>

## 🔑 Usage

A simple example about how to adopt our Inf-CL loss for contrastive learning.

```python
import torch
import torch.distributed as dist

from inf_cl import cal_inf_loss


def create_cl_tensors(rank, world_size):
    # Parameters
    dtype = torch.float32
    num_heads = 3        # Number of attention heads
    seq_length_q = 32768 # Sequence length
    seq_length_k = 32768
    d_model = 256        # Dimension of each head (must be 16, 32, 64, or 128)

    # Randomly initialize inputs
    q = torch.rand((seq_length_q // world_size, num_heads * d_model), dtype=dtype, device=f"cuda:{rank}")
    k = torch.rand((seq_length_k // world_size, num_heads * d_model), dtype=dtype, device=f"cuda:{rank}")
    l = torch.ones([], dtype=dtype, device=f"cuda:{rank}") * np.log(1 / 0.07)

    q = F.normalize(q, p=2, dim=-1).requires_grad_() # Query
    k = F.normalize(k, p=2, dim=-1).requires_grad_() # Key
    l = l.requires_grad_() # Logit scale

    return q, k, l


if __name__ == "__main__":
    # Assume that the distributed environment has been initialized
    dist.init_process_group("nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Exampled by Image-Text Contrastive Learning, q is the global image features, 
    # k is the text features, and l is the logit scale.
    q, k, l = create_cl_tensors(rank, world_size)

    # labels are diagonal elements by default. 
    # labels = torch.arange(q.shape[0])
    loss = cal_inf_loss(q, k, scale=l.exp())
```


## 📑 Citation

If you find Inf-CL useful for your research and applications, please cite using this BibTeX:
```bibtex
@article{damonlpsg2024infcl,
  title={Breaking the Memory Barrier: Near Infinite Batch Size Scaling for Contrastive Loss},
  author={Zesen Cheng, Hang Zhang, Kehan Li, Sicong Leng, Zhiqiang Hu, Fei Wu, Deli Zhao, Xin Li, Lidong Bing},
  journal={arXiv preprint arXiv:},
  year={2024},
  url = {https://arxiv.org/abs/}
}
```


## 🔒 License

This project is released under the Apache 2.0 license as found in the LICENSE file.
The service is a research preview intended for **non-commercial use ONLY**, subject to the model Licenses of CLIP, Terms of Use of the data generated by OpenAI, and Laion. Please get in touch with us if you find any potential violations.