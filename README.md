# DPM-TSE: A Diffusion Probabilistic Model For Target Sound Extraction

Official Pytorch Implementation of [DPM-TSE: A Diffusion Probabilistic Model For Target Sound Extraction](https://arxiv.org/abs/2310.04567)

--------------------

<img src="img\dpmtse.jpg" width="300px">

DPM-TSE  (💻WIP)

- [Demo](##demo)
- [Todo](##todo)
- [References](##references)
- [Acknowledgement](##acknowledgement)

## Demo

🎵 Listen to [examples](https://jhu-lcap.github.io/DPM-TSE/)

## Todo
- [x] Update codes and demo
- [x] Support 🤗 [Diffusers](https://github.com/huggingface/diffusers)
- [ ] Upload checkpoints
- [ ] Pipeline tutorial
- [ ] Merge to [Your-Stable-Audio](https://github.com/haidog-yaqub/Your-Stable-Audio)
## References
If you find the code useful for your research, please consider citing:

```bibtex
@article{hai2023dpm,
  title={DPM-TSE: A Diffusion Probabilistic Model for Target Sound Extraction},
  author={Hai, Jiarui and Wang, Helin and Yang, Dongchao and Thakkar, Karan and Dehak, Najim and Elhilali, Mounya},
  journal={arXiv preprint arXiv:2310.04567},
  year={2023}
}
```

This repo is inspired by:

```bibtex
@article{popov2021diffusion,
  title={Diffusion-based voice conversion with fast maximum likelihood sampling scheme},
  author={Popov, Vadim and Vovk, Ivan and Gogoryan, Vladimir and Sadekova, Tasnima and Kudinov, Mikhail and Wei, Jiansheng},
  journal={arXiv preprint arXiv:2109.13821},
  year={2021}
}
```
```bibtex
@article{lin2023common,
  title={Common Diffusion Noise Schedules and Sample Steps are Flawed},
  author={Lin, Shanchuan and Liu, Bingchen and Li, Jiashi and Yang, Xiao},
  journal={arXiv preprint arXiv:2305.08891},
  year={2023}
}
```

# Acknowledgement

We borrow code from following repos:

 - `Diffusion Schedulers` are based on 🤗 [Diffusers](https://github.com/huggingface/diffusers)
 - `2D UNet` is based on 🤗 [Diffusers]