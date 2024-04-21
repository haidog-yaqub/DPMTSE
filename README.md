# DPM-TSE: A Diffusion Probabilistic Model for Target Sound Extraction

Official Pytorch Implementation of DPM-TSE

<img src="img\cover.png">

Paper link: [DPM-TSE: A Diffusion Probabilistic Model for Target Sound Extraction](https://arxiv.org/abs/2310.04567)

## Demo

🎵 Listen to [examples](https://jhu-lcap.github.io/DPM-TSE/)

## Content
- [Todo](#todo)
- [Examples](#examples)
- [References](#references)
- [Acknowledgement](#acknowledgement)

## Todo
- [x] Update code and demo
- [x] Support 🤗 [Diffusers](https://github.com/huggingface/diffusers)
- [x] Upload checkpoints
- [x] Pipeline tutorial
- [ ] Merge to [Your-Stable-Audio](https://github.com/haidog-yaqub/Your-Stable-Audio)

## Examples
- Download checkpoints and dataset from [this 🤗 link](https://huggingface.co/datasets/Higobeatz/DPM-TSE/tree/main)
- Prepare environment: [requirement.txt](requirements.txt)
``` shell
# Training
python src/train_ddim_cls.py --data-path 'data/fsd2018/' --autoencoder-path 'ckpts/first_stage.pt' --autoencoder-config 'ckpts/vae.yaml' --diffusion-config 'src/config/DiffTSE_cls_v_b_1000.yaml'
```
``` shell
# Inference
python src/tse.py --device 'cuda' --mixture 'example.wav' --target_sound 'Applause' --autoencoder-path 'ckpts/first_stage.pt' --autoencoder-config 'ckpts/vae.yaml' --diffusion-config 'src/config/DiffTSE_cls_v_b_1000.yaml' --diffusion-ckpt 'ckpts/base_v_1000.pt'
```

## References

If you find the code useful for your research, please consider citing:

```bibtex
@inproceedings{hai2024dpm,
  title={DPM-TSE: A Diffusion Probabilistic Model for Target Sound Extraction},
  author={Hai, Jiarui and Wang, Helin and Yang, Dongchao and Thakkar, Karan and Dehak, Najim and Elhilali, Mounya},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1196--1200},
  year={2024},
  organization={IEEE}
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

 - `Diffusion Schedulers` and `2D UNet` are based on 🤗 [Diffusers](https://github.com/huggingface/diffusers)
 - `16k HiFi-GAN vocoder` is borrowed from [AudioLDM](https://github.com/haoheliu/AudioLDM/tree/main)