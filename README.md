<!--<img src="img\cover.png">-->

# DPM-TSE

Official Pytorch Implementation of DPM-TSE: A Diffusion Probabilistic Model for Target Sound Extraction

Paper link: [DPM-TSE](https://arxiv.org/abs/2310.04567)

Listen to examples on [Homepage](https://jhu-lcap.github.io/DPM-TSE/)

🔥 Updates: [SoloAudio](https://wanghelin1997.github.io/SoloAudio-Demo/) is now available! This advanced diffusion-transformer-based model extracts target sounds from free-text input.

## Content
- [Usage](#usage)
- [References](#references)
- [Acknowledgement](#acknowledgement)

## Usage
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

# Acknowledgement

We borrow code from following repos:

 - `Diffusion Schedulers` and `2D UNet` are based on 🤗 [Diffusers](https://github.com/huggingface/diffusers)
 - `16k HiFi-GAN vocoder` is borrowed from [AudioLDM](https://github.com/haoheliu/AudioLDM/tree/main)
