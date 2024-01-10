<p align="center">
  <h1 align="center">RotationDrag: Point-based Image Editing with Rotated Diffusion Features</h1>
  <p align="center">
    <strong>Minxing Luo</strong>
    &nbsp;&nbsp;
    <strong>Wentao Cheng</strong>
    &nbsp;&nbsp;
    <strong>Jian Yang</strong>
  </p>
  <!---
  <br>
  <div align="center">
    <img src="./release-doc/asset/counterfeit-1.png", width="700">
    <img src="./release-doc/asset/counterfeit-2.png", width="700">
    <img src="./release-doc/asset/majix_realistic.png", width="700">
  </div>
  <div align="center">
    <img src="./release-doc/asset/github_video.gif", width="700">
  </div>
  <p align="center">
    <a href="https://arxiv.org/abs/2306.14435"><img alt='arXiv' src="https://img.shields.io/badge/arXiv-2306.14435-b31b1b.svg"></a>
    <a href="https://yujun-shi.github.io/projects/dragdiffusion.html"><img alt='page' src="https://img.shields.io/badge/Project-Website-orange"></a>
    <a href="https://twitter.com/YujunPeiyangShi"><img alt='Twitter' src="https://img.shields.io/twitter/follow/YujunPeiyangShi?label=%40YujunPeiyangShi"></a>
  </p>
  <br>
</p>
--->

## Disclaimer
This is a research project, NOT a commercial product.

## Installation

It is recommended to run our code on a Nvidia GPU with a linux system. We have not yet tested on other configurations. Currently, it requires around 14 GB GPU memory to run our method. We will continue to optimize memory efficiency

To install the required libraries, simply run the following command:
```
conda env create -f environment.yaml
conda activate rotdrag
```

## Run RotationDrag
To start with, in command line, run the following to start the gradio user interface:
```
python3 rot_ui.py
```

Basically, it consists of the following steps:

### Case 1: Dragging Input Real Images
#### 1) train a LoRA
* Drop our input image into the left-most box.
* Input a prompt describing the image in the "prompt" field
* Click the "Train LoRA" button to train a LoRA given the input image

#### 2) do "drag" editing
* Draw a mask in the left-most box to specify the editable areas.
* Click handle and target points in the middle box. Also, you may reset all points by clicking "Undo point".
* Click the "Run" button to run our algorithm. Edited results will be displayed in the right-most box.

<!---
### Case 2: Dragging Diffusion-Generated Images
#### 1) generate an image
* Fill in the generation parameters (e.g., positive/negative prompt, parameters under Generation Config & FreeU Parameters).
* Click "Generate Image".

#### 2) do "drag" on the generated image
* Draw a mask in the left-most box to specify the editable areas
* Click handle points and target points in the middle box.
* Click the "Run" button to run our algorithm. Edited results will be displayed in the right-most box.


## Explanation for parameters in the user interface:
#### General Parameters
|Parameter|Explanation|
|-----|------|
|prompt|The prompt describing the user input image (This will be used to train the LoRA and conduct "drag" editing).|
|lora_path|The directory where the trained LoRA will be saved.|


#### Algorithm Parameters
These parameters are collapsed by default as we normally do not have to tune them. Here are the explanations:
* Base Model Config

|Parameter|Explanation|
|-----|------|
|Diffusion Model Path|The path to the diffusion models. By default we are using "runwayml/stable-diffusion-v1-5". We will add support for more models in the future.|
|VAE Choice|The Choice of VAE. Now there are two choices, one is "default", which will use the original VAE. Another choice is "stabilityai/sd-vae-ft-mse", which can improve results on images with human eyes and faces (see [explanation](https://stable-diffusion-art.com/how-to-use-vae/))|

* Drag Parameters

|Parameter|Explanation|
|-----|------|
|n_pix_step|Maximum number of steps of motion supervision. **Increase this if handle points have not been "dragged" to desired position.**|
|lam|The regularization coefficient controlling unmasked region stays unchanged. Increase this value if the unmasked region has changed more than what was desired (do not have to tune in most cases).|
|n_actual_inference_step|Number of DDIM inversion steps performed (do not have to tune in most cases).|

* LoRA Parameters

|Parameter|Explanation|
|-----|------|
|LoRA training steps|Number of LoRA training steps (do not have to tune in most cases).|
|LoRA learning rate|Learning rate of LoRA (do not have to tune in most cases)|
|LoRA rank|Rank of the LoRA (do not have to tune in most cases).|

--->
<!---
## License
Code related to the DragDiffusion algorithm is under Apache 2.0 license.


## BibTeX
If you find our repo helpful, please consider leaving a star or cite our paper :)
```bibtex
@article{shi2023dragdiffusion,
  title={DragDiffusion: Harnessing Diffusion Models for Interactive Point-based Image Editing},
  author={Shi, Yujun and Xue, Chuhui and Pan, Jiachun and Zhang, Wenqing and Tan, Vincent YF and Bai, Song},
  journal={arXiv preprint arXiv:2306.14435},
  year={2023}
}
```

## Contact
For any questions on this project, please contact [Yujun](https://yujun-shi.github.io/) (shi.yujun@u.nus.edu)

## Acknowledgement
This work is inspired by the amazing [DragGAN](https://vcai.mpi-inf.mpg.de/projects/DragGAN/). The lora training code is modified from an [example](https://github.com/huggingface/diffusers/blob/v0.17.1/examples/dreambooth/train_dreambooth_lora.py) of diffusers. Image samples are collected from [unsplash](https://unsplash.com/), [pexels](https://www.pexels.com/zh-cn/), [pixabay](https://pixabay.com/). Finally, a huge shout-out to all the amazing open source diffusion models and libraries.

## Related Links
* [Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold](https://vcai.mpi-inf.mpg.de/projects/DragGAN/)
* [MasaCtrl: Tuning-free Mutual Self-Attention Control for Consistent Image Synthesis and Editing](https://ljzycmd.github.io/projects/MasaCtrl/)
* [Emergent Correspondence from Image Diffusion](https://diffusionfeatures.github.io/)
* [DragonDiffusion: Enabling Drag-style Manipulation on Diffusion Models](https://mc-e.github.io/project/DragonDiffusion/)
* [FreeDrag: Point Tracking is Not You Need for Interactive Point-based Image Editing](https://lin-chen.site/projects/freedrag/)


## Common Issues and Solutions
1) For users struggling in loading models from huggingface due to internet constraint, please 1) follow this [links](https://zhuanlan.zhihu.com/p/475260268) and download the model into the directory "local\_pretrained\_models"; 2) Run "drag\_ui.py" and select the directory to your pretrained model in "Algorithm Parameters -> Base Model Config -> Diffusion Model Path".

--->
