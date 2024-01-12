import os
import cv2
import numpy as np
import gradio as gr
from copy import deepcopy
from einops import rearrange
from types import SimpleNamespace

import datetime
import PIL
from PIL import Image
from PIL.ImageOps import exif_transpose
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from diffusers import DDIMScheduler, AutoencoderKL, DPMSolverMultistepScheduler
from drag_pipeline import DragPipeline

from torchvision.utils import save_image
from pytorch_lightning import seed_everything
from loguru import logger
import json

from .logger import get_logger
from .drag_utils import (
    drag_diffusion_update,
    drag_diffusion_update_gen,
    free_drag_update,
    drag_diffusion_update_r,
    get_rot_axis,
)
from .lora_utils import train_lora
from .attn_utils import (
    register_attention_editor_diffusers,
    MutualSelfAttentionControl,
    register_attention_editor_diffusers_ori,
    unregister_attention_editor_diffusers,
)
from .draw_utils import (
    draw_handle_target_points,
    draw_featuremap,
    draw_handle_target_points_r,
)


# -------------- general UI functionality --------------
def clear_all(length=480):
    return (
        gr.Image.update(value=None, height=length, width=length),
        gr.Image.update(value=None, height=length, width=length),
        gr.Image.update(value=None, height=length, width=length),
        [],
        None,
        None,
    )


def clear_all_free(length=480):
    return (
        gr.Image.update(value=None, height=length, width=length),
        gr.Image.update(value=None, height=length, width=length),
        gr.Image.update(value=None, height=length, width=length),
        [],
        None,
        None,
    )


def mask_image(image, mask, color=[255, 0, 0], alpha=0.5):
    """Overlay mask on image for visualization purpose.
    Args:
        image (H, W, 3) or (H, W): input image
        mask (H, W): mask to be overlaid
        color: the color of overlaid mask
        alpha: the transparency of the mask
    """
    out = deepcopy(image)
    img = deepcopy(image)
    img[mask == 1] = color
    out = cv2.addWeighted(img, alpha, out, 1 - alpha, 0, out)
    return out


def store_img(img, length=512):
    image, mask = img["image"], np.float32(img["mask"][:, :, 0]) / 255.0
    height, width, _ = image.shape
    image = Image.fromarray(image)
    image = exif_transpose(image)
    image = image.resize((length, int(length * height / width)), PIL.Image.BILINEAR)
    mask = cv2.resize(
        mask, (length, int(length * height / width)), interpolation=cv2.INTER_NEAREST
    )
    image = np.array(image)

    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        masked_img = mask_image(image, 1 - mask, color=[0, 0, 0], alpha=0.3)
    else:
        masked_img = image.copy()
    # when new image is uploaded, `selected_points` should be empty
    return image, [], masked_img, mask


def mask_from_pic(mask_path, img, length=512):
    image = img["image"]
    height, width, _ = image.shape
    image = Image.fromarray(image)
    image = exif_transpose(image)
    image = image.resize((length, int(length * height / width)), PIL.Image.BILINEAR)
    image = np.array(image)

    mask_pic = cv2.imread(mask_path.name)
    mask = np.float32(mask_pic[:, :, 0]) / 255.0
    mask = cv2.resize(
        mask, (length, int(length * height / width)), interpolation=cv2.INTER_NEAREST
    )
    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        masked_img = mask_image(image, 1 - mask, color=[0, 0, 0], alpha=0.3)
    else:
        masked_img = image.copy()
    # when new image is uploaded, `selected_points` should be empty
    return image, [], masked_img, mask


def store_img_free(img, length=512):
    image, mask = img["image"], np.float32(img["mask"][:, :, 0]) / 255.0
    height, width, _ = image.shape
    image = Image.fromarray(image)
    image = exif_transpose(image)
    image = image.resize((length, int(length * height / width)), PIL.Image.BILINEAR)
    mask = cv2.resize(
        mask, (length, int(length * height / width)), interpolation=cv2.INTER_NEAREST
    )
    image = np.array(image)

    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        masked_img = mask_image(image, 1 - mask, color=[0, 0, 0], alpha=0.3)
    else:
        masked_img = image.copy()
    # when new image is uploaded, `selected_points` should be empty
    return image, [], masked_img, mask


# user click the image to get points, and show the points on the image
def get_points(img, sel_pix, evt: gr.SelectData):
    # collect the selected point
    sel_pix.append(evt.index)
    # draw points
    points = []
    for idx, point in enumerate(sel_pix):
        # print(point)
        if idx % 2 == 0:
            # draw a red circle at the handle point
            cv2.circle(img, tuple(point), 10, (255, 0, 0), -1)
        else:
            # draw a blue circle at the handle point
            cv2.circle(img, tuple(point), 10, (0, 0, 255), -1)
        points.append(tuple(point))
        # draw an arrow from handle point to target point
        if len(points) == 2:
            cv2.arrowedLine(
                img, points[0], points[1], (255, 255, 255), 4, tipLength=0.5
            )
            points = []
    return img if isinstance(img, np.ndarray) else np.array(img)


# locate point by coordinate
def locate_pt(x, y, img, sel_pix):
    sel_pix.append([x, y])
    points = []
    for idx, point in enumerate(sel_pix):
        # print(point)
        if idx % 2 == 0:
            # draw a red circle at the handle point
            cv2.circle(img, tuple(point), 10, (255, 0, 0), -1)
        else:
            # draw a blue circle at the handle point
            cv2.circle(img, tuple(point), 10, (0, 0, 255), -1)
        points.append(tuple(point))
        # draw an arrow from handle point to target point
        if len(points) == 2:
            cv2.arrowedLine(
                img, points[0], points[1], (255, 255, 255), 4, tipLength=0.5
            )
            points = []
    return img if isinstance(img, np.ndarray) else np.array(img), sel_pix


def load_config(config_path, img, sel_pix):
    with open(config_path.name, "r") as f:
        config = json.load(f)
        prompt = config["prompt"]
        points = config["point"]
        pix_step = config["pix_step"]
        sel_pix = points
        points = []
        for idx, point in enumerate(sel_pix):
            # print(point)
            if idx % 2 == 0:
                # draw a red circle at the handle point
                cv2.circle(img, tuple(point), 10, (255, 0, 0), -1)
            else:
                # draw a blue circle at the handle point
                cv2.circle(img, tuple(point), 10, (0, 0, 255), -1)
            points.append(tuple(point))
            # draw an arrow from handle point to target point
            if len(points) == 2:
                cv2.arrowedLine(
                    img, points[0], points[1], (255, 255, 255), 4, tipLength=0.5
                )
                points = []
    return (
        img if isinstance(img, np.ndarray) else np.array(img),
        sel_pix,
        prompt,
        pix_step,
    )


# clear all handle/target points
def undo_points(original_image, mask):
    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        masked_img = mask_image(original_image, 1 - mask, color=[0, 0, 0], alpha=0.3)
    else:
        masked_img = original_image.copy()
    return masked_img, []


# ------------------------------------------------------


# ----------- dragging user-input image utils -----------
def train_lora_interface(
    original_image,
    prompt,
    model_path,
    vae_path,
    lora_path,
    lora_step,
    lora_lr,
    lora_batch_size,
    lora_rank,
    progress=gr.Progress(),
):
    train_lora(
        original_image,
        prompt,
        model_path,
        vae_path,
        lora_path,
        lora_step,
        lora_lr,
        lora_batch_size,
        lora_rank,
        progress,
    )
    return "Training LoRA Done!"


def preprocess_image(image, device):
    image = torch.from_numpy(image).float() / 127.5 - 1  # [-1, 1]
    image = rearrange(image, "h w c -> 1 c h w")
    image = image.to(device)
    return image


def run_drag(
    source_image,
    image_with_clicks,
    mask,
    prompt,
    points,
    inversion_strength,
    lam,
    latent_lr,
    n_pix_step,
    model_path,
    vae_path,
    lora_path,
    start_step,
    start_layer,
    save_dir="./results",
    unet_feature_idx=[3],
    sample_interval=10,
):
    # initialize model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    model = DragPipeline.from_pretrained(model_path, scheduler=scheduler).to(device)
    # call this function to override unet forward function,
    # so that intermediate features are returned after forward
    model.modify_unet_forward()

    # set vae
    if vae_path != "default":
        model.vae = AutoencoderKL.from_pretrained(vae_path).to(
            model.vae.device, model.vae.dtype
        )

    # initialize parameters
    seed = 42  # random seed used by a lot of people for unknown reason
    seed_everything(seed)

    args = SimpleNamespace()
    args.prompt = prompt
    args.points = points
    args.n_inference_step = 50
    args.n_actual_inference_step = round(inversion_strength * args.n_inference_step)
    args.guidance_scale = 1.0
    unet_feature_idx.sort()
    args.unet_feature_idx = unet_feature_idx

    args.r_m = 1
    args.r_p = 3
    args.lam = lam

    args.lr = latent_lr
    args.n_pix_step = n_pix_step

    full_h, full_w = source_image.shape[:2]
    args.sup_res_h = int(0.5 * full_h)
    args.sup_res_w = int(0.5 * full_w)
    args.sample_interval = sample_interval
    args.prompt = prompt

    # print(args)
    save_prefix = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    if prompt != "":
        save_dir = os.path.join(save_dir, prompt.replace(" ", "_"))
    else:
        save_dir = os.path.join(save_dir, "None")
    save_dir = os.path.join(save_dir, save_prefix)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    logger = get_logger(save_dir + "/result.log")
    logger.info(args)

    source_image = preprocess_image(source_image, device)
    image_with_clicks = preprocess_image(image_with_clicks, device)

    # set lora
    if lora_path == "":
        print("applying default parameters")
        model.unet.set_default_attn_processor()
    else:
        print("applying lora: " + lora_path)
        model.unet.load_attn_procs(lora_path)

    # invert the source image
    # the latent code resolution is too small, only 64*64
    invert_code = model.invert(
        source_image,
        prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.n_inference_step,
        num_actual_inference_steps=args.n_actual_inference_step,
    )

    mask = torch.from_numpy(mask).float() / 255.0
    mask[mask > 0.0] = 1.0
    mask = rearrange(mask, "h w -> 1 1 h w").cuda()
    mask = F.interpolate(mask, (args.sup_res_h, args.sup_res_w), mode="nearest")

    handle_points = []
    target_points = []
    # here, the point is in x,y coordinate
    for idx, point in enumerate(points):
        cur_point = torch.tensor(
            [point[1] / full_h * args.sup_res_h, point[0] / full_w * args.sup_res_w]
        )
        cur_point = torch.round(cur_point)
        if idx % 2 == 0:
            handle_points.append(cur_point)
        else:
            target_points.append(cur_point)
    print("handle points:", handle_points)  # y,x (h,w)
    print("target points:", target_points)  # y,x (h,w)

    init_code = invert_code
    init_code_orig = deepcopy(init_code)
    model.scheduler.set_timesteps(args.n_inference_step)
    t = model.scheduler.timesteps[args.n_inference_step - args.n_actual_inference_step]

    # feature shape: [1280,16,16], [1280,32,32], [640,64,64], [320,64,64]
    # update according to the given supervision
    for updated_init_code, current_points, ft in drag_diffusion_update(
        model, init_code, t, handle_points, target_points, mask, args
    ):
        # hijack the attention module
        # inject the reference branch to guide the generation
        editor = MutualSelfAttentionControl(
            start_step=start_step,
            start_layer=start_layer,
            total_steps=args.n_inference_step,
            guidance_scale=args.guidance_scale,
        )
        if lora_path == "":
            ori_forward = register_attention_editor_diffusers_ori(
                model, editor, attn_processor="attn_proc"
            )
        else:
            ori_forward = register_attention_editor_diffusers_ori(
                model, editor, attn_processor="lora_attn_proc"
            )

        # inference the synthesized image
        gen_image = model(
            prompt=args.prompt,
            batch_size=2,
            latents=torch.cat([init_code_orig, updated_init_code], dim=0),
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.n_inference_step,
            num_actual_inference_steps=args.n_actual_inference_step,
        )[1].unsqueeze(dim=0)

        if lora_path == "":
            unregister_attention_editor_diffusers(
                model, ori_forward, attn_processor="attn_proc"
            )
        else:
            unregister_attention_editor_diffusers(
                model, ori_forward, attn_processor="lora_attn_proc"
            )

        # resize gen_image into the size of source_image
        # we do this because shape of gen_image will be rounded to multipliers of 8
        gen_image = F.interpolate(gen_image, (full_h, full_w), mode="bilinear")

        # save the original image, user editing instructions, synthesized image
        save_result = torch.cat(
            [
                source_image * 0.5 + 0.5,
                torch.ones((1, 3, full_h, 25)).cuda(),
                image_with_clicks * 0.5 + 0.5,
                torch.ones((1, 3, full_h, 25)).cuda(),
                gen_image[0:1],
            ],
            dim=-1,
        )
        # print(save_dir)
        save_dir_3_col = save_dir + "/3_col"
        if not os.path.isdir(save_dir_3_col):
            os.makedirs(save_dir_3_col)
        save_prefix = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
        save_image(save_result, os.path.join(save_dir_3_col, save_prefix + ".png"))

        out_image = gen_image.cpu().permute(0, 2, 3, 1).numpy()[0]
        out_image = (out_image * 255).astype(np.uint8)
        out_image = gen_image.cpu().permute(0, 2, 3, 1).numpy()[0]
        out_image = (out_image * 255).astype(np.uint8)
        # total_image.appnend(out_image)
        draw_handle_points = []
        draw_target_points = []
        for idx, point in enumerate(current_points):
            draw_cur_point = torch.tensor(
                [point[0] / args.sup_res_h * full_h, point[1] / args.sup_res_w * full_w]
            ).int()
            draw_handle_points.append(draw_cur_point)
        for idx, point in enumerate(target_points):
            draw_tar_point = torch.tensor(
                [point[0] / args.sup_res_h * full_h, point[1] / args.sup_res_w * full_w]
            ).int()
            draw_target_points.append(draw_tar_point)
        out_image = draw_handle_target_points(
            out_image, draw_handle_points, draw_target_points
        )
        logger.info(f"handle Points: {draw_handle_points}")
        logger.info(f"Target Points: {draw_target_points}")
        save_pts = PIL.Image.fromarray(out_image)
        save_dir_points = save_dir + "/points"
        if not os.path.isdir(save_dir_points):
            os.makedirs(save_dir_points)
        save_pts.save(os.path.join(save_dir_points, save_prefix + "_points.png"))
        fig_ft = draw_featuremap(ft)
        save_dir_ft = save_dir + "/ft"
        if not os.path.isdir(save_dir_ft):
            os.makedirs(save_dir_ft)
        fig_ft.savefig(
            os.path.join(save_dir_ft, save_prefix + "_ft.png"), bbox_inches="tight"
        )
        plt.close(fig_ft)
        drawable_init_code = updated_init_code.clone().cpu().detach()
        fig_latent = draw_featuremap(drawable_init_code)
        save_dir_lat = save_dir + "/latent"
        if not os.path.isdir(save_dir_lat):
            os.makedirs(save_dir_lat)
        fig_latent.savefig(
            os.path.join(save_dir_lat, save_prefix + "_lat.png"), bbox_inches="tight"
        )
        plt.close(fig_latent)
        yield out_image


# -------------------------------------------------------


def run_drag_r(
    source_image,
    image_with_clicks,
    mask,
    prompt,
    points,
    inversion_strength,
    lam,
    latent_lr,
    n_pix_step,
    model_path,
    vae_path,
    lora_path,
    start_step,
    start_layer,
    save_dir="./results",
    unet_feature_idx=[3],
    sample_interval=10,
    use_lora=True,
    lora_step=60,
    lora_lr=0.0005,
    lora_batch_size=4,
    lora_rank=16,
):
    if use_lora:
        train_lora(
            source_image,
            prompt,
            model_path,
            vae_path,
            lora_path,
            lora_step,
            lora_lr,
            lora_batch_size,
            lora_rank,
        )
    # initialize model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    model = DragPipeline.from_pretrained(model_path, scheduler=scheduler).to(device)
    # call this function to override unet forward function,
    # so that intermediate features are returned after forward
    model.modify_unet_forward()

    # set vae
    if vae_path != "default":
        model.vae = AutoencoderKL.from_pretrained(vae_path).to(
            model.vae.device, model.vae.dtype
        )

    # initialize parameters
    seed = 42  # random seed used by a lot of people for unknown reason
    seed_everything(seed)

    args = SimpleNamespace()
    args.prompt = prompt
    args.points = points
    args.n_inference_step = 50
    args.n_actual_inference_step = round(inversion_strength * args.n_inference_step)
    args.guidance_scale = 1.0
    unet_feature_idx.sort()
    args.unet_feature_idx = unet_feature_idx

    args.r_m = 1
    args.r_p = 3
    args.lam = lam

    args.lr = latent_lr
    args.n_pix_step = n_pix_step

    full_h, full_w = source_image.shape[:2]
    args.sup_res_h = int(0.5 * full_h)
    args.sup_res_w = int(0.5 * full_w)
    args.sample_interval = sample_interval
    args.prompt = prompt
    # args to be added in rotation
    # %-------------------------------------------------------%
    args.res_ratio = 0.5
    args.device = device

    # print(args)
    save_prefix = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    if prompt != "":
        save_dir = os.path.join(save_dir, prompt.replace(" ", "_"))
    else:
        save_dir = os.path.join(save_dir, "None")
    save_dir = os.path.join(save_dir, save_prefix + "_ori")

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    logger = get_logger(save_dir + "/result.log")
    logger.info(args)
    with open(save_dir + "/config.json", "w") as f:
        json.dump(
            {
                "point": points,
                "prompt": prompt,
                "pix_step": n_pix_step,
            },
            f,
        )
    source_image = preprocess_image(source_image, device)
    args.source_image = source_image.clone().detach()
    image_with_clicks = preprocess_image(image_with_clicks, device)

    # set lora
    if lora_path == "":
        logger.info("applying default parameters")
        model.unet.set_default_attn_processor()
    else:
        logger.info("applying lora: " + lora_path)
        model.unet.load_attn_procs(lora_path)

    # invert the source image
    # the latent code resolution is too small, only 64*64
    invert_code = model.invert(
        source_image,
        prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.n_inference_step,
        num_actual_inference_steps=args.n_actual_inference_step,
    )

    handle_points = []
    target_points = []
    axis = []
    # here, the point is in x,y coordinate
    for idx, point in enumerate(points):
        cur_point = torch.tensor(
            [point[1] / full_h * args.sup_res_h, point[0] / full_w * args.sup_res_w]
        )
        cur_point = torch.round(cur_point)
        if idx % 2 == 0:
            handle_points.append(cur_point)
        else:
            if (cur_point == handle_points[-1]).all():
                axis.append(cur_point)
                handle_points.pop()
            else:
                target_points.append(cur_point)
    logger.info(f"handle points:, {handle_points}")  # y,x (h,w)
    logger.info(f"target points:, {target_points}")  # y,x (h,w)
    save_mask = mask * 255
    # if mask.sum() == 0:
    #     save_mask = np.full((mask.shape[0], mask.shape[1]), 255, dtype=np.uint8)
    saved_mask = Image.fromarray(save_mask, mode="L")
    saved_mask.save(os.path.join(save_dir, "mask.png"))
    mask = torch.from_numpy(mask).float() / 255.0
    mask[mask > 0.0] = 1.0
    mask = rearrange(mask, "h w -> 1 1 h w").cuda()
    mask = F.interpolate(mask, (args.sup_res_h, args.sup_res_w), mode="nearest")
    # get rotate axis
    axis = get_rot_axis(handle_points, target_points, mask, axis, args)
    logger.info(f"rotation axis: {axis}")

    init_code = invert_code
    init_code_orig = deepcopy(init_code)
    model.scheduler.set_timesteps(args.n_inference_step)
    t = model.scheduler.timesteps[args.n_inference_step - args.n_actual_inference_step]

    # feature shape: [1280,16,16], [1280,32,32], [640,64,64], [320,64,64]
    # update according to the given supervision
    for updated_init_code, current_points, ft in drag_diffusion_update_r(
        model, init_code, t, handle_points, target_points, axis, mask, args
    ):
        # hijack the attention module
        # inject the reference branch to guide the generation
        editor = MutualSelfAttentionControl(
            start_step=start_step,
            start_layer=start_layer,
            total_steps=args.n_inference_step,
            guidance_scale=args.guidance_scale,
        )
        if lora_path == "":
            ori_forward = register_attention_editor_diffusers_ori(
                model, editor, attn_processor="attn_proc"
            )
        else:
            ori_forward = register_attention_editor_diffusers_ori(
                model, editor, attn_processor="lora_attn_proc"
            )

        # inference the synthesized image
        gen_image = model(
            prompt=args.prompt,
            batch_size=2,
            latents=torch.cat([init_code_orig, updated_init_code], dim=0),
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.n_inference_step,
            num_actual_inference_steps=args.n_actual_inference_step,
        )[1].unsqueeze(dim=0)

        if lora_path == "":
            unregister_attention_editor_diffusers(
                model, ori_forward, attn_processor="attn_proc"
            )
        else:
            unregister_attention_editor_diffusers(
                model, ori_forward, attn_processor="lora_attn_proc"
            )

        # resize gen_image into the size of source_image
        # we do this because shape of gen_image will be rounded to multipliers of 8
        gen_image = F.interpolate(gen_image, (full_h, full_w), mode="bilinear")

        # save the original image, user editing instructions, synthesized image
        save_result = torch.cat(
            [
                source_image * 0.5 + 0.5,
                torch.ones((1, 3, full_h, 25)).cuda(),
                image_with_clicks * 0.5 + 0.5,
                torch.ones((1, 3, full_h, 25)).cuda(),
                gen_image[0:1],
            ],
            dim=-1,
        )
        # print(save_dir)
        save_dir_3_col = save_dir + "/3_col"
        if not os.path.isdir(save_dir_3_col):
            os.makedirs(save_dir_3_col)
        save_prefix = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
        save_image(save_result, os.path.join(save_dir_3_col, save_prefix + ".png"))

        out_image = gen_image.cpu().permute(0, 2, 3, 1).numpy()[0]
        out_image = (out_image * 255).astype(np.uint8)
        out_image = gen_image.cpu().permute(0, 2, 3, 1).numpy()[0]
        out_image = (out_image * 255).astype(np.uint8)
        # total_image.appnend(out_image)
        draw_handle_points = []
        draw_target_points = []
        draw_ax_points = []
        for idx, point in enumerate(current_points):
            draw_cur_point = torch.tensor(
                [point[0] / args.sup_res_h * full_h, point[1] / args.sup_res_w * full_w]
            ).int()
            draw_handle_points.append(draw_cur_point)
        for idx, point in enumerate(target_points):
            draw_tar_point = torch.tensor(
                [point[0] / args.sup_res_h * full_h, point[1] / args.sup_res_w * full_w]
            ).int()
            draw_target_points.append(draw_tar_point)
        for idx, point in enumerate(axis):
            draw_ax_point = torch.tensor(
                [point[0] / args.sup_res_h * full_h, point[1] / args.sup_res_w * full_w]
            ).int()
            draw_ax_points.append(draw_ax_point)
        save_out = Image.fromarray(out_image)
        save_out.save(os.path.join(save_dir, save_prefix + ".png"))
        out_image = draw_handle_target_points_r(
            out_image, draw_handle_points, draw_target_points, draw_ax_points
        )
        logger.info(f"handle Points: {draw_handle_points}")
        logger.info(f"Target Points: {draw_target_points}")
        save_pts = PIL.Image.fromarray(out_image)
        save_dir_points = save_dir + "/points"
        if not os.path.isdir(save_dir_points):
            os.makedirs(save_dir_points)
        save_pts.save(os.path.join(save_dir_points, save_prefix + "_points.png"))
        # fig_ft = draw_featuremap(ft)
        # save_dir_ft = save_dir + "/ft"
        # if not os.path.isdir(save_dir_ft):
        #     os.makedirs(save_dir_ft)
        # fig_ft.savefig(
        #     os.path.join(save_dir_ft, save_prefix + "_ft.png"), bbox_inches="tight"
        # )
        # plt.close(fig_ft)
        # drawable_init_code = updated_init_code.clone().cpu().detach()
        # fig_latent = draw_featuremap(drawable_init_code)
        # save_dir_lat = save_dir + "/latent"
        # if not os.path.isdir(save_dir_lat):
        #     os.makedirs(save_dir_lat)
        # fig_latent.savefig(
        #     os.path.join(save_dir_lat, save_prefix + "_lat.png"), bbox_inches="tight"
        # )
        # plt.close(fig_latent)
        yield out_image


# -------------------------------------------------------


def run_freedrag(
    source_image,
    image_with_clicks,
    mask,
    prompt,
    points,
    inversion_strength,
    lam,
    latent_lr,
    n_pix_step,
    model_path,
    vae_path,
    lora_path,
    start_step,
    start_layer,
    l_expected,
    d_max,
    sample_interval,
    save_dir="./results",
    resolution=512,
    unet_feature_idx=[3],
    use_lora=True,
    lora_step=60,
    lora_lr=0.0005,
    lora_batch_size=4,
    lora_rank=16,
):
    if use_lora:
        train_lora(
            source_image,
            prompt,
            model_path,
            vae_path,
            lora_path,
            lora_step,
            lora_lr,
            lora_batch_size,
            lora_rank,
        )
    # initialize model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    model = DragPipeline.from_pretrained(model_path, scheduler=scheduler).to(device)
    # call this function to override unet forward function,
    # so that intermediate features are returned after forward
    model.modify_unet_forward()

    # set vae
    if vae_path != "default":
        model.vae = AutoencoderKL.from_pretrained(vae_path).to(
            model.vae.device, model.vae.dtype
        )

    # initialize parameters
    seed = 42  # random seed used by a lot of people for unknown reason
    seed_everything(seed)

    args = SimpleNamespace()
    args.lora_path = lora_path
    args.prompt = prompt
    args.points = points
    args.n_inference_step = 50
    args.n_actual_inference_step = round(inversion_strength * args.n_inference_step)
    args.guidance_scale = 1.0
    unet_feature_idx.sort()
    args.unet_feature_idx = unet_feature_idx

    args.r_m = 1
    args.r_p = 3
    args.lam = lam

    args.lr = latent_lr
    args.n_pix_step = n_pix_step

    full_h, full_w = source_image.shape[:2]
    args.sup_res_h = int(0.5 * full_h)
    args.sup_res_w = int(0.5 * full_w)

    # freedrag added
    # %---------------------%
    args.device = device
    args.res_ratio = 0.5
    args.l_expected = l_expected
    args.dmax = d_max
    args.sample_interval = sample_interval
    args.threshold_l = 0.5 * l_expected
    args.aa = torch.log(torch.tensor(9.0, device=device)) / (0.6 * l_expected)
    args.bb = 0.2 * l_expected
    # %---------------------%

    save_dir = os.path.join(
        save_dir,
        prompt.replace(" ", "_") + "_" + str(l_expected) + "_" + str(latent_lr),
    )
    save_prefix = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    save_dir = os.path.join(save_dir, save_prefix)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    logger = get_logger(save_dir + "/result.log")
    logger.info("Using model " + model_path)
    logger.info(args)

    source_image = preprocess_image(source_image, device)
    image_with_clicks = preprocess_image(image_with_clicks, device)

    # set lora
    if lora_path == "":
        logger.info("applying default parameters")
        model.unet.set_default_attn_processor()
    else:
        logger.info("applying lora: " + lora_path)
        model.unet.load_attn_procs(lora_path)

    # invert the source image
    # the latent code resolution is too small, only 64*64
    invert_code = model.invert(
        source_image,
        prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.n_inference_step,
        num_actual_inference_steps=args.n_actual_inference_step,
    )

    mask = torch.from_numpy(mask).float() / 255.0
    mask[mask > 0.0] = 1.0
    mask = rearrange(mask, "h w -> 1 1 h w").cuda()
    mask = F.interpolate(mask, (args.sup_res_h, args.sup_res_w), mode="nearest")

    handle_points = []
    target_points = []
    # here, the point is in x,y coordinate
    for idx, point in enumerate(points):
        cur_point = torch.tensor(
            [point[1] / full_h * args.sup_res_h, point[0] / full_w * args.sup_res_w],
            device=device,
        ).float()
        cur_point = torch.round(cur_point)
        if idx % 2 == 0:
            handle_points.append(cur_point)
        else:
            target_points.append(cur_point)
    handle_points = torch.stack(handle_points)
    target_points = torch.stack(target_points)
    logger.info(f"handle points: {handle_points}")  # y,x (h,w)
    logger.info(f"target points: {target_points}")  # y,x (h,w)

    init_code = invert_code
    init_code_orig = deepcopy(init_code)
    model.scheduler.set_timesteps(args.n_inference_step)
    t = model.scheduler.timesteps[args.n_inference_step - args.n_actual_inference_step]

    # feature shape: [1280,16,16], [1280,32,32], [640,64,64], [320,64,64]
    # update according to the given supervision
    # total_image = []
    global stop_flag
    stop_flag = False
    for updated_init_code, current_points, ft in free_drag_update(
        model, init_code, t, handle_points, target_points, mask, args
    ):
        # hijack the attention module
        # inject the reference branch to guide the generation

        # TODO: find a way to unregister the Masctrl or just remove it
        editor = MutualSelfAttentionControl(
            start_step=start_step,
            start_layer=start_layer,
            total_steps=args.n_inference_step,
            guidance_scale=args.guidance_scale,
        )
        if lora_path == "":
            ori_forward = register_attention_editor_diffusers_ori(
                model, editor, attn_processor="attn_proc"
            )
        else:
            ori_forward = register_attention_editor_diffusers_ori(
                model, editor, attn_processor="lora_attn_proc"
            )

        # inference the synthesized image
        gen_image = model(
            prompt=args.prompt,
            batch_size=2,
            height=resolution,
            width=resolution,
            latents=torch.cat([init_code_orig, updated_init_code], dim=0),
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.n_inference_step,
            num_actual_inference_steps=args.n_actual_inference_step,
        )[1].unsqueeze(dim=0)

        # resize gen_image into the size of source_image
        # we do this because shape of gen_image will be rounded to multipliers of 8
        gen_image = F.interpolate(gen_image, (full_h, full_w), mode="bilinear")

        if lora_path == "":
            unregister_attention_editor_diffusers(
                model, ori_forward, attn_processor="attn_proc"
            )
        else:
            unregister_attention_editor_diffusers(
                model, ori_forward, attn_processor="lora_attn_proc"
            )

        # save the original image, user editing instructions, synthesized image
        save_result = torch.cat(
            [
                source_image * 0.5 + 0.5,
                torch.ones((1, 3, full_h, 25)).cuda(),
                image_with_clicks * 0.5 + 0.5,
                torch.ones((1, 3, full_h, 25)).cuda(),
                gen_image[0:1],
            ],
            dim=-1,
        )
        # print(save_dir)
        save_dir_3_col = save_dir + "/3_col"
        if not os.path.isdir(save_dir_3_col):
            os.makedirs(save_dir_3_col)
        save_prefix = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
        save_image(save_result, os.path.join(save_dir_3_col, save_prefix + ".png"))

        out_image = gen_image.cpu().permute(0, 2, 3, 1).numpy()[0]
        out_image = (out_image * 255).astype(np.uint8)
        # total_image.appnend(out_image)
        draw_handle_points = []
        draw_target_points = []
        for idx, point in enumerate(current_points):
            draw_cur_point = torch.tensor(
                [point[0] / args.sup_res_h * full_h, point[1] / args.sup_res_w * full_w]
            ).int()
            draw_handle_points.append(draw_cur_point)
        for idx, point in enumerate(target_points):
            draw_tar_point = torch.tensor(
                [point[0] / args.sup_res_h * full_h, point[1] / args.sup_res_w * full_w]
            ).int()
            draw_target_points.append(draw_tar_point)
        # out_image = draw_handle_target_points(
        #     out_image, draw_handle_points, draw_target_points
        # )
        logger.info(f"handle Points: {draw_handle_points}")
        logger.info(f"Target Points: {draw_target_points}")
        save_pts = PIL.Image.fromarray(out_image)
        save_dir_points = save_dir + "/points"
        if not os.path.isdir(save_dir_points):
            os.makedirs(save_dir_points)
        save_pts.save(os.path.join(save_dir_points, save_prefix + "_points.png"))
        fig_ft = draw_featuremap(ft)
        save_dir_ft = save_dir + "/ft"
        if not os.path.isdir(save_dir_ft):
            os.makedirs(save_dir_ft)
        fig_ft.savefig(
            os.path.join(save_dir_ft, save_prefix + "_ft.png"), bbox_inches="tight"
        )
        plt.close(fig_ft)
        # drawable_init_code = updated_init_code.clone().cpu().detach()
        # fig_latent = draw_featuremap(drawable_init_code)
        # save_dir_lat = save_dir + "/latent"
        # if not os.path.isdir(save_dir_lat):
        #     os.makedirs(save_dir_lat)
        # fig_latent.savefig(
        #     os.path.join(save_dir_lat, save_prefix + "_lat.png"), bbox_inches="tight"
        # )
        # plt.close(fig_latent)
        yield out_image

