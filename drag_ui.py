import os
import gradio as gr

from utils.ui_utils import get_points, undo_points
from utils.ui_utils import clear_all, store_img, train_lora_interface, run_drag
from utils.ui_utils import (
    clear_all_free,
    store_img_free,
    run_freedrag,
    load_config,
    mask_from_pic,
)

LENGTH = 480  # length of the square area displaying/editing images

with gr.Blocks() as demo:
    # layout definition
    with gr.Row():
        gr.Markdown(
            """
        # Unofficial Implementation of Freedrag using diffusion
        """
        )

    # UI components for editing real images
    with gr.Tab(label="Free drag diffusion version"):
        mask_free = gr.State(value=None)  # store mask
        selected_points_free = gr.State([])  # store points
        original_image_free = gr.State(value=None)  # store original input image

        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    """<p style="text-align: center; font-size: 20px">Draw Mask</p>"""
                )
                canvas_free = gr.Image(
                    type="numpy",
                    tool="sketch",
                    label="Draw Mask",
                    show_label=True,
                    height=LENGTH,
                    width=LENGTH,
                )  # for mask painting
                use_lora_free = gr.Checkbox(label="Use LoRA")
            with gr.Column():
                gr.Markdown(
                    """<p style="text-align: center; font-size: 20px">Click Points</p>"""
                )
                input_image_free = gr.Image(
                    type="numpy",
                    label="Click Points",
                    show_label=True,
                    height=LENGTH,
                    width=LENGTH,
                )  # for points clicking
                undo_button_free = gr.Button("Undo point")
            with gr.Column():
                gr.Markdown(
                    """<p style="text-align: center; font-size: 20px">Editing Results</p>"""
                )
                output_image_free = gr.Image(
                    type="numpy",
                    label="Editing Results",
                    show_label=True,
                    height=LENGTH,
                    width=LENGTH,
                )
                with gr.Row():
                    run_button_free = gr.Button("Run")
                    clear_all_button_free = gr.Button("Clear All")

        # general parameters
        with gr.Row():
            prompt_free = gr.Textbox(label="Prompt")
            lora_path_free = gr.Textbox(value="./lora_tmp", label="LoRA path")
            save_dir_free = gr.Textbox(value="./results/free", label="Save path")
            lora_status_bar_free = gr.Textbox(label="display LoRA training status")

        with gr.Row():
            upload_button = gr.UploadButton(
                "Click to upload Mask", file_types=["image"]
            )
            load_json = gr.UploadButton("Load Config", file_types=["json"])

        # algorithm specific parameters
        with gr.Tab("Drag Config"):
            with gr.Row():
                n_pix_step_free = gr.Number(
                    value=2000,
                    label="number of pixel steps",
                    info="Number of gradient descent (motion supervision) steps on latent.",
                    precision=0,
                )
                lam_free = gr.Number(
                    value=0.1,
                    label="lam",
                    info="regularization strength on unmasked areas",
                    visible=False,
                )
                # n_actual_inference_step = gr.Number(value=40, label="optimize latent step", precision=0)
                inversion_strength_free = gr.Slider(
                    0,
                    1.0,
                    value=0.75,
                    label="inversion strength",
                    info="The latent at [inversion-strength * total-sampling-steps] is optimized for dragging.",
                )
                latent_lr_free = gr.Number(value=0.002, label="latent lr")
                start_step_free = gr.Number(
                    value=0, label="start_step", precision=0, visible=False
                )
                start_layer_free = gr.Number(
                    value=10, label="start_layer", precision=0, visible=False
                )

        with gr.Tab("Free Config"):
            with gr.Row():
                l_expected = gr.Number(
                    value=0.8,
                    label="l_expected",
                    info="Expected initial loss for each sub-motion",
                )
                d_max = gr.Number(
                    value=3,
                    label="d_max",
                    info="Max distance for each sub-motion (in the feature map) default=3",
                )
                sample_interval_free = gr.Number(
                    label="Interval", value=20, info="Sampling interval", visible=True
                )

        with gr.Tab("Base Model Config"):
            with gr.Row():
                local_models_dir_free = "local_pretrained_models"
                local_models_choice_free = [
                    os.path.join(local_models_dir_free, d)
                    for d in os.listdir(local_models_dir_free)
                    if os.path.isdir(os.path.join(local_models_dir_free, d))
                ]
                model_path_free = gr.Dropdown(
                    value="runwayml/stable-diffusion-v1-5",
                    label="Diffusion Model Path",
                    choices=[
                        "runwayml/stable-diffusion-v1-5",
                        "stabilityai/stable-diffusion-2-1",
                    ]
                    + local_models_choice_free,
                )
                vae_path_free = gr.Dropdown(
                    value="default",
                    label="VAE choice",
                    choices=["default", "stabilityai/sd-vae-ft-mse"]
                    + local_models_choice_free,
                )
                ft_layer_idx_free = gr.CheckboxGroup(
                    value=[3],
                    choices=[0, 1, 2, 3, 4],
                    label="Upsample feature layer index",
                    info="Starts from 1. 0 stands for mid block feature. 3 is default for sdv1-5",
                )

        with gr.Tab("LoRA Parameters"):
            with gr.Row():
                lora_step_free = gr.Number(
                    value=60,
                    label="LoRA training steps",
                    info="Suggest 200 for sdv2-1",
                    precision=0,
                )
                lora_lr_free = gr.Number(
                    value=0.0005,
                    label="LoRA learning rate",
                    info="Suggest 0.01 for sdv2-1",
                )
                lora_batch_size_free = gr.Number(
                    value=4, label="LoRA batch size", precision=0
                )
                lora_rank_free = gr.Number(value=16, label="LoRA rank", precision=0)
                lora_resolution_free = gr.Number(
                    value=512,
                    label="LoRA train resolution",
                    info="512 for 1-5, 768 for 2-1",
                    precision=0,
                )

    # UI components for editing real images
    with gr.Tab(label="DragDiffusion"):
        mask = gr.State(value=None)  # store mask
        selected_points = gr.State([])  # store points
        original_image = gr.State(value=None)  # store original input image
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    """<p style="text-align: center; font-size: 20px">Draw Mask</p>"""
                )
                canvas = gr.Image(
                    type="numpy",
                    tool="sketch",
                    label="Draw Mask",
                    show_label=True,
                    height=LENGTH,
                    width=LENGTH,
                )  # for mask painting
                train_lora_button = gr.Button("Train LoRA")
            with gr.Column():
                gr.Markdown(
                    """<p style="text-align: center; font-size: 20px">Click Points</p>"""
                )
                input_image = gr.Image(
                    type="numpy",
                    label="Click Points",
                    show_label=True,
                    height=LENGTH,
                    width=LENGTH,
                )  # for points clicking
                undo_button = gr.Button("Undo point")
            with gr.Column():
                gr.Markdown(
                    """<p style="text-align: center; font-size: 20px">Editing Results</p>"""
                )
                output_image = gr.Image(
                    type="numpy",
                    label="Editing Results",
                    show_label=True,
                    height=LENGTH,
                    width=LENGTH,
                )
                with gr.Row():
                    run_button = gr.Button("Run")
                    clear_all_button = gr.Button("Clear All")

        # general parameters
        with gr.Row():
            prompt = gr.Textbox(label="Prompt")
            lora_path = gr.Textbox(value="./lora_tmp", label="LoRA path")
            save_dir = gr.Textbox(value="./results/dragdiffusion", label="Save path")
            sample_interval = gr.Number(
                label="Sampling Interval", value=10, visible=True
            )
            lora_status_bar = gr.Textbox(label="display LoRA training status")

        # algorithm specific parameters
        with gr.Tab("Drag Config"):
            with gr.Row():
                n_pix_step = gr.Number(
                    value=40,
                    label="number of pixel steps",
                    info="Number of gradient descent (motion supervision) steps on latent.",
                    precision=0,
                )
                lam = gr.Number(
                    value=0.1,
                    label="lam",
                    info="regularization strength on unmasked areas",
                )
                # n_actual_inference_step = gr.Number(value=40, label="optimize latent step", precision=0)
                inversion_strength = gr.Slider(
                    0,
                    1.0,
                    value=0.75,
                    label="inversion strength",
                    info="The latent at [inversion-strength * total-sampling-steps] is optimized for dragging.",
                )
                latent_lr = gr.Number(value=0.01, label="latent lr")
                start_step = gr.Number(
                    value=0, label="start_step", precision=0, visible=False
                )
                start_layer = gr.Number(
                    value=10, label="start_layer", precision=0, visible=False
                )

        with gr.Tab("Base Model Config"):
            with gr.Row():
                local_models_dir = "local_pretrained_models"
                local_models_choice = [
                    os.path.join(local_models_dir, d)
                    for d in os.listdir(local_models_dir)
                    if os.path.isdir(os.path.join(local_models_dir, d))
                ]
                model_path = gr.Dropdown(
                    value="runwayml/stable-diffusion-v1-5",
                    label="Diffusion Model Path",
                    choices=[
                        "runwayml/stable-diffusion-v1-5",
                        "stabilityai/stable-diffusion-2-1",
                    ]
                    + local_models_choice,
                )
                vae_path = gr.Dropdown(
                    value="default",
                    label="VAE choice",
                    choices=["default", "stabilityai/sd-vae-ft-mse"]
                    + local_models_choice,
                )
                ft_layer_idx = gr.CheckboxGroup(
                    value=[3],
                    choices=[0, 1, 2, 3, 4],
                    label="Upsample feature layer index",
                    info="Starts from 1. 0 stands for mid block feature. 3 is default for sdv1-5",
                )

        with gr.Tab("LoRA Parameters"):
            with gr.Row():
                lora_step = gr.Number(
                    value=60, label="LoRA training steps", precision=0
                )
                lora_lr = gr.Number(value=0.0005, label="LoRA learning rate")
                lora_batch_size = gr.Number(
                    value=4, label="LoRA batch size", precision=0
                )
                lora_rank = gr.Number(value=16, label="LoRA rank", precision=0)

    # event definition
    # event for dragging user-input real image

    canvas_free.edit(
        store_img_free,
        [canvas_free, lora_resolution_free],
        [original_image_free, selected_points_free, input_image_free, mask_free],
    )
    input_image_free.select(
        get_points,
        [input_image_free, selected_points_free],
        [input_image_free],
    )
    upload_button.upload(
        mask_from_pic,
        [upload_button, canvas_free],
        [original_image_free, selected_points_free, input_image_free, mask_free],
    )
    load_json.upload(
        load_config,
        [load_json, input_image_free, selected_points_free],
        [input_image_free, selected_points_free, prompt_free, n_pix_step_free],
    )
    undo_button_free.click(
        undo_points,
        [original_image_free, mask_free],
        [input_image_free, selected_points_free],
    )
    run_button_free.click(
        run_freedrag,
        [
            original_image_free,
            input_image_free,
            mask_free,
            prompt_free,
            selected_points_free,
            inversion_strength_free,
            lam_free,
            latent_lr_free,
            n_pix_step_free,
            model_path_free,
            vae_path_free,
            lora_path_free,
            start_step_free,
            start_layer_free,
            l_expected,
            d_max,
            sample_interval_free,
            save_dir_free,
            lora_resolution_free,
            ft_layer_idx_free,
            use_lora_free,
            lora_step_free,
            lora_lr_free,
            lora_batch_size_free,
            lora_rank_free,
        ],
        [output_image_free],
    )

    clear_all_button_free.click(
        clear_all_free,
        [gr.Number(value=LENGTH, visible=False, precision=0)],
        [
            canvas_free,
            input_image_free,
            output_image_free,
            selected_points_free,
            original_image_free,
            mask_free,
        ],
    )

    canvas.edit(
        store_img, [canvas], [original_image, selected_points, input_image, mask]
    )
    input_image.select(
        get_points,
        [input_image, selected_points],
        [input_image],
    )
    undo_button.click(
        undo_points, [original_image, mask], [input_image, selected_points]
    )
    train_lora_button.click(
        train_lora_interface,
        [
            original_image,
            prompt,
            model_path,
            vae_path,
            lora_path,
            lora_step,
            lora_lr,
            lora_batch_size,
            lora_rank,
        ],
        [lora_status_bar],
    )
    run_button.click(
        run_drag,
        [
            original_image,
            input_image,
            mask,
            prompt,
            selected_points,
            inversion_strength,
            lam,
            latent_lr,
            n_pix_step,
            model_path,
            vae_path,
            lora_path,
            start_step,
            start_layer,
            save_dir,
            ft_layer_idx,
            sample_interval,
        ],
        [output_image],
    )
    clear_all_button.click(
        clear_all,
        [gr.Number(value=LENGTH, visible=False, precision=0)],
        [canvas, input_image, output_image, selected_points, original_image, mask],
    )


demo.queue().launch(share=True, debug=True)
