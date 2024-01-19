import os
import gradio as gr

from utils.ui_utils import get_points, undo_points
from utils.ui_utils import (
    clear_all,
    store_img,
    run_drag_r,
    locate_pt,
    mask_from_pic,
    load_config,
)

LENGTH = 480  # length of the square area displaying/editing images

with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown(
            """
        # RotationDrag
        """
        )
        # UI components for editing real images
    with gr.Tab(label="DragDiffusion in Rotation"):
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

                use_lora = gr.Checkbox(label="Use LoRA")
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
            lora_path = gr.Textbox(value="./lora_tmp/rotation", label="LoRA path")
            save_dir = gr.Textbox(value="./results/rot", label="Save path")
            sample_interval = gr.Number(
                label="Sampling Interval", value=20, visible=True
            )

        with gr.Row():
            x_location = gr.Number(
                label="x location", value=0, precision=0, visible=True
            )
            y_location = gr.Number(
                label="y location", value=0, precision=0, visible=True
            )
            set_point = gr.Button("Set Point", visible=True)
            upload_button = gr.UploadButton(
                "Click to upload Mask", file_types=["image"]
            )
            load_json = gr.UploadButton("Load Config", file_types=["json"])

        # algorithm specific parameters
        with gr.Tab("Drag Config"):
            with gr.Row():
                n_pix_step = gr.Number(
                    value=80,
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
    upload_button.upload(
        mask_from_pic,
        [upload_button, canvas],
        [original_image, selected_points, input_image, mask],
    )
    load_json.upload(
        load_config,
        [load_json, input_image, selected_points],
        [input_image, selected_points, prompt, n_pix_step],
    )
    set_point.click(
        locate_pt,
        [x_location, y_location, input_image, selected_points],
        [input_image, selected_points],
    )
    run_button.click(
        run_drag_r,
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
            use_lora,
            lora_step,
            lora_lr,
            lora_batch_size,
            lora_rank,
        ],
        [output_image],
    )
    clear_all_button.click(
        clear_all,
        [gr.Number(value=LENGTH, visible=False, precision=0)],
        [canvas, input_image, output_image, selected_points, original_image, mask],
    )

demo.queue().launch(share=True, debug=True)
