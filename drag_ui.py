# *************************************************************************
# Copyright (2023) Bytedance Inc.
#
# Copyright (2023) DragDiffusion Authors 
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 
#
#     http://www.apache.org/licenses/LICENSE-2.0 
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
# *************************************************************************

import os
import gradio as gr

from utils.ui_utils import get_points, undo_points
from utils.ui_utils import clear_all, store_img, train_lora_interface, run_drag
from utils.ui_utils import clear_all_gen, store_img_gen, gen_img, run_drag_gen
from utils.ui_utils import clear_all_free,store_img_free, train_lora_interface_free,run_freedrag,change_stop_state

LENGTH=480 # length of the square area displaying/editing images

with gr.Blocks() as demo:
    # layout definition
    with gr.Row():
        gr.Markdown("""
        # Unofficial Implementation of Freedrag using diffusion
        """)

    # UI components for editing real images
    with gr.Tab(label="Free Dragging Real Image"):
        mask_free = gr.State(value=None) # store mask
        selected_points_free = gr.State([]) # store points
        original_image_free = gr.State(value=None) # store original input image


        with gr.Row():
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Draw Mask</p>""")
                canvas_free = gr.Image(type="numpy", tool="sketch", label="Draw Mask",
                    show_label=True, height=LENGTH, width=LENGTH) # for mask painting
                train_lora_button_free = gr.Button("Train LoRA")
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Click Points</p>""")
                input_image_free = gr.Image(type="numpy", label="Click Points",
                    show_label=True, height=LENGTH, width=LENGTH) # for points clicking
                undo_button_free = gr.Button("Undo point")
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Editing Results</p>""")
                output_image_free = gr.Image(type="numpy", label="Editing Results",
                    show_label=True, height=LENGTH, width=LENGTH)
                with gr.Row():
                    run_button_free = gr.Button("Run")
                    clear_all_button_free = gr.Button("Clear All")
        stop_free = gr.Button('Stop',interactive=False,visible=False)

        # general parameters
        with gr.Row():
            prompt_free = gr.Textbox(label="Prompt")
            lora_path_free = gr.Textbox(value="./lora_tmp", label="LoRA path")
            save_dir_free = gr.Textbox(value="./results/free", label="Save path")
            lora_status_bar_free = gr.Textbox(label="display LoRA training status")

        # algorithm specific parameters
        with gr.Tab("Drag Config"):
            with gr.Row():
                n_pix_step_free = gr.Number(
                    value=2000,
                    label="number of pixel steps",
                    info="Number of gradient descent (motion supervision) steps on latent.",
                    precision=0)
                lam_free = gr.Number(value=0.1, label="lam", info="regularization strength on unmasked areas",visible=False)
                # n_actual_inference_step = gr.Number(value=40, label="optimize latent step", precision=0)
                inversion_strength_free = gr.Slider(0, 1.0,
                    value=0.75,
                    label="inversion strength",
                    info="The latent at [inversion-strength * total-sampling-steps] is optimized for dragging.")
                latent_lr_free = gr.Number(value=0.002, label="latent lr")
                start_step_free = gr.Number(value=0, label="start_step", precision=0, visible=False)
                start_layer_free = gr.Number(value=10, label="start_layer", precision=0, visible=False)

        with gr.Tab("Free Config"):
            with gr.Row():
                l_expected = gr.Number(value=1.0,label="l_expected", info="Expected initial loss for each sub-motion")
                d_max = gr.Number(value=3,label='d_max',info="Max distance for each sub-motion (in the feature map) default=3")
                sample_interval = gr.Number(label='Interval',value=200001,info="Sampling interval",visible=False)


        with gr.Tab("Base Model Config"):
            with gr.Row():
                local_models_dir_free = 'local_pretrained_models'
                local_models_choice_free = \
                    [os.path.join(local_models_dir_free,d) for d in os.listdir(local_models_dir_free) if os.path.isdir(os.path.join(local_models_dir_free,d))]
                model_path_free = gr.Dropdown(value="runwayml/stable-diffusion-v1-5",
                    label="Diffusion Model Path",
                    choices=[
                        "runwayml/stable-diffusion-v1-5",
                        "stabilityai/stable-diffusion-2-1",
                    ] + local_models_choice_free
                )
                vae_path_free = gr.Dropdown(value="default",
                    label="VAE choice",
                    choices=["default",
                    "stabilityai/sd-vae-ft-mse"] + local_models_choice_free
                )
                ft_layer_idx_free = gr.CheckboxGroup(value=[3],choices=[0,1,2,3,4],label="Upsample feature layer index", info = "Starts from 1. 0 stands for mid block feature. 3 is default for sdv1-5")

        with gr.Tab("LoRA Parameters"):
            with gr.Row():
                lora_step_free = gr.Number(value=60, label="LoRA training steps", precision=0)
                lora_lr_free = gr.Number(value=0.0005, label="LoRA learning rate")
                lora_batch_size_free = gr.Number(value=4, label="LoRA batch size", precision=0)
                lora_rank_free = gr.Number(value=16, label="LoRA rank", precision=0)

    # UI components for editing real images
    with gr.Tab(label="Editing Real Image"):
        mask = gr.State(value=None) # store mask
        selected_points = gr.State([]) # store points
        original_image = gr.State(value=None) # store original input image
        with gr.Row():
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Draw Mask</p>""")
                canvas = gr.Image(type="numpy", tool="sketch", label="Draw Mask",
                    show_label=True, height=LENGTH, width=LENGTH) # for mask painting
                train_lora_button = gr.Button("Train LoRA")
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Click Points</p>""")
                input_image = gr.Image(type="numpy", label="Click Points",
                    show_label=True, height=LENGTH, width=LENGTH) # for points clicking
                undo_button = gr.Button("Undo point")
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Editing Results</p>""")
                output_image = gr.Image(type="numpy", label="Editing Results",
                    show_label=True, height=LENGTH, width=LENGTH)
                with gr.Row():
                    run_button = gr.Button("Run")
                    clear_all_button = gr.Button("Clear All")

        # general parameters
        with gr.Row():
            prompt = gr.Textbox(label="Prompt")
            lora_path = gr.Textbox(value="./lora_tmp", label="LoRA path")
            save_dir = gr.Textbox(value="./results/dragdiffusion", label="Save path")
            lora_status_bar = gr.Textbox(label="display LoRA training status")

        # algorithm specific parameters
        with gr.Tab("Drag Config"):
            with gr.Row():
                n_pix_step = gr.Number(
                    value=40,
                    label="number of pixel steps",
                    info="Number of gradient descent (motion supervision) steps on latent.",
                    precision=0)
                lam = gr.Number(value=0.1, label="lam", info="regularization strength on unmasked areas")
                # n_actual_inference_step = gr.Number(value=40, label="optimize latent step", precision=0)
                inversion_strength = gr.Slider(0, 1.0,
                    value=0.75,
                    label="inversion strength",
                    info="The latent at [inversion-strength * total-sampling-steps] is optimized for dragging.")
                latent_lr = gr.Number(value=0.01, label="latent lr")
                start_step = gr.Number(value=0, label="start_step", precision=0, visible=False)
                start_layer = gr.Number(value=10, label="start_layer", precision=0, visible=False)

        with gr.Tab("Base Model Config"):
            with gr.Row():
                local_models_dir = 'local_pretrained_models'
                local_models_choice = \
                    [os.path.join(local_models_dir,d) for d in os.listdir(local_models_dir) if os.path.isdir(os.path.join(local_models_dir,d))]
                model_path = gr.Dropdown(value="runwayml/stable-diffusion-v1-5",
                    label="Diffusion Model Path",
                    choices=[
                        "runwayml/stable-diffusion-v1-5",
                        "stabilityai/stable-diffusion-2-1",
                    ] + local_models_choice
                )
                vae_path = gr.Dropdown(value="default",
                    label="VAE choice",
                    choices=["default",
                    "stabilityai/sd-vae-ft-mse"] + local_models_choice
                )
                ft_layer_idx = gr.CheckboxGroup(value=[3],choices=[0,1,2,3,4],label="Upsample feature layer index", info = "Starts from 1. 0 stands for mid block feature. 3 is default for sdv1-5")

        with gr.Tab("LoRA Parameters"):
            with gr.Row():
                lora_step = gr.Number(value=60, label="LoRA training steps", precision=0)
                lora_lr = gr.Number(value=0.0005, label="LoRA learning rate")
                lora_batch_size = gr.Number(value=4, label="LoRA batch size", precision=0)
                lora_rank = gr.Number(value=16, label="LoRA rank", precision=0)

    # UI components for editing generated images
    with gr.Tab(label="Editing Generated Image"):
        mask_gen = gr.State(value=None) # store mask
        selected_points_gen = gr.State([]) # store points
        original_image_gen = gr.State(value=None) # store the diffusion-generated image
        intermediate_latents_gen = gr.State(value=None) # store the intermediate diffusion latent during generation
        with gr.Row():
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Draw Mask</p>""")
                canvas_gen = gr.Image(type="numpy", tool="sketch", label="Draw Mask",
                    show_label=True, height=LENGTH, width=LENGTH) # for mask painting
                gen_img_button = gr.Button("Generate Image")
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Click Points</p>""")
                input_image_gen = gr.Image(type="numpy", label="Click Points",
                    show_label=True, height=LENGTH, width=LENGTH) # for points clicking
                undo_button_gen = gr.Button("Undo point")
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Editing Results</p>""")
                output_image_gen = gr.Image(type="numpy", label="Editing Results",
                    show_label=True, height=LENGTH, width=LENGTH)
                with gr.Row():
                    run_button_gen = gr.Button("Run")
                    clear_all_button_gen = gr.Button("Clear All")

        # general parameters
        with gr.Row():
            pos_prompt_gen = gr.Textbox(label="Positive Prompt")
            neg_prompt_gen = gr.Textbox(label="Negative Prompt")

        with gr.Tab("Generation Config"):
            with gr.Row():
                local_models_dir = 'local_pretrained_models'
                local_models_choice = \
                    [os.path.join(local_models_dir,d) for d in os.listdir(local_models_dir) if os.path.isdir(os.path.join(local_models_dir,d))]
                model_path_gen = gr.Dropdown(value="runwayml/stable-diffusion-v1-5",
                    label="Diffusion Model Path",
                    choices=[
                        "runwayml/stable-diffusion-v1-5",
                        "gsdf/Counterfeit-V2.5",
                        "emilianJR/majicMIX_realistic",
                        "SG161222/Realistic_Vision_V2.0",
                        "stablediffusionapi/interiordesignsuperm",
                        "stablediffusionapi/dvarch",
                    ] + local_models_choice
                )
                vae_path_gen = gr.Dropdown(value="default",
                    label="VAE choice",
                    choices=["default",
                    "stabilityai/sd-vae-ft-mse"] + local_models_choice
                )
                lora_path_gen = gr.Textbox(value="", label="LoRA path")
                gen_seed = gr.Number(value=65536, label="Generation Seed", precision=0)
                height = gr.Number(value=512, label="Height", precision=0)
                width = gr.Number(value=512, label="Width", precision=0)
                guidance_scale = gr.Number(value=7.5, label="CFG Scale")
                scheduler_name_gen = gr.Dropdown(
                    value="DDIM",
                    label="Scheduler",
                    choices=[
                        "DDIM",
                        "DPM++2M",
                        "DPM++2M_karras"
                    ]
                )
                n_inference_step_gen = gr.Number(value=50, label="Total Sampling Steps", precision=0)

        with gr.Tab("FreeU Parameters"):
            with gr.Row():
                b1_gen = gr.Slider(label='b1',
                                info='1st stage backbone factor',
                                minimum=1,
                                maximum=1.6,
                                step=0.05,
                                value=1.1)
                b2_gen = gr.Slider(label='b2',
                                info='2nd stage backbone factor',
                                minimum=1,
                                maximum=1.6,
                                step=0.05,
                                value=1.1)
                s1_gen = gr.Slider(label='s1',
                                info='1st stage skip factor',
                                minimum=0,
                                maximum=1,
                                step=0.05,
                                value=0.8)
                s2_gen = gr.Slider(label='s2',
                                info='2nd stage skip factor',
                                minimum=0,
                                maximum=1,
                                step=0.05,
                                value=0.8)

        with gr.Tab(label="Drag Config"):
            with gr.Row():
                n_pix_step_gen = gr.Number(
                    value=40,
                    label="Number of Pixel Steps",
                    info="Number of gradient descent (motion supervision) steps on latent.",
                    precision=0)
                lam_gen = gr.Number(value=0.1, label="lam", info="regularization strength on unmasked areas")
                # n_actual_inference_step_gen = gr.Number(value=40, label="optimize latent step", precision=0)
                inversion_strength_gen = gr.Slider(0, 1.0,
                    value=0.75,
                    label="Inversion Strength",
                    info="The latent at [inversion-strength * total-sampling-steps] is optimized for dragging.")
                latent_lr_gen = gr.Number(value=0.01, label="latent lr")
                start_step_gen = gr.Number(value=0, label="start_step", precision=0, visible=False)
                start_layer_gen = gr.Number(value=10, label="start_layer", precision=0, visible=False)

    # event definition
    # event for dragging user-input real image



    canvas_free.edit(
        store_img_free,
        [canvas_free],
        [original_image_free, selected_points_free, input_image_free, mask_free]
    )
    input_image_free.select(
        get_points,
        [input_image_free, selected_points_free],
        [input_image_free],
    )
    undo_button_free.click(
        undo_points,
        [original_image_free, mask_free],
        [input_image_free, selected_points_free]
    )
    train_lora_button_free.click(
        train_lora_interface_free,
        [original_image_free,
        prompt_free,
        model_path_free,
        vae_path_free,
        lora_path_free,
        lora_step_free,
        lora_lr_free,
        lora_batch_size_free,
        lora_rank_free],
        [lora_status_bar_free]
    )
    run_button_free.click(
        run_freedrag,
        [original_image_free,
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
        sample_interval,
        save_dir_free,
        ft_layer_idx_free,
        ],
        [output_image_free,
        stop_free]
    )

    stop_free.click(change_stop_state)

    clear_all_button.click(
        clear_all_free,
        [gr.Number(value=LENGTH, visible=False, precision=0)],
        [canvas_free,
        input_image_free,
        output_image_free,
        selected_points_free,
        original_image_free,
        mask_free]
    )

    canvas.edit(
        store_img,
        [canvas],
        [original_image, selected_points, input_image, mask]
    )
    input_image.select(
        get_points,
        [input_image, selected_points],
        [input_image],
    )
    undo_button.click(
        undo_points,
        [original_image, mask],
        [input_image, selected_points]
    )
    train_lora_button.click(
        train_lora_interface,
        [original_image,
        prompt,
        model_path,
        vae_path,
        lora_path,
        lora_step,
        lora_lr,
        lora_batch_size,
        lora_rank],
        [lora_status_bar]
    )
    run_button.click(
        run_drag,
        [original_image,
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
        ],
        [output_image]
    )
    clear_all_button.click(
        clear_all,
        [gr.Number(value=LENGTH, visible=False, precision=0)],
        [canvas,
        input_image,
        output_image,
        selected_points,
        original_image,
        mask]
    )

    # event for dragging generated image
    canvas_gen.edit(
        store_img_gen,
        [canvas_gen],
        [original_image_gen, selected_points_gen, input_image_gen, mask_gen]
    )
    input_image_gen.select(
        get_points,
        [input_image_gen, selected_points_gen],
        [input_image_gen],
    )
    gen_img_button.click(
        gen_img,
        [
        gr.Number(value=LENGTH, visible=False, precision=0),
        height,
        width,
        n_inference_step_gen,
        scheduler_name_gen,
        gen_seed,
        guidance_scale,
        pos_prompt_gen,
        neg_prompt_gen,
        model_path_gen,
        vae_path_gen,
        lora_path_gen,
        b1_gen,
        b2_gen,
        s1_gen,
        s2_gen,
        ],
        [canvas_gen, input_image_gen, output_image_gen, mask_gen, intermediate_latents_gen]
    )
    undo_button_gen.click(
        undo_points,
        [original_image_gen, mask_gen],
        [input_image_gen, selected_points_gen]
    )
    run_button_gen.click(
        run_drag_gen,
        [
        n_inference_step_gen,
        scheduler_name_gen,
        original_image_gen, # the original image generated by the diffusion model
        input_image_gen, # image with clicking, masking, etc.
        intermediate_latents_gen,
        guidance_scale,
        mask_gen,
        pos_prompt_gen,
        neg_prompt_gen,
        selected_points_gen,
        inversion_strength_gen,
        lam_gen,
        latent_lr_gen,
        n_pix_step_gen,
        model_path_gen,
        vae_path_gen,
        lora_path_gen,
        start_step_gen,
        start_layer_gen,
        b1_gen,
        b2_gen,
        s1_gen,
        s2_gen,
        ],
        [output_image_gen]
    )
    clear_all_button_gen.click(
        clear_all_gen,
        [gr.Number(value=LENGTH, visible=False, precision=0)],
        [canvas_gen,
        input_image_gen,
        output_image_gen,
        selected_points_gen,
        original_image_gen,
        mask_gen,
        intermediate_latents_gen,
        ]
    )


demo.queue().launch(share=True, debug=True)
