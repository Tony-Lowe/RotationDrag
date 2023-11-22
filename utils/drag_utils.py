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

import copy
import torch
import torch.nn.functional as F
import numpy as np
from loguru import logger
from torchvision.transforms.functional import rotate
from math import pi


def point_tracking(F0, F1, handle_points, handle_points_init, args):
    with torch.no_grad():
        for i in range(len(handle_points)):
            p_i0, p_i = handle_points_init[i], handle_points[i]
            f0 = F0[:, :, int(p_i0[0]), int(p_i0[1])]

            r1, r2 = int(p_i[0]) - args.r_p, int(p_i[0]) + args.r_p + 1
            c1, c2 = int(p_i[1]) - args.r_p, int(p_i[1]) + args.r_p + 1
            F1_neighbor = F1[:, :, r1:r2, c1:c2]
            all_dist = (
                (f0.unsqueeze(dim=-1).unsqueeze(dim=-1) - F1_neighbor).abs().sum(dim=1)
            )
            all_dist = all_dist.squeeze(dim=0)
            # WARNING: no boundary protection right now
            row, col = divmod(all_dist.argmin().item(), all_dist.shape[-1])
            handle_points[i][0] = p_i[0] - args.r_p + row
            handle_points[i][1] = p_i[1] - args.r_p + col
        return handle_points


def check_handle_reach_target(handle_points, target_points):
    # dist = (torch.cat(handle_points,dim=0) - torch.cat(target_points,dim=0)).norm(dim=-1)
    all_dist = list(map(lambda p, q: (p - q).norm(), handle_points, target_points))
    return (torch.tensor(all_dist) < 2.0).all()


# obtain the bilinear interpolated feature patch centered around (x, y) with radius r
def interpolate_feature_patch(feat, y, x, r):
    """
    return: (B,C,2r+1,2r+1)
    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x1.float() - x) * (y - y0.float())
    wc = (x - x0.float()) * (y1.float() - y)
    wd = (x - x0.float()) * (y - y0.float())

    Ia = feat[:, :, y0 - r : y0 + r + 1, x0 - r : x0 + r + 1]
    Ib = feat[:, :, y1 - r : y1 + r + 1, x0 - r : x0 + r + 1]
    Ic = feat[:, :, y0 - r : y0 + r + 1, x1 - r : x1 + r + 1]
    Id = feat[:, :, y1 - r : y1 + r + 1, x1 - r : x1 + r + 1]

    return Ia * wa + Ib * wb + Ic * wc + Id * wd


def interpolate_feature_patch_plus(feature, position):
    # feature: (1,C,H,W)
    # position: (N,2)
    # return: (N,C)
    device = feature.device

    y = position[:, 0]
    x = position[:, 1]

    x0 = x.long()
    x1 = x0 + 1
    y0 = y.long()
    y1 = y0 + 1

    wa = ((x1.float() - x) * (y1.float() - y)).to(device).unsqueeze(1).detach()
    wb = ((x1.float() - x) * (y - y0.float())).to(device).unsqueeze(1).detach()
    wc = ((x - x0.float()) * (y1.float() - y)).to(device).unsqueeze(1).detach()
    wd = ((x - x0.float()) * (y - y0.float())).to(device).unsqueeze(1).detach()

    Ia = feature[:, :, y0, x0].squeeze(0).transpose(1, 0)
    Ib = feature[:, :, y1, x0].squeeze(0).transpose(1, 0)
    Ic = feature[:, :, y0, x1].squeeze(0).transpose(1, 0)
    Id = feature[:, :, y1, x1].squeeze(0).transpose(1, 0)

    output = Ia * wa + Ib * wb + Ic * wc + Id * wd
    return output


def drag_diffusion_update(
    model, init_code, t, handle_points, target_points, mask, args
):
    assert len(handle_points) == len(
        target_points
    ), "number of handle point must equals target points"

    text_emb = model.get_text_embeddings(args.prompt).detach()
    # the init output feature of unet
    with torch.no_grad():
        unet_output, F0 = model.forward_unet_features(
            init_code,
            t,
            encoder_hidden_states=text_emb,
            layer_idx=args.unet_feature_idx,
            interp_res_h=args.sup_res_h,
            interp_res_w=args.sup_res_w,
        )
        x_prev_0, _ = model.step(unet_output, t, init_code)
        # init_code_orig = copy.deepcopy(init_code)

    # prepare optimizable init_code and optimizer
    init_code.requires_grad_(True)
    optimizer = torch.optim.Adam([init_code], lr=args.lr)

    # prepare for point tracking and background regularization
    handle_points_init = copy.deepcopy(handle_points)
    interp_mask = F.interpolate(
        mask, (init_code.shape[2], init_code.shape[3]), mode="nearest"
    )

    # prepare amp scaler for mixed-precision training
    scaler = torch.cuda.amp.GradScaler()
    for step_idx in range(args.n_pix_step):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            unet_output, F1 = model.forward_unet_features(
                init_code,
                t,
                encoder_hidden_states=text_emb,
                layer_idx=args.unet_feature_idx,
                interp_res_h=args.sup_res_h,
                interp_res_w=args.sup_res_w,
            )
            x_prev_updated, _ = model.step(unet_output, t, init_code)

            # do point tracking to update handle points before computing motion supervision loss
            if step_idx != 0:
                handle_points = point_tracking(
                    F0, F1, handle_points, handle_points_init, args
                )
                print("new handle points", handle_points)

            # break if all handle points have reached the targets
            if check_handle_reach_target(handle_points, target_points):
                ret_ft = F1.clone().cpu().detach()
                break

            loss = 0.0
            for i in range(len(handle_points)):
                pi, ti = handle_points[i], target_points[i]
                # skip if the distance between target and source is less than 1
                if (ti - pi).norm() < 2.0:
                    continue

                di = (ti - pi) / (ti - pi).norm()

                # motion supervision
                f0_patch = F1[
                    :,
                    :,
                    int(pi[0]) - args.r_m : int(pi[0]) + args.r_m + 1,
                    int(pi[1]) - args.r_m : int(pi[1]) + args.r_m + 1,
                ].detach()
                f1_patch = interpolate_feature_patch(
                    F1, pi[0] + di[0], pi[1] + di[1], args.r_m
                )
                loss += ((2 * args.r_m + 1) ** 2) * F.l1_loss(f0_patch, f1_patch)

            # masked region must stay unchanged
            loss += (
                args.lam
                * ((x_prev_updated - x_prev_0) * (1.0 - interp_mask)).abs().sum()
            )
            # loss += args.lam * ((init_code_orig-init_code)*(1.0-interp_mask)).abs().sum()
            print("loss total=%f" % (loss.item()))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        ret_ft = F1.clone().cpu().detach()
        if step_idx % args.sample_interval == 0:
            yield init_code, handle_points, ret_ft
    yield init_code, handle_points, ret_ft


def get_rotated_pt(current_pt, angles, args):
    """
    :params current_pt: current handle points shape of [2]
    :params angles: angles to rotate, shape of [intervals]
    :returns: rotated points shape of intervals*2
    """
    current_pt = current_pt.unsqueeze(0)
    # angles = angles.unsqueeze(1)
    current_pt_repeat = current_pt.repeat_interleave(angles.shape[0], dim=0)
    # angles_repeat = angles.repeat(current_pt.shape[0], 1)
    rotated_pt = torch.cat(
        (
            (current_pt_repeat[:, 0].unsqueeze(1) - args.sup_res_h * 0.5)
            * torch.cos(angles)
            - (current_pt_repeat[:, 1].unsqueeze(1) - args.sup_res_h * 0.5)
            * torch.sin(angles)
            + args.sup_res_h * 0.5,
            (current_pt_repeat[:, 0].unsqueeze(1) - args.sup_res_w * 0.5)
            * torch.sin(angles)
            + (current_pt_repeat[:, 1].unsqueeze(1) - args.sup_res_w * 0.5)
            * torch.cos(angles)
            + args.sup_res_w * 0.5,
        ),
        dim=1,
    )
    # print(rotated_pt.shape)
    return rotated_pt.detach()


def get_offset_matrix(win_r, ft_ratio):
    """
    generate offset matrix near the point
    :params win_r: window radius
    :params ft_ratio: feature resolution / image resolution
    """
    k = torch.linspace(-(win_r * ft_ratio), win_r * ft_ratio, steps=win_r)
    # k = torch.linspace(-(win_r//2),win_r//2,steps= win_r)
    k1 = k.repeat(win_r, 1).transpose(1, 0).flatten(0).unsqueeze(0)
    k2 = k.repeat(1, win_r)
    return torch.cat((k1, k2), dim=0).transpose(1, 0)


def get_rotated_features(model, angles, t, text_emb, args):
    """
    :params model: diffusion model
    :params source_image: source image, use it to generate features shape of 1*3*H*W
    :params angles: angles to rotate, shape of intervals*1, range: [0, pi]
    :returns: rotated features shape of intervals*1*H*W
    """
    rotated_image = rotate(args.source_image, angles.item() * 180 / pi)
    rotated_invert_code = model.invert(  # TODO: add them in args
        rotated_image,
        args.prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.n_inference_step,
        num_actual_inference_steps=args.n_actual_inference_step,
    )
    unet_output, rotated_features = model.forward_unet_features(
        rotated_invert_code,
        t,
        encoder_hidden_states=text_emb,
        layer_idx=args.unet_feature_idx,
        interp_res_h=args.sup_res_h,
        interp_res_w=args.sup_res_w,
    )
    return rotated_features, rotated_invert_code


def compute_angle(tar, src, args):
    """
    Note that tar and src are all tensor with the shape [2].
    Returns an angle in range [0, pi]
    """
    angles = torch.atan2(
        torch.tensor([tar[0] - args.sup_res_h * 0.5, src[0] - args.sup_res_h * 0.5]),
        torch.tensor([tar[1] - args.sup_res_w * 0.5, src[1] - args.sup_res_w * 0.5]),
    )
    angle = torch.tensor([angles[0] - angles[1]])
    return angle


def get_each_angle(
    model,
    t,
    text_emb,
    current,
    target_final,
    curr_ini,
    F1,
    offset_matrix,
    args,
):
    """
    :param current: tensor shape of 2, current handle points
    :param target_final: tensor same shape with current, final taget points
    :param curr_ini: Initial source point
    :param F1: Updated Feature
    :param max_angle: maximum angle of rotation [0, pi]
    :param offset_matrix: help to compute the patch around handle points
    :param args: args passed by main threads, has source_image, prompt,etc
    """
    curr_angle = compute_angle(current, curr_ini, args)
    logger.info(f"current angle: {curr_angle}")
    angle_remain = compute_angle(target_final, current, args)
    logger.info(f"remaining angle: {angle_remain}")
    angle_max = args.max_angle
    interval_number = args.interval_number
    intervals = torch.arange(
        0, 1 + 1 / interval_number, 1 / interval_number, device=current.device
    )[1:].unsqueeze(
        1
    )  # [intervals, 1]
    target_angle_max = min(angle_max / (angle_remain + 1e-8), 1) * angle_remain
    candidate_angles = curr_angle.unsqueeze(0) + intervals * target_angle_max.unsqueeze(
        0
    )  # [intervals, 1]
    # logger.info(f"candidate_angles: {candidate_angles}")
    candidate_points = get_rotated_pt(
        current, candidate_angles, args
    )  # [intervals * 2]
    # logger.info(f"candidate_points: {candidate_points}")
    candidate_points_repeat = candidate_points.repeat_interleave(
        offset_matrix.shape[0], dim=0
    )  # [intervals * 9, 2]
    offset_matrix_repeat = offset_matrix.repeat(
        intervals.shape[0], 1
    )  # [intervals * 9, 2]
    candidate_points_local = candidate_points_repeat + offset_matrix_repeat
    # ft_patch_all = []
    dist_all = []
    for idx, angle in enumerate(candidate_angles):
        ft, rotated_lat = get_rotated_features(
            model, angle, t, text_emb, args
        )  # 1*c*h*w
        ft_patch = interpolate_feature_patch_plus(
            ft, candidate_points_local[idx * 9 : (idx + 1) * 9, :]
        )  # [9,C]
        f1_patch = interpolate_feature_patch_plus(
            F1, candidate_points_local[idx * 9 : (idx + 1) * 9, :]
        )  # [9]
        dist = abs(ft_patch - f1_patch).mean()  # []
        dist_all.append(dist)
        # ft_patch_all.append(ft_patch)
        # if idx == 0:
        # pt_ft_all = ft[
        #     :, :, int(candidate_points[idx, 0]), int(candidate_points[idx, 1])
        # ]  # [1,C]
        # dif_patch = dist
        # ft_patch_all = ft_patch
        # else:
        # pt_ft_all = torch.cat(
        #     (
        #         pt_ft_all,
        #         ft[
        #             :,
        #             :,
        #             int(candidate_points[idx, 0]),
        #             int(candidate_points[idx, 1]),
        #         ],
        #     )
        # )  # In the end, we get a tensor shape of [intervals,C]
        # dif_patch = torch.cat((dif_patch, dist))  # [intervals]
        # ft_patch_all = torch.cat(
        #     (ft_patch_all, ft_patch)
        # )  # In the end, we get a tensor shape of [intervals*9,C]
    # ft_patch_all = ft_patch_all.reshape((intervals.shape[0], -1))
    # dif_patch = abs(
    #     ft_patch_all
    #     - interpolate_feature_patch_plus(F1, current + offset_matrix)
    #     .flatten(0)
    #     .unsqueeze(0)
    # ).mean(
    #     1
    # )  # intervals*[distance in feature space]

    # point tracking
    dif_patch = torch.tensor(dist_all)
    min_idx = torch.argmin(dif_patch)
    # f0 = ft_patch_all[min_idx, :]
    # y1, y2 = int(current[0]) - args.r_p, int(current[0]) + args.r_p
    # x1, x2 = int(current[1]) - args.r_p, int(current[1]) + args.r_p
    # F1_neighbor = F1[:, :, y1:y2, x1:x2]
    # all_dist = (f0.unsqueeze(dim=-1).unsqueeze(dim=-1) - F1_neighbor).abs().sum(dim=1)
    # all_dist = all_dist.squeeze(dim=0)
    # row, col = divmod(all_dist.argmin().item(), all_dist.shape[-1])
    # current[0] = current[0] - args.r_p + row
    # current[1] = current[1] - args.r_p + col
    return candidate_points[min_idx]


def get_current_target_r(
    model,
    t,
    text_emb,
    handle_points,
    target_points,
    handle_points_ini,
    F1,
    offset_matrix,
    args,
):
    for idx in range(len(target_points)):
        if check_handle_reach_target(handle_points[idx], target_points[idx]):
            continue
        handle_points[idx] = get_each_angle(
            model,
            t,
            text_emb,
            handle_points[idx],
            target_points[idx],
            handle_points_ini[idx],
            F1,
            offset_matrix,
            args,
        )
    return handle_points


def point_tracking_r(model, t, text_emb, F1, handle_points, handle_points_init, args):
    with torch.no_grad():
        for idx in range(len(handle_points)):
            p_i0, p_i = handle_points_init[idx], handle_points[idx]
            angle0 = compute_angle(p_i, p_i0, args)
            cpy_img = args.source_image.clone().detach()
            angle_180 = angle0.item() * 180 / pi
            logger.info(f"Angle in point tracking: {angle_180}")
            rotated_img = rotate(cpy_img, angle_180)
            lat_r = model.invert(
                rotated_img,
                prompt=args.prompt,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.n_inference_step,
                num_actual_inference_steps=args.n_actual_inference_step,
            )
            p_i0_r = get_rotated_pt(p_i0, angle0, args).squeeze()
            unet_output, F0 = model.forward_unet_features(
                lat_r,
                t,
                encoder_hidden_states=text_emb,
                layer_idx=args.unet_feature_idx,
                interp_res_h=args.sup_res_h,
                interp_res_w=args.sup_res_w,
            )
            f0 = F0[:,:,int(p_i0_r[0]),int(p_i0_r[1])]
            r1, r2 = int(p_i[0]) - args.r_p, int(p_i[0]) + args.r_p + 1
            c1, c2 = int(p_i[1]) - args.r_p, int(p_i[1]) + args.r_p + 1
            F1_neighbor = F1[:, :, r1:r2, c1:c2]
            all_dist = (
                (f0.unsqueeze(dim=-1).unsqueeze(dim=-1) - F1_neighbor).abs().sum(dim=1)
            )
            # WARNING: no boundary protection right now
            row, col = divmod(all_dist.argmin().item(), all_dist.shape[-1])
            handle_points[idx][0] = p_i[0] - args.r_p + row
            handle_points[idx][1] = p_i[1] - args.r_p + col
        return handle_points


def drag_diffusion_update_r(
    model, init_code, t, handle_points, target_points, mask, args
):
    assert len(handle_points) == len(
        target_points
    ), "number of handle point must equals target points"

    text_emb = model.get_text_embeddings(args.prompt).detach()
    # the init output feature of unet
    with torch.no_grad():
        unet_output, F0 = model.forward_unet_features(
            init_code,
            t,
            encoder_hidden_states=text_emb,
            layer_idx=args.unet_feature_idx,
            interp_res_h=args.sup_res_h,
            interp_res_w=args.sup_res_w,
        )
        x_prev_0, _ = model.step(unet_output, t, init_code)
        # init_code_orig = copy.deepcopy(init_code)

    # prepare optimizable init_code and optimizer
    init_code.requires_grad_(True)
    optimizer = torch.optim.Adam([init_code], lr=args.lr)

    # prepare for point tracking and background regularization
    handle_points_init = copy.deepcopy(handle_points)
    interp_mask = F.interpolate(
        mask, (init_code.shape[2], init_code.shape[3]), mode="nearest"
    )

    # prepare amp scaler for mixed-precision training
    scaler = torch.cuda.amp.GradScaler()
    offset_matrix = get_offset_matrix(
        args.r_p, args.res_ratio
    )  # TODO: Check whether they are in args
    for step_idx in range(args.n_pix_step):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            unet_output, F1 = model.forward_unet_features(
                init_code,
                t,
                encoder_hidden_states=text_emb,
                layer_idx=args.unet_feature_idx,
                interp_res_h=args.sup_res_h,
                interp_res_w=args.sup_res_w,
            )
            x_prev_updated, _ = model.step(unet_output, t, init_code)

            # do point tracking to update handle points before computing motion supervision loss
            if step_idx != 0:
                # handle_points = get_current_target_r(
                #     model,
                #     t,
                #     text_emb,
                #     handle_points,
                #     target_points,
                #     handle_points_init,
                #     F1,
                #     offset_matrix,
                #     args,
                # )
                handle_points = point_tracking_r(model,t,text_emb,F1,handle_points,handle_points_init,args)
                logger.info(f"new handle points: {handle_points}")

            # break if all handle points have reached the targets
            if check_handle_reach_target(handle_points, target_points):
                ret_ft = F1.clone().cpu().detach()
                break

            loss = 0.0
            for i in range(len(handle_points)):
                p_i, t_i = handle_points[i], target_points[i]
                # skip if the distance between target and source is less than 1
                if (t_i - p_i).norm() < 2.0:
                    continue

                d_i = (t_i - p_i) / (t_i - p_i).norm()

                angle = compute_angle(p_i + d_i, p_i, args)
                # %----------------------------%
                # Adding Rotation Modification
                with torch.no_grad():
                    rot_img = args.source_image.clone().detach()
                    angle_180 = angle.item() * 180 / pi
                    logger.info(f"Angle in optimization: {angle_180}")
                    rot_img = rotate(rot_img, angle_180)
                    lat_r = model.invert(
                        rot_img,
                        prompt=args.prompt,
                        guidance_scale=args.guidance_scale,
                        num_inference_steps=args.n_inference_step,
                        num_actual_inference_steps=args.n_actual_inference_step,
                    )
                    unet_output_r, F0r = model.forward_unet_features(
                        lat_r,
                        t,
                        encoder_hidden_states=text_emb,
                        layer_idx=args.unet_feature_idx,
                        interp_res_h=args.sup_res_h,
                        interp_res_w=args.sup_res_w,
                    )
                # %----------------------------%

                # motion supervision
                rotated_p_i = get_rotated_pt(p_i, angle, args).squeeze()
                f0_patch = F0r[
                    :,
                    :,
                    int(rotated_p_i[0]) - args.r_m : int(rotated_p_i[0]) + args.r_m + 1,
                    int(rotated_p_i[1]) - args.r_m : int(rotated_p_i[1]) + args.r_m + 1,
                ].detach()
                # I use rotated pts to prevent unreasonable rotation leading to comparing wrong feature
                f1_patch = interpolate_feature_patch(
                    F1, rotated_p_i[0], rotated_p_i[1], args.r_m
                )
                loss += ((2 * args.r_m + 1) ** 2) * F.l1_loss(f0_patch, f1_patch)

            # masked region must stay unchanged
            loss += (
                args.lam
                * ((x_prev_updated - x_prev_0) * (1.0 - interp_mask)).abs().sum()
            )
            # loss += args.lam * ((init_code_orig-init_code)*(1.0-interp_mask)).abs().sum()
            logger.info("loss total=%f" % (loss.item()))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        ret_ft = F1.clone().cpu().detach()
        if step_idx % args.sample_interval == 0:
            logger.info("Sampling Interval reached")
            yield init_code, handle_points, ret_ft
    yield init_code, handle_points, ret_ft


def update_signs(
    sign_point_pairs,
    current_point,
    target_point,
    loss_supervised,
    threshold_d,
    threshold_l,
):
    distance = (current_point - target_point).pow(2).sum(dim=1).pow(0.5)
    sign_point_pairs[distance < threshold_d] = 1
    sign_point_pairs[distance >= threshold_d] = 0
    sign_point_pairs[loss_supervised > threshold_l] = 0


def get_each_point(
    current,
    target_final,
    L,
    feature_map,
    max_distance,
    template_feature,
    loss_initial,
    loss_end,
    offset_matrix,
    threshold_l,
):
    d_max = max_distance
    d_remain = (current - target_final).pow(2).sum().pow(0.5)
    interval_number = 10  # for point localization
    intervals = torch.arange(
        0, 1 + 1 / interval_number, 1 / interval_number, device=current.device
    )[1:].unsqueeze(1)

    if loss_end < threshold_l:
        target_max = current + min(d_max / (d_remain + 1e-8), 1) * (
            target_final - current
        )
        candidate_points = (1 - intervals) * current.unsqueeze(
            0
        ) + intervals * target_max.unsqueeze(0)
        candidate_points_repeat = candidate_points.repeat_interleave(
            offset_matrix.shape[0], dim=0
        )
        offset_matrix_repeat = offset_matrix.repeat(intervals.shape[0], 1)

        candidate_points_local = candidate_points_repeat + offset_matrix_repeat
        features_all = interpolate_feature_patch_plus(
            feature_map, candidate_points_local
        )

        features_all = features_all.reshape((intervals.shape[0], -1))
        dif_location = abs(
            features_all - template_feature.flatten(0).unsqueeze(0)
        ).mean(1)
        min_idx = torch.argmin(abs(dif_location - L))
        current_best = candidate_points[min_idx, :]
        return current_best

    elif loss_end < loss_initial:
        return current

    else:
        current = current - min(d_max / (d_remain + 1e-8), 1) * (
            target_final - current
        )  # rollback
        d_remain = (current - target_final).pow(2).sum().pow(0.5)
        target_max = current + min(2 * d_max / (d_remain + 1e-8), 1) * (
            target_final - current
        )  # double the localization range

        candidate_points = (1 - intervals) * current.unsqueeze(
            0
        ) + intervals * target_max.unsqueeze(0)
        candidate_points_repeat = candidate_points.repeat_interleave(
            offset_matrix.shape[0], dim=0
        )
        offset_matrix_repeat = offset_matrix.repeat(intervals.shape[0], 1)
        candidate_points_local = candidate_points_repeat + offset_matrix_repeat
        features_all = interpolate_feature_patch_plus(
            feature_map, candidate_points_local
        )
        features_all = features_all.reshape((intervals.shape[0], -1))
        dif_location = abs(
            features_all - template_feature.flatten(0).unsqueeze(0)
        ).mean(1)
        min_idx = torch.argmin(dif_location)  # l=0 in this case
        current_best = candidate_points[min_idx, :]
        return current_best


def get_current_target(
    sign_points,
    current_target,
    target_point,
    L,
    feature_map,
    max_distance,
    template_feature,
    loss_initial,
    loss_end,
    offset_matrix,
    threshold_l,
):
    # L is the expectation
    for k in range(target_point.shape[0]):
        if (
            sign_points[k] == 0
        ):  # sign_points ==0 means the remains distance to target point is larger than the preset threshold
            current_target[k, :] = get_each_point(
                current_target[k, :],
                target_point[k, :],
                L,
                feature_map,
                max_distance,
                template_feature[k],
                loss_initial[k],
                loss_end[k],
                offset_matrix,
                threshold_l,
            )
    return current_target


def get_lad(loss_k, a, b):
    # in freedrag, they wrote, as I quote, "xishu = xishu = 1/(1+(a*(loss_k-b)).exp())"
    lad = 1 / (1 + (a * (loss_k - b)).exp())
    return lad


def free_drag_update(model, init_code, t, handle_points, target_points, mask, args):
    """
    :param init_code: latent
    """
    assert (
        handle_points.shape[0] == target_points.shape[0]
    ), "number of handle point must equals target points"

    text_emb = model.get_text_embeddings(args.prompt).detach()
    # the init output feature of unet
    with torch.no_grad():
        unet_output, F0 = model.forward_unet_features(
            init_code,
            t,
            encoder_hidden_states=text_emb,
            layer_idx=args.unet_feature_idx,
            interp_res_h=args.sup_res_h,
            interp_res_w=args.sup_res_w,
        )
        # F0 has all the feature from midblock to upblock 3
        # x_prev_0,_ = model.step(unet_output, t, init_code) # x_prev_0 is the sample you get when you don't drag
        # init_code_orig = copy.deepcopy(init_code)
    latent_trainable = init_code.detach().clone().requires_grad_(True)
    # latent_untrainable = init_code.detach().clone().requires_grad_(False)

    optimizer = torch.optim.Adam([{"params": latent_trainable}], lr=args.lr)
    # ,  eps=1e-08, weight_decay=0, amsgrad=False)
    Loss_l1 = torch.nn.L1Loss()

    use_mask = False
    if torch.any(mask):
        mask = torch.tensor(mask, dtype=torch.float32, device=args.device)
        interp_mask = F.interpolate(mask, (F0.shape[2], F0.shape[3]), mode="bilinear")
        mask_resized = interp_mask.repeat(1, F0.shape[1], 1, 1) > 0
        use_mask = True

    point_pairs_number = target_points.shape[0]
    template_feature = []
    # TODO: r_p is the win_r in freedrag. It should be 3
    offset_matrix = get_offset_matrix(args.r_p, args.res_ratio).to(args.device)
    for idx in range(point_pairs_number):
        template_feature.append(
            interpolate_feature_patch_plus(F0, handle_points[idx, :] + offset_matrix)
        )
    step_num = 0
    current_targets = handle_points.clone().to(args.device)
    # target_points = target_points.to(args.device)
    current_feature_map = F0.detach()
    sign_points = torch.zeros(point_pairs_number).to(
        args.device
    )  # determiner if the localization point is closest to target point
    loss_ini = torch.zeros(point_pairs_number).to(args.device)
    loss_end = torch.zeros(point_pairs_number).to(args.device)
    step_threshold = args.n_pix_step
    while step_num < args.n_pix_step:
        if torch.all(sign_points == 1):
            logger.info("Target Points reached successfully!")
            yield latent_input, current_targets, F1.clone().cpu().detach()
            break
        current_targets = get_current_target(
            sign_points,
            current_targets,
            target_points,
            args.l_expected,
            current_feature_map,
            args.dmax,
            template_feature,
            loss_ini,
            loss_end,
            offset_matrix,
            args.threshold_l,
        )
        logger.info(f"current_targets {current_targets}")
        d_remain = (current_targets - target_points).pow(2).sum(dim=1).pow(0.5)
        for step in range(5):
            logger.info(f"step: {step_num}")
            step_num += 1

            # latent_input = torch.cat((latent_trainable,latent_untrainable),dim=1) # Why???
            latent_input = latent_trainable  # Freedrag wrote the above line. I can't figure out why.
            # autocast boosts speed signifincantly
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                unet_output, F1 = model.forward_unet_features(
                    latent_input,
                    t,
                    encoder_hidden_states=text_emb,
                    layer_idx=args.unet_feature_idx,
                    interp_res_h=args.sup_res_h,
                    interp_res_w=args.sup_res_w,
                )
                # x_prev_updated,_ = model.step(unet_output, t, latent_input[0])

                loss_supervised = torch.zeros(point_pairs_number).to(args.device)
                current_feature = []  # F_r^k
                for idx in range(point_pairs_number):
                    current_feature.append(
                        interpolate_feature_patch_plus(
                            F1, current_targets[idx, :] + offset_matrix
                        )
                    )
                    loss_supervised[idx] = Loss_l1(
                        current_feature[idx], template_feature[idx].detach()
                    )

                loss_featrue = loss_supervised.sum()

                if use_mask:
                    loss_mask = Loss_l1(F1[~mask_resized], F0[~mask_resized].detach())
                    loss = loss_featrue + 10 * loss_mask
                else:
                    loss = loss_featrue
            logger.info(f"Loss: {loss}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step_num % args.sample_interval == 0:
                yield latent_input, current_targets, F1.clone().cpu().detach()

            if step == 0:
                loss_ini = loss_supervised

            logger.info(f"Loss_ini: {loss_ini}")

            if loss_supervised.max() < 0.5 * args.threshold_l:
                break

            if step_num == args.n_pix_step or step_num > step_threshold + 10:
                logger.info("Terminated by step_threshold")
                yield latent_input, current_targets, F1.clone().cpu().detach()
                break
        if step_num == args.n_pix_step or step_num > step_threshold + 10:
            break
        with torch.no_grad():
            # latent_input = torch.cat((latent_trainable,latent_untrainable),dim=1) # Why??? Is this some StyleGan feature?
            latent_input = latent_trainable
            # print("latent trainable shape",latent_trainable.shape)
            unet_output, F1 = model.forward_unet_features(
                latent_input,
                t,
                encoder_hidden_states=text_emb,
                layer_idx=args.unet_feature_idx,
                interp_res_h=args.sup_res_h,
                interp_res_w=args.sup_res_w,
            )

            current_feature = []
            for idx in range(point_pairs_number):
                current_feature.append(
                    interpolate_feature_patch_plus(
                        F1, current_targets[idx, :] + offset_matrix
                    )
                )
                loss_end[idx] = Loss_l1(
                    current_feature[idx], template_feature[idx].detach()
                )

            logger.info(f"Loss_end: {loss_end}")

        if d_remain.max() < args.res_ratio:
            step_threshold = step_num
        update_signs(
            sign_points,
            current_targets,
            target_points,
            loss_end,
            args.res_ratio,
            0.5 * args.threshold_l,
        )
        for idx in range(point_pairs_number):
            if sign_points[idx] == 1:
                lad = 1  # lad as in lambda in the equation
            else:
                lad = get_lad(loss_end[idx].detach(), args.aa, args.bb)
            template_feature[idx] = (
                lad * current_feature[idx].detach() + (1 - lad) * template_feature[idx]
            )

        # print("loss_ini: ",loss_ini)

        current_feature_map = F1.detach()


def drag_diffusion_update_gen(
    model, init_code, t, handle_points, target_points, mask, args
):
    assert len(handle_points) == len(
        target_points
    ), "number of handle point must equals target points"

    # positive prompt embedding
    text_emb = model.get_text_embeddings(args.prompt).detach()
    if args.guidance_scale > 1.0:
        unconditional_input = model.tokenizer(
            [args.neg_prompt], padding="max_length", max_length=77, return_tensors="pt"
        )
        unconditional_emb = model.text_encoder(
            unconditional_input.input_ids.to(text_emb.device)
        )[0].detach()
        text_emb = torch.cat([unconditional_emb, text_emb], dim=0)

    # the init output feature of unet
    with torch.no_grad():
        if args.guidance_scale > 1.0:
            model_inputs_0 = copy.deepcopy(torch.cat([init_code] * 2))
        else:
            model_inputs_0 = copy.deepcopy(init_code)
        unet_output, F0 = model.forward_unet_features(
            model_inputs_0,
            t,
            encoder_hidden_states=text_emb,
            layer_idx=args.unet_feature_idx,
            interp_res_h=args.sup_res_h,
            interp_res_w=args.sup_res_w,
        )
        if args.guidance_scale > 1.0:
            # strategy 1: discard the unconditional branch feature maps
            # F0 = F0[1].unsqueeze(dim=0)
            # strategy 2: concat pos and neg branch feature maps for motion-sup and point tracking
            # F0 = torch.cat([F0[0], F0[1]], dim=0).unsqueeze(dim=0)
            # strategy 3: concat pos and neg branch feature maps with guidance_scale consideration
            coef = args.guidance_scale / (2 * args.guidance_scale - 1.0)
            F0 = torch.cat([(1 - coef) * F0[0], coef * F0[1]], dim=0).unsqueeze(dim=0)

            unet_output_uncon, unet_output_con = unet_output.chunk(2, dim=0)
            unet_output = unet_output_uncon + args.guidance_scale * (
                unet_output_con - unet_output_uncon
            )
        x_prev_0, _ = model.step(unet_output, t, init_code)
        # init_code_orig = copy.deepcopy(init_code)

    # prepare optimizable init_code and optimizer
    init_code.requires_grad_(True)
    optimizer = torch.optim.Adam([init_code], lr=args.lr)

    # prepare for point tracking and background regularization
    handle_points_init = copy.deepcopy(handle_points)
    interp_mask = F.interpolate(
        mask, (init_code.shape[2], init_code.shape[3]), mode="nearest"
    )

    # prepare amp scaler for mixed-precision training
    scaler = torch.cuda.amp.GradScaler()
    for step_idx in range(args.n_pix_step):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            if args.guidance_scale > 1.0:
                model_inputs = init_code.repeat(2, 1, 1, 1)
            else:
                model_inputs = init_code
            unet_output, F1 = model.forward_unet_features(
                model_inputs,
                t,
                encoder_hidden_states=text_emb,
                layer_idx=args.unet_feature_idx,
                interp_res_h=args.sup_res_h,
                interp_res_w=args.sup_res_w,
            )
            if args.guidance_scale > 1.0:
                # strategy 1: discard the unconditional branch feature maps
                # F1 = F1[1].unsqueeze(dim=0)
                # strategy 2: concat positive and negative branch feature maps for motion-sup and point tracking
                # F1 = torch.cat([F1[0], F1[1]], dim=0).unsqueeze(dim=0)
                # strategy 3: concat pos and neg branch feature maps with guidance_scale consideration
                coef = args.guidance_scale / (2 * args.guidance_scale - 1.0)
                F1 = torch.cat([(1 - coef) * F1[0], coef * F1[1]], dim=0).unsqueeze(
                    dim=0
                )

                unet_output_uncon, unet_output_con = unet_output.chunk(2, dim=0)
                unet_output = unet_output_uncon + args.guidance_scale * (
                    unet_output_con - unet_output_uncon
                )
            x_prev_updated, _ = model.step(unet_output, t, init_code)

            # do point tracking to update handle points before computing motion supervision loss
            if step_idx != 0:
                handle_points = point_tracking(
                    F0, F1, handle_points, handle_points_init, args
                )
                print("new handle points", handle_points)

            # break if all handle points have reached the targets
            if check_handle_reach_target(handle_points, target_points):
                break

            loss = 0.0
            for i in range(len(handle_points)):
                pi, ti = handle_points[i], target_points[i]
                # skip if the distance between target and source is less than 1
                if (ti - pi).norm() < 2.0:
                    continue

                di = (ti - pi) / (ti - pi).norm()

                # motion supervision
                f0_patch = F1[
                    :,
                    :,
                    int(pi[0]) - args.r_m : int(pi[0]) + args.r_m + 1,
                    int(pi[1]) - args.r_m : int(pi[1]) + args.r_m + 1,
                ].detach()
                f1_patch = interpolate_feature_patch(
                    F1, pi[0] + di[0], pi[1] + di[1], args.r_m
                )
                loss += ((2 * args.r_m + 1) ** 2) * F.l1_loss(f0_patch, f1_patch)

            # masked region must stay unchanged
            loss += (
                args.lam
                * ((x_prev_updated - x_prev_0) * (1.0 - interp_mask)).abs().sum()
            )
            # loss += args.lam * ((init_code_orig - init_code)*(1.0-interp_mask)).abs().sum()
            print("loss total=%f" % (loss.item()))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return init_code
