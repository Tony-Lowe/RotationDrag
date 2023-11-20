import torch
import PIL
from copy import deepcopy
import math
import numpy as np
from typing import List, Tuple
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA


def get_ellipse_coords(point, radius: int = 5):
    """
    Returns the coordinates of an ellipse centered at the given point.

    Args:
        point (Tuple[int, int]): The center point of the ellipse.
        radius (int): The radius of the ellipse.

    Returns:
        A tuple containing the coordinates of the ellipse in the format (x_min, y_min, x_max, y_max).
    """
    center = point
    return (
        center[0] - radius,
        center[1] - radius,
        center[0] + radius,
        center[1] + radius,
    )


def draw_handle_target_points(
    img: PIL.Image.Image,
    handle_points: List,
    target_points: List,
    radius: int = 5,
    color="red",
):
    """
    Draws handle and target points with arrow pointing towards the target point.

    Args:
        img (PIL.Image.Image): The image to draw on.
        handle_points torch.Tensor: A list of handle [x,y] points.
        target_points torch.Tensor: A list of target [x,y] points.
        radius (int): The radius of the handle and target points.
    """
    if not isinstance(img, PIL.Image.Image):
        img = PIL.Image.fromarray(img)

    if len(handle_points) == len(target_points) + 1:
        target_points = deepcopy(target_points) + [None]

    draw = PIL.ImageDraw.Draw(img)
    for handle_point, target_point in zip(handle_points, target_points):
        handle_point = [handle_point[1], handle_point[0]]
        # Draw the handle point
        handle_coords = get_ellipse_coords(handle_point, radius)
        draw.ellipse(handle_coords, fill=color)

        if target_point is not None:
            target_point = [target_point[1], target_point[0]]
            # Draw the target point
            target_coords = get_ellipse_coords(target_point, radius)
            draw.ellipse(target_coords, fill="blue")

            # Draw arrow head
            arrow_head_length = radius * 1.5

            # Compute the direction vector of the line
            dx = target_point[0] - handle_point[0]
            dy = target_point[1] - handle_point[1]
            angle = math.atan2(dy, dx)

            # Shorten the target point by the length of the arrowhead
            shortened_target_point = (
                target_point[0] - arrow_head_length * math.cos(angle),
                target_point[1] - arrow_head_length * math.sin(angle),
            )

            # Draw the arrow (main line)
            draw.line(
                [tuple(handle_point), shortened_target_point],
                fill="white",
                width=int(0.8 * radius),
            )

            # Compute the points for the arrowhead
            arrow_point1 = (
                target_point[0] - arrow_head_length * math.cos(angle - math.pi / 6),
                target_point[1] - arrow_head_length * math.sin(angle - math.pi / 6),
            )

            arrow_point2 = (
                target_point[0] - arrow_head_length * math.cos(angle + math.pi / 6),
                target_point[1] - arrow_head_length * math.sin(angle + math.pi / 6),
            )

            # Draw the arrowhead
            draw.polygon(
                [tuple(target_point), arrow_point1, arrow_point2], fill="white"
            )
    return np.array(img)


def compute_featruemap(desc):
    N, C, H, W = desc.shape
    desc_reshaped = desc.cpu().view(C, -1).permute(1, 0)
    n_components = 3
    pca = PCA(n_components=n_components)
    reduced_desc = pca.fit_transform(desc_reshaped)
    # Stack the reduced channel data back into the original shape
    heatmap = reduced_desc.reshape(H, W, n_components)
    # Visualize the reduced feature map (example for a single channel)
    # heatmap = heatmap.reshape(H, W)
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    return heatmap


def draw_featuremap(desc: torch.Tensor) -> plt.Figure:
    """
    draw point correspondences on images.
    :param desc: image's descriptor.
    :param image1: a PIL image.
    :return: a figure of images with all marked keypoints.
    """

    heatmap = compute_featruemap(desc)

    plt.autoscale(tight=True)

    fig, ax = plt.subplots()
    plt.axis("off")
    ax.axis("off")
    ax.imshow(heatmap, cmap="viridis")
    ax.set_xlim(0, desc.shape[3])
    ax.set_ylim(desc.shape[2], 0)

    fig.subplots_adjust(left=None, bottom=None, right=None, wspace=0, hspace=None)
    return fig
