import math

import torch


def to_keypoints(tensor, scale, translation, default_scale=4):
    """
    Convert predicted heatmap to keypoint locations

    Parameters
    ----------
    tensor : torch.tensor, shape (b, k, h, w)
        b - batch size
        k - number of keypoints
        h - height
        w - width
        Heatmap tensor that should be converted to keypoint locations
    scale : torch.tensor (b, )
        b - batch size
        Scale of the original image
    translation : array, shape (b, 1, 2)
        Trranslation of the original image
    default_scale : float
        Difference between input and output size

    Returns
    -------
    keypoints : torch.tensor, shape (b, k, 2)
        b - batch size
        k - number of keypoints
        Keypoints locations
    """
    batch_size, n_keypoints, height, width = tensor.shape

    flat_tensor = tensor.reshape((batch_size, n_keypoints, -1))
    values, idx = torch.max(flat_tensor, dim=2)

    keypoints = torch.zeros(batch_size, n_keypoints, 2, device=tensor.device)
    keypoints[:, :, 0] = idx % width
    keypoints[:, :, 1] = idx // width

    # Post-processing step to improve performance at tight PCK thresholds
    # https://github.com/microsoft/human-pose-estimation.pytorch/issues/47
    for n in range(keypoints.size(0)):
        for p in range(keypoints.size(1)):
            heatmap = tensor[n, p]
            px = int(math.floor(keypoints[n, p, 0] + 0.5))
            py = int(math.floor(keypoints[n, p, 1] + 0.5))
            if 1 < px < width - 1 and 1 < py < height - 1:
                diff = torch.tensor(
                    [
                        heatmap[py, px + 1] - heatmap[py, px - 1],
                        heatmap[py + 1, px] - heatmap[py - 1, px],
                    ],
                    device=tensor.device,
                )
                keypoints[n, p] += torch.sign(diff) * 0.25
    keypoints += 0.5

    keypoints *= scale * default_scale
    keypoints += translation

    return keypoints
