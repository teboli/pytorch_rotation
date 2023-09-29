import torch
import math

import matplotlib.pyplot as plt
import numpy as np

from torch_rotation import rotate_three_pass

def rotate_pytorch(image, angle=0, mode='bilinear', padding_mode='zeros'):
    shape = image.shape
    amat = torch.zeros(shape[0], 2, 3, device=image.device)
    if isinstance(angle, float):
        amat[:, 0, 0] = math.cos(angle)
        amat[:, 0, 1] = -math.sin(angle) * shape[-2] / shape[-1]  # (h/w)
        amat[:, 1, 0] = math.sin(angle) * shape[-1] / shape[-2]  # (w/h)
        amat[:, 1, 1] = math.cos(angle)
    else:
        amat[:, 0, 0] = torch.cos(angle)
        amat[:, 0, 1] = -torch.sin(angle) * shape[-2] / shape[-1]  # (h/w)
        amat[:, 1, 0] = torch.sin(angle) * shape[-1] / shape[-2]   # (h/w)
        amat[:, 1, 1] = torch.cos(angle)

    grid = torch.nn.functional.affine_grid(
        theta=amat,
        size=shape,
        align_corners=False
    )
    image_rotated = torch.nn.functional.grid_sample(
        input=image,
        grid=grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=False
    )

    return image_rotated.clamp(0, 1)


if __name__ == '__main__':
    img = plt.imread('data/cat.jpg')
    img = img.astype(np.float32) / 255
    h, w = img.shape[:2]

    I = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
    # I = torch.rand(10, 3, 50, 50, 50)

    angle = 135 * math.pi / 180

    I_rot = rotate_three_pass(I, angle)

    I_rot_pt = rotate_pytorch(I, angle, mode="bicubic")

    img_rot = I_rot.squeeze(0).permute(1,2,0).numpy()
    img_rot_pt = I_rot_pt.squeeze(0).permute(1,2,0).numpy()

    err = np.abs(img_rot - img_rot_pt)

    plt.figure(figsize=(10,4))
    plt.subplot(1,3,1)
    plt.imshow(img_rot)
    plt.title("Three pass")
    plt.subplot(1,3,2)
    plt.imshow(img_rot_pt)
    plt.title("Bicubic")
    plt.subplot(1,3,3)
    plt.imshow(err / err.max())
    plt.title("Difference")
    plt.show()
