import torch
import math

import matplotlib.pyplot as plt
import numpy as np

from torch_rotation import rotate_three_pass

if __name__ == '__main__':
    img = plt.imread('data/cat.jpg')
    img = img.astype(np.float32) / 255
    h, w = img.shape[:2]

    I = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)

    angle = 22.5 * math.pi / 180

    I_rot = rotate_three_pass(I, angle)

    img_rot = I_rot.squeeze(0).permute(1,2,0).numpy()

    plt.figure()
    plt.imshow(img_rot)
    plt.show()
