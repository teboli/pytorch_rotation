import torch
import math

def rotate_pytorch(image, angle=0, mode='bilinear', padding_mode='zeros'):
    shape = image.shape
    amat = torch.zeros(shape[0], 2, 3, device=image.device)
    if isinstance(angle, float):
        amat[:, 0, 0] = math.cos(angle)
        amat[:, 0, 1] = -math.sin(angle)
        amat[:, 1, 0] = math.sin(angle)
        amat[:, 1, 1] = math.cos(angle)
    else:
        amat[:, 0, 0] = torch.cos(angle)
        amat[:, 0, 1] = -torch.sin(angle)
        amat[:, 1, 0] = torch.sin(angle)
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
    import time
    import matplotlib.pyplot as plt
    import numpy as np

    from torch_rotation import rotate_three_pass

    img = plt.imread('data/cat.jpg')
    img = img.astype(np.float32) / 255
    h, w = img.shape[:2]

    I = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
    I = torch.nn.functional.pad(I, [w//2, w//2, h//2, h//2])

    n_rot = 16

    angle = 360 / n_rot * np.pi / 180

    I_torch_bilinear = I.clone().detach()
    I_torch_bicubic = I.clone().detach()
    I_three_taps = I.clone().detach()

    times_torch_bilinear = np.zeros(n_rot)
    for n in range(n_rot):
        start = time.time()
        I_torch_bilinear = rotate_pytorch(I_torch_bilinear, angle, mode='bilinear')
        times_torch_bilinear[n] = time.time() - start
    
    times_torch_bicubic = np.zeros(n_rot)
    for n in range(n_rot):
        start = time.time()
        I_torch_bicubic = rotate_pytorch(I_torch_bicubic, angle, mode='bicubic')
        times_torch_bicubic[n] = time.time() - start

    times_three_taps = np.zeros(n_rot)
    for n in range(n_rot):
        start = time.time()
        I_three_taps = rotate_three_pass(I_three_taps, angle)
        times_three_taps[n] = time.time() - start

    I = I[..., h//2:-h//2, w//2:-w//2]
    I_torch_bilinear = I_torch_bilinear[..., h//2:-h//2, w//2:-w//2]
    I_torch_bicubic = I_torch_bicubic[..., h//2:-h//2, w//2:-w//2]
    I_three_taps = I_three_taps[..., h//2:-h//2, w//2:-w//2]

    img = I.squeeze(0).permute(1,2,0).numpy()
    img_torch_bilinear = I_torch_bilinear.squeeze(0).permute(1,2,0).numpy()
    img_torch_bicubic = I_torch_bicubic.squeeze(0).permute(1,2,0).numpy()
    img_three_taps = I_three_taps.squeeze(0).permute(1,2,0).numpy()

    def psnr(im1, im2):
        mse = np.mean((im1 - im2)**2)
        return 10 * np.log10(1 / mse)

    print('Pytorch (bilinear):    %f +/- %f' % (np.mean(times_torch_bilinear), np.std(times_torch_bilinear)))
    print('Pytorch (bicubic):     %f +/- %f' % (np.mean(times_torch_bicubic), np.std(times_torch_bicubic)))
    print('Three taps:            %f +/- %f' % (np.mean(times_three_taps), np.std(times_three_taps)))

    plt.figure(figsize=(10,10))
    plt.subplot(2,2,1)
    plt.imshow(img)
    plt.title("Original")
    plt.subplot(2,2,2)
    plt.imshow(img_three_taps)
    plt.title("Three taps: PSNR = %2.2f dB" % psnr(img, img_three_taps))
    plt.subplot(2,2,3)
    plt.imshow(img_torch_bilinear)
    plt.title("Pytorch (bilinear): PSNR = %2.2f dB" % psnr(img, img_torch_bilinear))
    plt.subplot(2,2,4)
    plt.imshow(img_torch_bicubic)
    plt.title("Pytorch (bicubic): PSNR = %2.2f dB" % psnr(img, img_torch_bicubic))
    plt.tight_layout()
    plt.show()
