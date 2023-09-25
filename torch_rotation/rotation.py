from typing import Union

import math

import torch
import torch.fft as fft
import torch.nn.functional as F


def rotate_three_pass(input: torch.Tensor, 
                      angle: Union[float, torch.Tensor] = 0.0, 
                      N: int = -1,
                      padding_mode: str = 'constant',
                      value: float = 0.0
                      ) -> torch.Tensor:
    """
    Implementation of the three tap rotation by ...

    Args:
        input: (B,C,H,W) or (B,C,D,H,W) the image/volume to be rotated.
        angle: float or (B,) the angles in radians for rotation.
        N: int, the order of interpolation. -1 is sinc interpolation.
        mode: str, the type of border handling.
        value: float, in the case mode == 'constant', the value to fill the holes.

    Return:
        input_rotated: (B,C,H,W), the rotated images.
    """
    assert(input.ndim == 4 or input.ndim == 5)
    assert(N >= -1)  # Either 'infinite' or positive order.
    assert(padding_mode in ['constant', 'replicate', 'reflect', 'circular'])

    if input.ndim == 4:
        ## If N = -1, we do FFT-based translations.
        if N == -1:
            return rotate2d_three_pass_fft(
                I=input,
                theta=angle,
                padding_mode=padding_mode,
                value=value
            )
        ## Otherwise, do polynomial interpolation in the spatial domain.
        else:
            if N == 1:
                return rotate2d_three_pass_spatial(
                    I=input,
                    theta=angle,
                    mode='bilinear',
                    padding_mode=padding_mode,
                    value=value
                )
            elif N == 3:
                return rotate2d_three_pass_spatial(
                    I=input,
                    theta=angle,
                    mode='bicubic',
                    padding_mode=padding_mode,
                    value=value
                )
            else:
                raise NotImplementedError()
    else:
        raise NotImplementedError()




def rotate2d_three_pass_fft(I: torch.Tensor, 
                            theta: Union[float, torch.Tensor], 
                            padding_mode: str, 
                            value: float
                            ) -> torch.Tensor:
    B, C, H, W = I.shape
    device = I.device

    if padding_mode != 'circular':
        pad_h = H // 2
        pad_w = W // 2
        I = torch.nn.functional.pad(I, [pad_w, pad_w, pad_h, pad_h], \
                                    mode=padding_mode, value=value)
        H, W = I.shape[-2:]

    ## Handling odd case by adding + 1 because after ifftshift,
    ## if the size of the vector is odd, we have -1 as first frequency
    ## and thus it creates a mismatch with the frequency of the
    ## transform of the image.
    offsetx = 1 * (H % 2 == 1)
    offsety = 1 * (W % 2 == 1)
    fx = fft.ifftshift(torch.arange(-H//2+offsetx, H//2+offsetx, device=device))[:, None]
    fy = fft.ifftshift(torch.arange(-W//2+offsety, W//2+offsety, device=device))[None, :]

    ## Row or col-specific shift.
    if isinstance(theta, float):
        a = math.tan(theta/2) * torch.arange(H, device=device).add(-H//2)[:, None]
        b = -math.sin(theta) * torch.arange(W, device=device).add(-W//2)[None, :]
    else:
        a = torch.tan(theta/2, device=device).view(B,1,1,1) * \
            (torch.arange(H, device=device).add(-H//2).view(1,1,H,1))
        b = -torch.sin(theta, device=device).view(B,1,1,1) * \
            (torch.arange(W, device=device).add(-W//2).view(1,1,1,W))
    
    ## FFT domain shear.
    shear_fft_v = (-2j * math.pi * fx * b / H).exp()
    shear_fft_h = (-2j * math.pi * fy * a / W).exp()

    ## Horizontal shear.
    I_fft = fft.fft(I, dim=-1) * shear_fft_h
    I = fft.ifft(I_fft, dim=-1).real.clamp(0, 1)

    ## Vertical shear.
    I_fft = fft.fft(I, dim=-2) * shear_fft_v
    I = fft.ifft(I_fft, dim=-2).real.clamp(0, 1)

    ## Horizontal shear.
    I_fft = fft.fft(I, dim=-1) * shear_fft_h
    I = fft.ifft(I_fft, dim=-1).real.clamp(0, 1)

    if padding_mode == 'circular':
        return I
    else:
        return I[..., pad_h:-pad_h, pad_w:-pad_w]


def rotate2d_three_pass_spatial(I: torch.Tensor, 
                                theta: Union[float, torch.Tensor], 
                                mode: str,
                                padding_mode: str,
                                value: float) -> torch.Tensor:
    assert(mode in ['bilinear', 'bicubic'])

    B, C, H, W = I.shape
    device = I.device

    if padding_mode != 'circular':
        pad_h = H // 2
        pad_w = W // 2
        I = torch.nn.functional.pad(I, [pad_w, pad_w, pad_h, pad_h], 
                                    mode=padding_mode, value=value)
        H, W = I.shape[-2:]
    
    ## Spatial domain shear grids.
    shear_h = torch.zeros(B,2,3, device=device)
    shear_h[:, 0, 0] = shear_h[:, 1, 1] = 1
    
    shear_v = shear_h.detach().clone()

    if isinstance(theta, float):
        shear_h[:, 0, 1] = -math.tan(theta/2)
        shear_v[:, 1, 0] = math.sin(theta)
    else:
        shear_h[:, 0, 1] = -torch.tan(theta/2)
        shear_v[:, 1, 0] = torch.sin(theta)

    ## TODO: fix the scaling induces by affine_grid.
    shear_h_grid = torch.nn.functional.affine_grid(shear_h, (B,C,H,W), align_corners=False)
    shear_v_grid = torch.nn.functional.affine_grid(shear_v, (B,C,H,W), align_corners=False)

    ## Horizontal shear.
    I = torch.nn.functional.grid_sample(
        input=I,
        grid=shear_h_grid,
        mode=mode,
        align_corners=False
    ).clamp(0, 1)
    I = torch.nn.functional.grid_sample(
        input=I,
        grid=shear_v_grid,
        mode=mode,
        align_corners=False
    ).clamp(0, 1)
    I = torch.nn.functional.grid_sample(
        input=I,
        grid=shear_h_grid,
        mode=mode,
        align_corners=False
    ).clamp(0, 1)

    if padding_mode == 'circular':
        return I
    else:
        return I[..., pad_h:-pad_h, pad_w:-pad_w]
