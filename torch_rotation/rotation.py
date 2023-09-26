from typing import Union

import math

import torch
import torch.fft as fft
import torch.nn.functional as F

xyz_to_dims = {'x': (2,3,4), 'y': (3,4,2), 'z': (4,2,3)}


def rotate_three_pass(input: torch.Tensor, 
                      angle: Union[float, int, torch.Tensor] = 0.0, 
                      N: int = -1,
                      padding_mode: str = 'constant',
                      value: float = 0.0,
                      order_3d: str = 'xyz',
                      ) -> torch.Tensor:
    """
    Implementation of the three tap rotation by ...

    Args:
        input: (B,C,H,W) or (B,C,D,H,W) the image/volume to be rotated.
        angle: float or (B,) pr (B,3) the angles in radians for rotation. If (B,3) format,
                it is an angle per 3D axis and it is sampled according to 'order_3d'.
        N: int, the order of interpolation. -1 is sinc (FFT_based) interpolation. It is
                recommaned to leave this setting unchanged for optimal accuracy.
        mode: str, the type of border handling.
        value: float, in the case mode == 'constant', the value to fill the holes.
        order_3d: the order of axes around which we sequentially achieve a 2D rotation.

    Return:
        input_rotated: (B,C,H,W) or (B,C,L,H,W), the rotated image/volume.
    """
    assert(input.ndim == 4 or input.ndim == 5)
    assert(N >= -1)  # Either 'infinite' or positive order.
    assert(padding_mode in ['constant', 'replicate', 'reflect', 'circular'])
    assert(order_3d in ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx'])

    if input.ndim == 4:
        return rotate2d_three_pass(input, angle, N, padding_mode, value)
    else:
        ## TODO: So far, we handle only the basic 3D rotation: 
        ## https://en.wikipedia.org/wiki/Rotation_matrix#Basic_3D_rotations
        ## In the future we plan to expand to the general 3D case:
        ## https://en.wikipedia.org/wiki/Rotation_matrix#General_3D_rotations
        if isinstance(angle, torch.Tensor):
            if angle.ndim > 1:
                raise NotImplementedError("For the moment, only handle basic 3D rotations. \
                                          Valid angle variables or float and (B,)-sized tensors.")
        B, C, L, H, W = input.shape
        I = input
        I = rotate2d_three_pass(I, angle, N, padding_mode, value)  # rotation wrt x
        I = I.permute(0, 1, 3, 4, 2).contiguous()  # (B, C, H, W, L)
        I = rotate2d_three_pass(I, angle, N, padding_mode, value)  # rotation wrt y
        I = I.permute(0, 1, 3, 4, 2).contiguous()  # (B, C, W, L, H)
        I = rotate2d_three_pass(I, angle, N, padding_mode, value)  # rotation wrt z
        I = I.permute(0, 1, 3, 4, 2).contiguous()  # (B, C, L, H, W)    
        return I

def rotate2d_three_pass(input: torch.Tensor, 
                        angle: Union[float, torch.Tensor] = 0.0, 
                        N: int = -1,
                        padding_mode: str = 'constant',
                        value: float = 0.0,) -> torch.Tensor:
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



def rotate2d_three_pass_fft(I: torch.Tensor, 
                            theta: Union[float, torch.Tensor], 
                            padding_mode: str, 
                            value: float
                            ) -> torch.Tensor:
    B = I.shape[0]
    C = I.shape[1:-2]
    H, W = I.shape[-2:]
    device = I.device

    if padding_mode != 'circular':
        pad_h = H // 2
        pad_w = W // 2
        I = I.view(B, -1, H, W)  # for padding, we need a 4D tensor.
        I = torch.nn.functional.pad(I, [pad_w, pad_w, pad_h, pad_h], \
                                    mode=padding_mode, value=value)
        H, W = I.shape[-2:]
        I = I.view(B, *C, H, W)  # going back to the original shape.

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
