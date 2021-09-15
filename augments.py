import random
import numpy as np
import torch
import torch.nn.functional as F
from scipy.fftpack import dct, idct


def apply_augment(im1, im2):
    augs = ["blend", "rgb", "mixup", "cutout", "cutmix", "cutmixup", "cutblur"]
    probs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    alphas = [0.6, 1.0, 1.2, 0.001, 0.7, 0.7, 0.7]
    aux_prob, aux_alpha = 1.0, 1.2
    # mix_p = None
    mix_p = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4]
    # mix_p = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4]  # RealSR
    # probs = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]  # JPEG
    # probs = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]  # Carn

    idx = np.random.choice(len(augs), p=mix_p)
    aug = augs[idx]
    prob = float(probs[idx])
    alpha = float(alphas[idx])
    mask = None

    if aug == "none":
        im1_aug, im2_aug = im1.clone(), im2.clone()
    elif aug == "blend":
        im1_aug, im2_aug = blend(
            im1.clone(), im2.clone(),
            prob=prob, alpha=alpha
        )
    elif aug == "mixup":
        im1_aug, im2_aug, = mixup(
            im1.clone(), im2.clone(),
            prob=prob, alpha=alpha,
        )
    elif aug == "cutout":
        im1_aug, im2_aug, mask, _ = cutout(
            im1.clone(), im2.clone(),
            prob=prob, alpha=alpha
        )
    elif aug == "cutmix":
        im1_aug, im2_aug = cutmix(
            im1.clone(), im2.clone(),
            prob=prob, alpha=alpha,
        )
    elif aug == "cutmixup":
        im1_aug, im2_aug = cutmixup(
            im1.clone(), im2.clone(),
            mixup_prob=aux_prob, mixup_alpha=aux_alpha,
            cutmix_prob=prob, cutmix_alpha=alpha,
        )
    elif aug == "cutblur":
        im1_aug, im2_aug = cutblur(
            im1.clone(), im2.clone(),
            prob=prob, alpha=alpha
        )
    elif aug == "rgb":
        im1_aug, im2_aug = rgb(
            im1.clone(), im2.clone(),
            prob=prob
        )
    else:
        raise ValueError("{} is not invalid.".format(aug))

    return im1_aug, im2_aug, mask, aug


def blend(im1, im2, prob=1.0, alpha=0.6):
    """blend aug"""
    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2

    c = torch.empty((im2.size(0), 3, 1, 1), device=im2.device).uniform_(0, 255)
    rim2 = c.repeat((1, 1, im2.size(2), im2.size(3)))
    rim1 = c.repeat((1, 1, im1.size(2), im1.size(3)))

    v = np.random.uniform(alpha, 1)
    im1 = v * im1 + (1 - v) * rim1
    im2 = v * im2 + (1 - v) * rim2

    return im1, im2


def mixup(im1, im2, prob=1.0, alpha=1.2):
    """mixup aug"""
    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2

    v = np.random.beta(alpha, alpha)
    r_index = torch.randperm(im1.size(0)).to(im2.device)

    im1 = v * im1 + (1 - v) * im1[r_index, :]
    im2 = v * im2 + (1 - v) * im2[r_index, :]
    return im1, im2


def _cutmix(im2, prob=1.0, alpha=1.0):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return None

    cut_ratio = np.random.randn() * 0.01 + alpha

    h, w = im2.size(2), im2.size(3)
    ch, cw = np.int(h * cut_ratio), np.int(w * cut_ratio)

    fcy = np.random.randint(0, h - ch + 1)
    fcx = np.random.randint(0, w - cw + 1)
    tcy, tcx = fcy, fcx
    rindex = torch.randperm(im2.size(0)).to(im2.device)

    return {
        "rindex": rindex, "ch": ch, "cw": cw,
        "tcy": tcy, "tcx": tcx, "fcy": fcy, "fcx": fcx,
    }


def cutmix(im1, im2, prob=1.0, alpha=1.0):
    """cutmix aug"""
    c = _cutmix(im2, prob, alpha)
    if c is None:
        return im1, im2

    scale = im1.size(2) // im2.size(2)
    rindex, ch, cw = c["rindex"], c["ch"], c["cw"]
    tcy, tcx, fcy, fcx = c["tcy"], c["tcx"], c["fcy"], c["fcx"]

    hch, hcw = ch * scale, cw * scale
    hfcy, hfcx, htcy, htcx = fcy * scale, fcx * scale, tcy * scale, tcx * scale

    im2[..., tcy:tcy + ch, tcx:tcx + cw] = im2[rindex, :, fcy:fcy + ch, fcx:fcx + cw]
    im1[..., htcy:htcy + hch, htcx:htcx + hcw] = im1[rindex, :, hfcy:hfcy + hch, hfcx:hfcx + hcw]

    return im1, im2


def cutmixup(im1, im2,
        mixup_prob=1.0, mixup_alpha=1.0,
        cutmix_prob=1.0, cutmix_alpha=1.0
):
    """cut mixup aug"""
    c = _cutmix(im2, cutmix_prob, cutmix_alpha)
    if c is None:
        return im1, im2

    scale = im1.size(2) // im2.size(2)
    rindex, ch, cw = c["rindex"], c["ch"], c["cw"]
    tcy, tcx, fcy, fcx = c["tcy"], c["tcx"], c["fcy"], c["fcx"]

    hch, hcw = ch * scale, cw * scale
    hfcy, hfcx, htcy, htcx = fcy * scale, fcx * scale, tcy * scale, tcx * scale

    v = np.random.beta(mixup_alpha, mixup_alpha)
    if mixup_alpha <= 0 or np.random.rand(1) >= mixup_prob:
        im2_aug = im2[rindex, :]
        im1_aug = im1[rindex, :]

    else:
        im2_aug = v * im2 + (1 - v) * im2[rindex, :]
        im1_aug = v * im1 + (1 - v) * im1[rindex, :]

    # apply mixup to inside or outside
    if np.random.random() > 0.5:
        im2[..., tcy:tcy + ch, tcx:tcx + cw] = im2_aug[..., fcy:fcy + ch, fcx:fcx + cw]
        im1[..., htcy:htcy + hch, htcx:htcx + hcw] = im1_aug[..., hfcy:hfcy + hch, hfcx:hfcx + hcw]
    else:
        im2_aug[..., tcy:tcy + ch, tcx:tcx + cw] = im2[..., fcy:fcy + ch, fcx:fcx + cw]
        im1_aug[..., htcy:htcy + hch, htcx:htcx + hcw] = im1[..., hfcy:hfcy + hch, hfcx:hfcx + hcw]
        im2, im1 = im2_aug, im1_aug

    return im1, im2


def cutblur(im1, im2, prob=1.0, alpha=0.5, train=True):
    ''' im1: HR, im2: LR '''
    # match the resolution of (LR, HR) due to CutBlur
    im1 = im1.unsqueeze(0)
    im2 = im2.unsqueeze(0)
    if im1.size() != im2.size():
        scale = im1.size(2) // im2.size(2)
        im2 = F.interpolate(im2, scale_factor=scale, mode="nearest")
    
    if im1.size() != im2.size():
        raise ValueError("im1 and im2 have to be the same resolution.")

    if train is not True:
        im1 = im1.squeeze()
        im2 = im2.squeeze()
        return im1, im2

    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2

    cut_ratio = np.random.randn() * 0.01 + alpha

    h, w = im2.size(2), im2.size(3)
    ch, cw = np.int(h * cut_ratio), np.int(w * cut_ratio)
    cy = np.random.randint(0, h - ch + 1)
    cx = np.random.randint(0, w - cw + 1)

    # apply CutBlur to inside or outside
    if np.random.random() > 0.5:
        im2[..., cy:cy + ch, cx:cx + cw] = im1[..., cy:cy + ch, cx:cx + cw]
    else:
        im2_aug = im1.clone()
        im2_aug[..., cy:cy + ch, cx:cx + cw] = im2[..., cy:cy + ch, cx:cx + cw]
        im2 = im2_aug
    
    # im2 = F.interpolate(im2, scale_factor=1/scale, mode="nearest")

    im1 = im1.squeeze()
    im2 = im2.squeeze()

    return im1, im2


def cutout(im1, im2, prob=1.0, alpha=0.1):
    scale = im1.size(2) // im2.size(2)
    fsize = (im2.size(0), 1) + im2.size()[2:]

    if alpha <= 0 or np.random.rand(1) >= prob:
        fim2 = np.ones(fsize)
        fim2 = torch.tensor(fim2, dtype=torch.float, device=im2.device)
        fim1 = F.interpolate(fim2, scale_factor=scale, mode="nearest")
        return im1, im2, fim1, fim2

    fim2 = np.random.choice([0.0, 1.0], size=fsize, p=[alpha, 1 - alpha])
    fim2 = torch.tensor(fim2, dtype=torch.float, device=im2.device)
    fim1 = F.interpolate(fim2, scale_factor=scale, mode="nearest")

    im2 *= fim2

    return im1, im2, fim1, fim2


def rgb(im1, im2, prob=1.0):
    if np.random.rand(1) >= prob:
        return im1, im2

    perm = np.random.permutation(3)
    im1 = im1[:, perm]
    im2 = im2[:, perm]

    return im1, im2


# Code for SFM
def fully_random_drop_mask(w=256, h=256, radius=0, p=0.5):
    """random drop mask"""
    mask_random = np.random.rand(w, h)  # random [0-1]
    mask = np.ones((w, h))
    mask[mask_random > p] = 0  # random drop
    return mask


def circular_random_drop_mask(w=256, h=256, SFM_center_radius_perc=-1, SFM_center_sigma_perc=0.05):
    '''
    (w,h) are the dimensions of the mask

    IF (SFM_center_radius_perc=-1)
        the masked regions are selected randomly in circular shape, with the maximum at "radius"
        when "radius" is 0, it is set to the max default value
    ELSE
        the masked regions are always centered at "SFM_center_radius_perc*radius", and stretch inwards and
        outwards with a Gaussian probability, with sigma=SFM_center_sigma_perc*radius
    '''

    radius = np.sqrt(w * w + h * h)  # radius
    SFM_center_sigma = SFM_center_sigma_perc * radius
    SFM_center_radius = SFM_center_radius_perc * radius

    X, Y = np.meshgrid(np.linspace(0, h - 1, h), np.linspace(0, w - 1, w))
    D = np.sqrt(X * X + Y * Y)

    # random SFM (SFM_center_radius 0) vs SFM around a center of given distance
    if SFM_center_radius_perc == -1:
        a1 = random.random() * radius
        a2 = random.random() * radius
        if (a1 > a2):
            tmp = a2;
            a2 = a1;
            a1 = tmp
        mask = np.ones((w, h))
        mask[(D > a1) & (D < a2)] = 0

    else:
        if SFM_center_radius > radius or SFM_center_radius < 0:
            raise Exception('SFM_center_radius out of bounds.')

        a1 = random.gauss(0, SFM_center_sigma)
        a2 = random.gauss(0, SFM_center_sigma)

        a1 = abs(a1)
        a2 = abs(a2)

        a1 = SFM_center_radius - a1
        a2 = SFM_center_radius + a2

        mask = np.ones((w, h))
        mask[(D > a1) & (D < a2)] = 0

    return mask


def random_drop(img, mode=1, SFM_center_radius_perc=-1, SFM_center_sigma_perc=0.05):
    ''' mode=0:fully random drop, mode=1: circular random drop, mode=2 sweeping mode

        **sweeping mode**:
            SFM_center_radius_perc: determines the center of the band to be erased
                                    it is a percentage of the max radius
            SFM_center_sigma_perc:  determines the sigma for the width of the band
                                    sigma=radius*SFM_center_sigma_perc
    '''

    (c, w, h) = np.shape(img)
    if mode == 0:
        mask = fully_random_drop_mask(w, h)
    if mode == 1:
        mask = circular_random_drop_mask(w, h)
    if mode == 2:
        mask = circular_random_drop_mask(w, h, SFM_center_radius_perc, SFM_center_sigma_perc)

    if c == 3:
        img0_dct = dct(dct(img[0, :, :], axis=0, norm='ortho'), axis=1, norm='ortho')
        img1_dct = dct(dct(img[1, :, :], axis=0, norm='ortho'), axis=1, norm='ortho')
        img2_dct = dct(dct(img[2, :, :], axis=0, norm='ortho'), axis=1, norm='ortho')
        img0_dct = img0_dct * mask
        img1_dct = img1_dct * mask
        img2_dct = img2_dct * mask
        img[0, :, :] = idct(idct(img0_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
        img[1, :, :] = idct(idct(img1_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
        img[2, :, :] = idct(idct(img2_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
    elif c == 1:
        img_dct = dct(dct(img[0, :, :], axis=0, norm='ortho'), axis=1, norm='ortho')
        img_dct = img_dct * mask
        img[0, :, :] = idct(idct(img_dct, axis=0, norm='ortho'), axis=1, norm='ortho')

    return (img, mask)


def get_mask_low_high(w=256, h=256, radius_perc=-1, mask_mode=-1):
    '''
    (w,h) are the dimensions of the mask
    if mask_mode==1 low frequencies are cut off
    if mask_mode==2 high frequencies are cut off

    returns a binary mask of low or of high frequencies, cut-off at radius_perc*radius
    '''

    if radius_perc < 0:
        raise Exception('radius_perc must be positive.')

    radius = np.sqrt(w * w + h * h)
    center_radius = radius_perc * radius

    X, Y = np.meshgrid(np.linspace(0, h - 1, h), np.linspace(0, w - 1, w))
    D = np.sqrt(X * X + Y * Y)

    if mask_mode == 1:
        a1 = 0
        a2 = center_radius
    elif mask_mode == 2:
        a1 = center_radius
        a2 = radius
    else:
        raise Exception('mask_mode must be 1 or 2.')

    mask = np.ones((w, h))
    mask[(D >= a1) & (D <= a2)] = 0

    return mask


def test_sfm():
    DCT_DOR = 0.5  # for a 50% SFM rate (DCT dropout rate)
    image_SFM = np.zeros(image.size(), dtype='float32')  # (c,h,w)
    dct_bool = np.random.choice([1, 0], size=(image.size()[0],), p=[DCT_DOR, 1 - DCT_DOR])  # check if drop for image
    for img_idx in range(image.size()[0]):
        if dct_bool[img_idx] == 1:
            # random_drop settings for targeted mode (used in denoising experiments):
            image_numpy, mask = random_drop(image[img_idx, ].cpu().data.numpy(), mode=2,
                                            SFM_center_radius_perc=0.85, SFM_center_sigma_perc=0.15)
            image_SFM[img_idx, ] = img_numpy
    image = torch.from_numpy(image_SFM).cuda()


if __name__ == '__main__':
    print(__file__)