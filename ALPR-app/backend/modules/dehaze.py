import cv2
import numpy as np
from PIL import Image


def _dark_channel(img, size=15):
    b, g, r = cv2.split(img)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark = cv2.erode(dc, kernel)
    return dark


def _get_atmosphere(img, dark_channel):
    [h, w] = img.shape[:2]
    image_size = h * w
    numpx = int(max(image_size / 1000, 1))
    darkvec = dark_channel.reshape(image_size)
    imvec = img.reshape(image_size, 3)

    indices = darkvec.argsort()[image_size - numpx::]

    atmsum = np.zeros([1, 3])
    for ind in range(numpx):
        atmsum = atmsum + imvec[indices[ind]]

    a = atmsum / numpx
    return a


def _get_transmission(img, a, omega=0.95, size=15):
    im3 = np.empty(img.shape, img.dtype)
    for ind in range(3):
        im3[:, :, ind] = img[:, :, ind] / a[0, ind]
    transmission = 1 - omega * _dark_channel(im3, size)
    return transmission


def _guided_filter(i, p, r, eps):
    mean_I = cv2.boxFilter(i, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(i * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(i * i, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * i + mean_b
    return q


def _recover(img, t, a, tx=0.1):
    res = np.empty(img.shape, img.dtype)
    t = cv2.max(t, tx)
    for ind in range(3):
        res[:, :, ind] = (img[:, :, ind] - a[0, ind]) / t + a[0, ind]
    
    # Normalize result safely
    res = np.clip(res, 0.0, 1.0)
    return res


def dehazing_mathematically(image_np, omega=0.95, t0=0.1, radius=15, eps=1e-3):
    """
    Enhances a hazy image mathematically using the Dark Channel Prior method.
    
    Args:
        image_np (numpy.ndarray): Input RGB image array.
        omega (float): Amount of haze to remove (0 to 1).
        t0 (float): Minimum transmission value to avoid noise amplification.
        radius (int): Window size for processing.
        eps (float): Epsilon for guided filter.
        
    Returns:
        numpy.ndarray: The enhanced image.
    """
    img = image_np.astype('float64') / 255.0

    dark = _dark_channel(img, radius)
    a = _get_atmosphere(img, dark)
    te = _get_transmission(img, a, omega, radius)
    
    # Convert image to grayscale for guided filter guidance image
    gray = cv2.cvtColor(np.uint8(img * 255), cv2.COLOR_RGB2GRAY).astype('float64') / 255.0
    t = _guided_filter(gray, te, radius * 4, eps)

    res = _recover(img, t, a, t0)
    
    final_output = (res * 255.0).round().astype(np.uint8)
    return final_output


def process_with_dehaze(image, omega=0.95, t0=0.1, radius=15, eps=1e-3, math_fallback=True):
    """
    Restore a hazy image and return the result as a PIL image.
    Currently uses mathematical Dark Channel Prior (DCP) by default 
    as the primary dehaze method, but structured to allow DL models.
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    else:
        img_np = np.array(image).copy()

    # Deep learning logic would go here, 
    # e.g., lazily loading a model if available.
    
    if math_fallback:
        dehazed_img = dehazing_mathematically(img_np, omega, t0, radius, eps)
        return Image.fromarray(dehazed_img)
    
    return image
