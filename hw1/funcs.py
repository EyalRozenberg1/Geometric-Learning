import numpy as np
from numpy import fft
from PIL import Image


"""
define a Laplace operator
"""
def Laplace(N):
    L = 2 * np.eye(N, N)
    # L[0, 0] = 1  # deal with boundaries
    for k in range(N-1):
        L[k, k + 1] = -1
        L[k + 1, k] = -1
    return -L


def s_heat_equation(img, L_left, L_right):
    """
    :param img: numpy float-array
    :return: single step laplace operator on rows and columns
    """
    return (L_left @ img) + (img @ L_right)


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def fourier2d(image):
    """
    :param image
    :return: its fourier transform
    """
    fft_img = fft.fft2(image)
    fft_img = fft.fftshift(fft_img)
    return fft_img

def inv_fourier2d(f_x):
    x = np.fft.ifftshift(f_x)
    x = np.fft.ifft2(x)
    return np.abs(x)

def gaussian(ax, ay, Nx, Ny, dt):
    """
    :param num_samples: Number of samples (elements in the dataset)
    % grid constants
    :param ax: length x [m]
    :param ay: length y [m]
    :param Nx: # of x pixels
    :param Ny: # of y pixels
    :param dt: time step
    :return psi: gaussian
    """
    x = np.linspace(-ax / 2, ax / 2, Nx)
    y = np.linspace(-ay / 2, ay / 2, Ny)
    X, Y = np.meshgrid(x, y)

    psi = np.exp(-((2 * np.pi) ** 2) * (X**2 + Y**2) * dt)
    return psi