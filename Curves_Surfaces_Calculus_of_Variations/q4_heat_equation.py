from skimage import data
import matplotlib.pyplot as plt
from Curves_Surfaces_Calculus_of_Variations.funcs import *
import time

# load gray-scale "cameraman" image to np.array.
# pad boundaries for dealing with boundary conditions
# img = np.pad(data.camera(), 1, mode='constant').astype(np.float)
img = np.array(data.camera(), dtype=np.float)
steps = 100
method = 0  # 0-Euler, 1-Fourier
dt = 5e-7 if method == 1 else 0.1  # time step | Euler method dt=0.1 | Fourier method dt=5e-7

z = np.linspace(0, img.shape[0], img.shape[1])
X, Y = np.meshgrid(z, z)

rows, cols = img.shape

if method == 0:
    print("Euler")
    start = time.time()

    """
    Derive the a representation of the Laplace Operator
    """
    L_left = Laplace(rows)  # left operator
    if rows == cols:
        L_right = L_left
    else:
        L_right = Laplace(cols)  # right operator

    image = img.copy()
    gif = []
    for l in range(steps):
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121)
        plt.title(f'image, iter={l}', fontsize=18)
        ax.imshow(image, cmap=plt.cm.gray)
        ax = fig.add_subplot(122, projection='3d')
        plt.title(f'surface, iter={l}', fontsize=18)
        surf = ax.plot_surface(X, Y, image / image.max(), cmap=plt.cm.gist_gray, antialiased=False, rstride=10, cstride=10)
        ax.view_init(elev=40., azim=90.)
        ax.invert_xaxis()
        ax.set_zlim(0, 1)
        gif.append(fig2img(fig).convert('RGB'))
        plt.close()
        # gif.append(Image.fromarray(image.astype(np.uint8)).convert('RGB'))
        image += dt * s_heat_equation(image, L_left, L_right)
    # plt.imshow(image, cmap=plt.cm.gray)
    # plt.show()
    end = time.time()
    print("total time: ", end - start)
    gif[0].save('..\\figs\\Euler.gif', save_all=True, append_images=gif[1:], optimize=False, duration=70, loop=0)
# Fourier
else:
    print("Heat kernel")
    start = time.time()
    
    # calculate the 2d fourier transform of the image
    img_f = fourier2d(img)
    image = inv_fourier2d(img_f)
    gif = []
    for l in range(steps):
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121)
        plt.title(f'image, iter={l}', fontsize=18)
        ax.imshow(image, cmap=plt.cm.gray)
        ax = fig.add_subplot(122, projection='3d')
        plt.title(f'surface, iter={l}', fontsize=18)
        surf = ax.plot_surface(X, Y, image / image.max(), cmap=plt.cm.gist_gray, antialiased=False, rstride=10, cstride=10)
        ax.view_init(elev=40., azim=90.)
        ax.invert_xaxis()
        ax.set_zlim(0, 1)
        gif.append(fig2img(fig).convert('RGB'))
        plt.close()

        # define a gaussian if the fourier domain with increasing time-step equal to dt*step
        psi = gaussian(ax=rows, ay=cols, Nx=rows, Ny=cols, dt=(1+l)*dt)
        image_f_step = np.multiply(psi, img_f)
        image = inv_fourier2d(image_f_step)

    end = time.time()
    print("total time: ", end - start)
    gif[0].save('..\\figs\\Kernel.gif', save_all=True, append_images=gif[1:], optimize=False, duration=70, loop=0)
