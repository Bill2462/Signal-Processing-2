import cv2
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage.filters import convolve
from tqdm import tqdm

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

def rl_animation(raw_image, psf, niter):
    
    # Normalize PSF.
    psf /= psf.sum()
    
    # Magic proerties involved here,
    # We can compute H^T in this way.
    psf_adjug = psf[::-1]
    
    # Initialize O to the mean of the image.
    lucy = np.ones( raw_image.shape ) * raw_image.mean()
    
    samples = []
    
    for i in tqdm(range(niter)):
        # Convolve estimate wth the point spread function.
        estimate = convolve(lucy, psf, mode='mirror')
        estimate[np.isnan(estimate)] = 0
        
        # Divide raw image by estimate and convolve with the adjugate
        correction = convolve(raw_image/estimate, psf_adjug, mode='mirror')
        correction[np.isnan(correction)] = 0
        
        # Multiply to get the next value.
        lucy *= correction
        
        samples.append({"n": i, "img": np.copy(lucy)})
        
    return samples

sample = cv2.imread("../samples/crater.png", cv2.IMREAD_GRAYSCALE)
psf = gkern(kernlen=30, nsig=5)
pictures = rl_animation(sample, psf, 20)

# Splice the array to obtain multiple

fig, ax = plt.subplots()

anims1 = []
print("Rendering...")
for pic in tqdm(pictures):
    n = pic["n"]
    im = plt.imshow(pic["img"], cmap='gray', animated=True)
    titleTXT = f"R-L iteration: {n}"
    title = ax.text(0.5, 1.05, titleTXT, 
                        size=plt.rcParams["axes.titlesize"],
                        ha="center", transform=ax.transAxes)
    anims1.append([im, title])

ani1 = animation.ArtistAnimation(fig, anims1, interval=1000,
                                 repeat_delay=1000, blit=True)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=1, bitrate=1800)

print("Saving...")

ani1.save('ani1.mp4', writer=writer)
