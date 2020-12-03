import cv2
import os
import scipy.stats as st
import numpy as np
from scipy.ndimage.filters import convolve
from multiprocessing import Pool

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

def rl(raw_image, psf, niter):
    
    # Normalize PSF.
    psf /= psf.sum()
    
    # Magic proerties involved here,
    # We can compute H^T in this way.
    psf_adjug = psf[::-1]
    
    # Initialize O to the mean of the image.
    lucy = np.ones( raw_image.shape ) * raw_image.mean()

    for i in range(niter):
        # Convolve estimate wth the point spread function.
        estimate = convolve(lucy, psf, mode='mirror')
        estimate[np.isnan(estimate)] = 0
        
        # Divide raw image by estimate and convolve with the adjugate
        correction = convolve(raw_image/estimate, psf_adjug, mode='mirror')
        correction[np.isnan(correction)] = 0
        
        # Multiply to get the next value.
        lucy *= correction
        
    return lucy

# Now let's define the grid search.
kernel_size_start = 1
kernel_size_end = 40
kernel_size_nstep = 4

nsig_start = 1
nsig_end = 5
nsig_nstep = 10

n_iter = 500

kernels = np.linspace(kernel_size_start, kernel_size_end, kernel_size_nstep, dtype=np.int16)
nsigs = np.linspace(nsig_start, nsig_end, kernel_size_nstep)

sim_data = []
for kernel in kernels:
    for nsig in nsigs:
        sim_data.append({"kernel": kernel, "nsig": nsig})

# Define results
sample = "../samples/crater.png"
target_dir = "results"

# Let's load sample
sample = cv2.imread(sample, cv2.IMREAD_GRAYSCALE)

def process(data):
    kernel = data["kernel"]
    nsig = data["nsig"]
    psf = gkern(kernel, nsig)
    estimate = rl(sample, psf, n_iter)
    filename = os.path.join(target_dir, f"sig_{nsig}_kern_{kernel}.png")
    cv2.imwrite(filename, estimate)

def group_list(l, group_size):
    """
    :param l:           list
    :param group_size:  size of each group
    :return:            Yields successive group-sized lists from l.
    """
    for i in range(0, len(l), group_size):
        yield l[i:i+group_size]

# process
worker_count = 5

batches = []
for batch in group_list(sim_data, worker_count):
    batches.append(batch)

if __name__ == '__main__':
    batchIndex = 0
    for i in range(0, len(batches)):
        print(f"Running batch {batchIndex+1}/{len(batches)} ...")
        batchIndex += 1
            
        #run all simulations
        with Pool(processes=worker_count) as pool:
            pool.map(process, batches[batchIndex])
