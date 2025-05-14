import jax
import jax.numpy as jnp

def psnr(img1: jnp.ndarray, img2: jnp.ndarray) -> jnp.ndarray:
  """
  Compute the Peak Signal-to-Noise Ratio (PSNR) for a single pair of images.

  A small epsilon is added to avoid division by zero.
  """
  img1 = img1.astype(jnp.float32)
  img2 = img2.astype(jnp.float32)
  mse = jnp.mean((img1 - img2) ** 2)
  mse = jnp.maximum(mse, 1e-10)  # Avoid division-by-zero problems.
  return 10.0 * jnp.log10((255.0 ** 2) / mse)

def gaussian_kernel(window_size: int, sigma: float) -> jnp.ndarray:
  """
  Create a 2D Gaussian kernel.

  Args:
    window_size: The size of the window (typically 11).
    sigma: The standard deviation of the Gaussian.

  Returns:
    A (window_size x window_size) normalized Gaussian kernel.
  """
  # Create a 1D range centered at zero.
  ax = jnp.arange(-window_size // 2 + 1., window_size // 2 + 1.)
  # Create a 2D grid of (x,y) coordinates.
  xx, yy = jnp.meshgrid(ax, ax)
  kernel = jnp.exp(-(xx**2 + yy**2) / (2. * sigma**2))
  return kernel / jnp.sum(kernel)

def convolve(img: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
  """
  Convolve an image (or a batch of images) with a given kernel.
  
  The convolution is done per channel using group convolution.

  Args:
    img: A tensor of shape [N, H, W, C].
    kernel: A 2D kernel of shape [kH, kW].

  Returns:
    The convolved image tensor of shape [N, H, W, C].
  """
  kH, kW = kernel.shape
  # Reshape kernel to [kH, kW, 1, 1].
  kernel = jnp.reshape(kernel, (kH, kW, 1, 1))
  # We want to apply the same kernel to each channel separately.
  C = img.shape[-1]
  kernel = jnp.tile(kernel, (1, 1, 1, C))
  
  return jax.lax.conv_general_dilated(
      img,
      kernel,
      window_strides=(1, 1),
      padding='SAME',
      dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
      feature_group_count=C  # This ensures each channel is convolved separately.
  )

def ssim(
    img1: jnp.ndarray,
    img2: jnp.ndarray,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    L: float = 255.0
) -> jnp.ndarray:
  """
  Compute the Structural Similarity (SSIM) index between two images.
  
  SSIM is computed locally using a Gaussian filter and then averaged over all
  image locations and channels.
  
  Args:
    img1: First image tensor, shape [N, H, W, C].
    img2: Second image tensor, shape [N, H, W, C].
    window_size: Size of the Gaussian filter window (default 11).
    sigma: Standard deviation of the Gaussian (default 1.5).
    k1: Constant to stabilize the division with weak denominator (default 0.01).
    k2: Constant to stabilize the division with weak denominator (default 0.03).
    L: Dynamic range of the pixel values (default 255.0 for images in [0, 255]).
  
  Returns:
    A scalar representing the mean SSIM index over the batch.
  """
  # Ensure the images are in floating point.
  img1 = img1.astype(jnp.float32)
  img2 = img2.astype(jnp.float32)
  
  # Create the Gaussian kernel.
  kernel = gaussian_kernel(window_size, sigma)
  
  # Compute local means.
  mu1 = convolve(img1, kernel)
  mu2 = convolve(img2, kernel)
  
  mu1_sq = mu1 ** 2
  mu2_sq = mu2 ** 2
  mu1_mu2 = mu1 * mu2
  
  # Compute local variances and covariance.
  # Using the formula: var(X) = E[X²] - E[X]²
  sigma1_sq = convolve(img1 ** 2, kernel) - mu1_sq
  sigma2_sq = convolve(img2 ** 2, kernel) - mu2_sq
  # Using the formula: cov(X,Y) = E[XY] - E[X]E[Y]
  sigma12 = convolve(img1 * img2, kernel) - mu1_mu2
  
  # Constants for numerical stability.
  C1 = (k1 * L) ** 2
  C2 = (k2 * L) ** 2
  
  # Compute the SSIM map.
  numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
  denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
  ssim_map = numerator / denominator
  
  # SSIM should be calculated for each channel separately, then averaged
  # First take mean over height and width (spatial dimensions)
  ssim_per_channel = jnp.mean(ssim_map, axis=(1, 2))
  # Then average over channels and batch
  return jnp.mean(ssim_per_channel)