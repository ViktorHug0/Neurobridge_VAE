import numpy as np
import random

# Use English instead of Chinese comments

try:
    # Much faster C-implementation for moving average along time axis
    from scipy.ndimage import uniform_filter1d  # type: ignore
except Exception:  # pragma: no cover
    uniform_filter1d = None

class RandomTimeShift:
    """
    Randomly shift the EEG signal along the time axis (axis=-1).
    max_shift indicates the maximum shift amount (forward or backward), in terms of number of samples.
    """
    def __init__(self, max_shift=5):
        self.max_shift = max_shift

    def __call__(self, eeg_data: np.ndarray) -> np.ndarray:
        # eeg_data shape: (..., time)
        shift = random.randint(-self.max_shift, self.max_shift)
        if shift != 0:
            eeg_data = np.roll(eeg_data, shift, axis=-1)
        return eeg_data


class RandomGaussianNoise:
    """
    Adds random Gaussian noise to the EEG signal.
    std indicates the standard deviation of the noise.
    """
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, eeg_data: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.std, size=eeg_data.shape)
        return eeg_data + noise


class RandomChannelDropout:
    """
    Randomly drop some channels (set them to zero).
    drop_prob indicates the probability of dropping a channel.
    Assume eeg_data shape: (channel, time) or (channel, time, ...).
    """
    def __init__(self, drop_prob=0.1):
        self.drop_prob = drop_prob

    def __call__(self, eeg_data: np.ndarray) -> np.ndarray:
        # Assuming the first dimension is the channel dimension
        channels = eeg_data.shape[0]
        for ch in range(channels):
            # Randomly decide whether to drop this channel
            if random.random() < self.drop_prob:
                eeg_data[ch] = 0
        return eeg_data


class RandomSmooth:
    """
    A simple smoothing operation, which can be understood as a simple convolution / moving average along the time axis.
    kernel_size indicates the size of the moving average kernel.
    """
    def __init__(self, kernel_size=5, smooth_prob=0.5):
        self.kernel_size = kernel_size
        self.smooth_prob = smooth_prob

    def __call__(self, eeg_data: np.ndarray) -> np.ndarray:
        # Expected shape: (channel, time)
        ch, _time_len = eeg_data.shape
        smoothed = np.array(eeg_data, copy=True)

        # Decide which channels to smooth
        mask = np.random.rand(ch) < self.smooth_prob
        if not np.any(mask):
            return smoothed

        if uniform_filter1d is not None:
            # Fast moving-average smoothing along time axis
            smoothed[mask] = uniform_filter1d(
                eeg_data[mask],
                size=self.kernel_size,
                axis=-1,
                mode="nearest",
            )
            return smoothed

        # Fallback: numpy convolution (slower than scipy but still vectorized)
        k = int(self.kernel_size)
        if k <= 1:
            return smoothed
        kernel = np.ones(k, dtype=eeg_data.dtype) / k
        pad = k // 2
        x = np.pad(eeg_data[mask], ((0, 0), (pad, pad)), mode="edge")
        smoothed[mask] = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="valid"), -1, x)
        return smoothed



class RandomApply:
    """
    Randomly apply a given transform to the EEG data with a probability p.
    """
    def __init__(self, transform, p=0.5):
        self.transform = transform
        self.p = p

    def __call__(self, eeg_data: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            return self.transform(eeg_data)
        return eeg_data