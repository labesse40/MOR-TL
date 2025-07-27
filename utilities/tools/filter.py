import sys
import numba
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq


def frequencyFilter(signal, fmax, dt, padding=False):
    """Filter the desired 2d np.array signal. It filters the signal in the interval [0, fmax+1]

    Parameters:
    -----------
        signal : 2d np.array
            2d numpy array
        fmax : float
            Max frequency to keep in the signal
        dt : float
            Time step between values in signal
        padding : bool
            Whether the filtered signal should be padded (to avoid discontinuity)

    Returns
    -------
        y : 2d np.array
            Real part of filtered array
    """
    if str(fmax) == "all":
        return signal

    pad = int(round(signal.shape[0]/2))
    n = signal.shape[0] + 2*pad

    tf = fftfreq(n, dt)
    y_fft = np.zeros((n,signal.shape[1]), dtype="complex_")
    y = np.zeros(y_fft.shape, dtype="complex_")

    for i in range(y_fft.shape[1]):
        y_fft[pad:n-pad,i] = signal[:,i]
        y_fft[:,i] = fft(y_fft[:,i]) # Perform Fourier transform

    isup = np.where(tf>=fmax)[0]
    imax = np.where(tf[isup]>=fmax+1)[0][0]
    i1 = isup[0]
    i2 = isup[imax]

    iinf = np.where(tf<=-fmax)[0]
    imin = np.where(tf[iinf]<=-fmax-1)[0][-1]

    i3 = iinf[imin]
    i4 = iinf[-1]

    for i in range(y_fft.shape[1]):
        y_fft[i1:i2,i] = np.cos((isup[0:imax] - i1)/(i2-i1) * np.pi/2)**2 * y_fft[i1:i2,i]
        y_fft[i3:i4,i] = np.cos((iinf[imin:-1] - i4)/(i3-i4) * np.pi/2)**2 * y_fft[i3:i4,i]
        y_fft[i2:i3,i] = 0

    for i in range(y.shape[1]):
        y[:,i] = ifft(y_fft[:,i])# Perform inverse Fourier transform

    if padding:
        return np.real(y)
    else:
        return np.real(y[pad:n-pad])


@numba.njit
def suavxyF(M, NI, NJ):
    """
    Smooth a 2D numpy array using a moving average filter.

    This function applies a moving average filter to smooth the input 2D numpy array `M`. 
    The smoothing is performed by averaging the values of neighboring elements within 
    a rectangular window defined by `NI` (number of rows) and `NJ` (number of columns) 
    around each element. The edges of the array are handled by adjusting the window size 
    to fit within the bounds of the array.

    Parameters:
    -----------
        M : 2d np.array
            Input 2D numpy array to be smoothed. Must not be None.
        NI : int
            Number of neighboring rows to include in the smoothing window. Must be greater than 0.
        NJ : int
            Number of neighboring columns to include in the smoothing window. Must be greater than 0.

    Returns:
    --------
        R : 2d np.array
            Smoothed 2D numpy array where each element is replaced by the average of its 
            neighbors within the defined window. The output array has the same shape as the input array.

    Notes:
    ------
        - If `NI` or `NJ` is less than or equal to 0, the function will terminate with an error.
        - The function ensures that the smoothing window does not exceed the boundaries of the input array.
    """
    if M is not None:
        NX = M.shape[0]
        NY = M.shape[1]
    else:
        raise ValueError("Error: input is None")

    if (NI is not None) and (NJ is not None):
        if (NI<0 or NJ<0):
            raise ValueError("Error: NI and NJ must be higher than 0")
    else:
        raise ValueError("Error: NI and NJ must have a valid value")

    R = np.zeros((NX,NY))
    for ij in range(NY):
        j1 = ij+1-NJ
        j2 = ij+1+NJ
        if (j1<1): j1=1
        if (j2>NY): j2=NY
        for ii in range(NX):
            i1 = ii+1-NI
            i2 = ii+1+NI
            if (i1<1): i1=1
            if (i2>NX): i2=NX

            SM = 0
            N = 0
            for jj in range(j1,j2):
                for ji in range(i1,i2):
                    N+=1
                    SM = SM + M[ji-1,jj-1]

            R[ii,ij] = SM/N

    return R
