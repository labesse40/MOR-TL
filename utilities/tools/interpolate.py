import numpy as np
from scipy.interpolate import RegularGridInterpolator
from fractions import Fraction
from scipy.signal import resample_poly


def linearInterpolation( data, tmax, dt, dtNew ):
    """Interpolate field for each simulation timestep

    Parameters
    ----------
        data : array of float
            Field to interpolate
        maxTime : float
            End time of simulation
        dt : float
            Current field time step
        dtNew : float
            New time step
        

    Returns
    --------
        dinterp : array
            Interpolated array with dtNew time step
    """
 
    if data.shape[0] == ( int( tmax / dtNew ) + 1 ):
        # If the data is already in the new time step, return it
        return data

    if data.shape[0] != ( int( tmax / dt ) + 1 ):
        dt = tmax / (data.shape[0] - 1) # Adjust dt if data does not match expected time steps

    ratio = dt / dtNew
    frac = Fraction(ratio).limit_denominator(1000)
    
    # Resample the data using a polyphase filter, accounting for arbitrary (non-integer) ratios
    # and properly applying an anti-aliasing filter.
    dinterp = resample_poly(data, frac.numerator, frac.denominator, axis=0)

    return dinterp


def resample_w_interp(data,Nscale):
    """
    Resample a 2D square matrix using linear interpolation.

    This function performs linear interpolation to resample a 2D square matrix 
    to a new resolution while maintaining the aspect ratio. The scaling factor 
    is applied uniformly to both dimensions.

    Parameters:
        data (numpy.ndarray): A 2D square matrix representing the input data to be resampled.
        Nscale (float): Scaling of dimensions.

    Returns:
        numpy.ndarray: A 2D matrix resampled to the specified resolution.

    """

    Nx = round(data.shape[0] * Nscale)
    Ny = round(data.shape[1] * Nscale)

    x = np.linspace(0, data.shape[0] - 1, data.shape[0])
    y = np.linspace(0, data.shape[1] - 1, data.shape[1])

    X = np.linspace(0, data.shape[0] - 1, Nx)
    Y = np.linspace(0, data.shape[1] - 1, Ny)
    X, Y = np.meshgrid(X, Y, indexing='ij')

    interpoperator = RegularGridInterpolator((x, y), data, method='linear', bounds_error=False, fill_value=None)
    newdata = interpoperator(np.stack([X, Y], axis=-1))

    return newdata