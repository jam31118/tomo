import numpy as np

from numbers import Number
def get_test_image(width, dtype=float):
    """Return a toy image on which tomographic reconstruction will be tested.
    
    Returned image's shape is (N,N) where

    N = width + 1 + width
      = 2 * width + 1
    """

    ## Check input arguments
    assert isinstance(width, Number)
    if width != int(width): raise TypeError("`width` should be an integer.")
    
    ## Define shape of the image
    image_size = (width*2+1, width*2+1)

    ## Allocate memory for storing image
    test_img = np.zeros(image_size, dtype=dtype)

    ## Add some signals
    test_img[:,:] += 1
    test_img[width-2:width+12, width-23:width+1] += 1
    test_img[width-6:width+8, width-23:width+30] += 2
    test_img[width-2:width+2, width-1:width+50] += 2
    
    ## Return test image
    return test_img


