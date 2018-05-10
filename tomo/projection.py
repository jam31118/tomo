"""Tools for projecting a N-D data to lower dimension"""

from numbers import Real

import numpy as np


def overlap_flat(positionFromCenter, r=1):
    """
    Calculate the area of a part of a circle with radius r,
    above a line that has relative ``positionFromCenter``
    
    Thus the shape of the domain is like a moon, divided by a straight line.
    The area of the domain is retured.
    """
    ## Check if the input value is in valid range
    ## .. i.e. the positionFromCenter should be within from -r to r
    ## .. where r is the radius of a circle
    assert np.all(np.abs(positionFromCenter) <= r)
    
    ## Calculate the moon-shaped area
    area = r**2 * np.arccos(positionFromCenter) \
        - positionFromCenter*np.sqrt(r**2 - positionFromCenter**2)
        
    ## Return the calculated area
    return area


def overlap(distanceFromCenter, angle, r=1, threshold=1e-6):
    """
    Calculate overlap between incoming ray and the target to be reconstructed
    """
    # Maximum width of range that the distanceFromCenter shouldn't exceed
    range_width = 2.0*r/np.cos(angle)
    
    # Calculate absolute value of distanceFromeCenter
    distanceFromCenter_abs = np.abs(distanceFromCenter)
    
    # Check whether the distanceFromCenter doesn't exceed(overflow) the valid range
    # It is possible that overflow is originated from small numeric errors
    # .. In that case, the size of overflow is checked 
    # .. whether it is small enough to be considered as just numeric error.
    # .. In order to check it, the overflow value is compared with threshold which is set small
    if np.abs(distanceFromCenter) > 2.0*r/np.cos(angle):
        overflowFromValidRange = np.abs(distanceFromCenter) - range_width
        overflowIsPermissable = overflowFromValidRange < threshold
        
        # If the overflow value is smaller than threshold, it is considered as numeric error
        # .. and the distanceFromCenter_abs is set to the maximum value of valid range (i.e.'range_width')
        if overflowIsPermissable: distanceFromCenter_abs = range_width
        
        # If the overflow isn't permissable, then the distanceFromCenter is considered as abnormal
        # .. and an AssertionError is raised
        else: raise AssertionError(
            "distanceFromCenter_abs > range_width <=> %f > %f"
            % ( distanceFromCenter_abs, range_width )
        )
        
    return overlap_flat(distanceFromCenter_abs * np.cos(angle) - r)


def get_w_matrix(angle, line_index, array_size, r = 1.0, flatten=False, threshold=1e-8):
    """
    angle: scalar, numeric
    .. The angle of projecting ray
    .. The origin is the center pixel of the image to be reconstructed
    .. Each dimension should be odd integers in order to have well-defined origin
    
    line_index: scalar, numeric
    .. the offset of ray from the center ray
    
    array_size: 2-int-tuple,
    .. the shape of image to be reconstructed
    """
    
    # Assuming circle per pixel with radius r
    pixel_edge_length = 2*r
    
    # If the input angle is outside of valid range (from -pi/2 to +pi/2)
    # .. bring the angle into the range
    if (angle > np.pi / 2.0):
        while (angle > np.pi / 2.0):
            angle = angle - np.pi
            # When ever rotating 180 degree, the line_index should also be reversed w.r.t. origin
            line_index = -line_index 
    if (angle < -np.pi / 2.0):
        while (angle < -np.pi / 2.0):
            angle = angle + np.pi
            line_index = -line_index
    
    # If the angle doesn't belong to the valid range,
    # .. even after the bringing procedule, raise error
    if (np.abs(angle) > np.pi / 2.0):
        raise AssertionError("Valid angle range: -\pi / 2 <= angle <= \pi /2 ")
    
    if ((int(array_size[0]) % 2) == 0) or ((int(array_size[1]) % 2) == 0):
        # then, it is even number
        raise AssertionError(
            "Each dimention of array size should be odd integer (input size = (%d,%d))"
            %(array_size[0],array_size[1])
        )
    
    array_range_half_width = (array_size[0] // 2, array_size[1] // 2)
    #print("array_range_half_width == ",array_range_half_width)
    
    # initialization
    w_array = np.zeros(array_size)
    
    # equation of line y = y(x) is
    # .. y(x) = y_of_x = np.tan(angle) * x + (line_index * r)/np.cos(angle)
    x_min = -array_range_half_width[0]
    x_max = array_range_half_width[0]
    y_min = -array_range_half_width[1]
    y_max = array_range_half_width[1]
    for x in range(x_min,x_max+1):
        if (angle == np.pi / 2):
            if ((x > line_index) or (x < line_index)): continue
            y_of_x = 0
            y_lower = y_min; y_upper = y_max
        else:
            y_of_x = np.tan(angle) * x + (line_index * r)/np.cos(angle)
            y_width = 2*r / np.cos(angle) / pixel_edge_length
            y_lower = max([y_min, y_of_x - y_width])
            y_upper = min([y_max, y_of_x + y_width])
        
        # Fill in the weight matrix
        for y in range(int(np.ceil(y_lower)),int(np.floor(y_upper)+1)):
            #print(x+array_range_half_width[0], y+array_range_half_width[1])
            w_array[x+array_range_half_width[0],y+array_range_half_width[1]] \
                = overlap((y_of_x - y)*pixel_edge_length, angle) # should substract by 1 for indexing

    # Normalization
    area_per_pixel = np.pi * r**2   # Assuming circle
    w_array = w_array / area_per_pixel

    # Check normalization then 
    assert (w_array.max() - 1) < threshold
    assert (0 - w_array.min()) < threshold
    
    # cutoff the numerical tiny bits
    w_array[w_array > 1] = 1
    w_array[w_array < 0] = 0
    
    if (flatten): 
        return w_array.flatten()
    
    return w_array


def project_2d_to_1d(img, angle, ray_index_range_width):
    
    ## Check input arguments
    for arg in [angle, ray_index_range_width]: assert isinstance(arg, Real)
    assert ray_index_range_width == int(ray_index_range_width)
    assert isinstance(img, np.ndarray)
    
    numOfIndex = 2 * ray_index_range_width + 1
    projected = np.zeros((numOfIndex,), dtype=float)
    for line_index in range(-ray_index_range_width,ray_index_range_width+1):
        ray_w_array = get_w_matrix(angle, line_index, img.shape)
        projected[line_index + ray_index_range_width] = np.inner(ray_w_array.flatten(), img.flatten())
    return projected


def get_sinogram_from_image(image_array, angles, ray_index_range_width, verbose=False):
    
    for arr_arg in [image_array, angles]: assert isinstance(arr_arg, np.ndarray)
    assert (image_array.ndim == 2) and (angles.ndim == 1)
    assert ray_index_range_width == int(ray_index_range_width)
    assert ray_index_range_width >= 0
    
    num_of_angle = angles.size
    num_of_ray = 2 * ray_index_range_width + 1
    
    sinogram_array_shape = (num_of_angle, num_of_ray)
    sinogram = np.empty(sinogram_array_shape, dtype=float)
    
    for idx, angle in enumerate(angles):
        if verbose: print("Projecting at angle {0:.3f} pi".format(angle / np.pi))
        sinogram[idx,:] = project_2d_to_1d(image_array, angle, ray_index_range_width)
    
    return sinogram


