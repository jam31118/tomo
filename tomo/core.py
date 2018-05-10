import numpy as np

from .projection import get_w_matrix


class Tomo(object):
    image_array_order = 'C'

    def __init__(self, image, projection_measured, angles, index_range_width, relaxation_coef=1.0,r=1.0):
        """
        ## Arguments
        # image_size
        - if tuple with length 2: shape of image array
        - if numpy.ndarray: initial image array (initial guess)
        # projection_measured: 1D array-like
        .. should have size of len(angles) * len(2*index_range_width+1)
        """
        
        self.image_size = None
        self.initial_image_array = None
        if type(image) in [tuple, list]:
            assert len(image) == 2
            self.image_size = image
        elif isinstance(image, np.ndarray):
            self._check_constraint_on_2d_image_array(image)
            self.image_size = image.shape
            self.initial_image_array = image
        else: raise TypeError("Unexpected type {0} for 'image'".format(type(image)))
        assert self.image_size is not None

#        if (self.image_size != None):
            #self.image_flat_length = self.image_size[0] * self.image_size[1]
            # Initialize image
#            if self.initial_image_array is None:
#                self.current_image = np.ones(self.image_flat_length)
#            else: self.current_image = self.initial_image_array.flatten(order=self.image_array_order)
#        else: raise Exception("`self.image_size` shouldn't be None")

        self.current_image = None
        self.image_flat_length = self.image_size[0] * self.image_size[1]
        self.initializeImage()  # `self.current_image` is initialized
        assert self.current_image is not None

        
        self.angles = angles
        self.index_range_width = index_range_width
        self.line_index_list = range(-self.index_range_width, self.index_range_width+1)
        self.numOfLineIndex = 2 * self.index_range_width + 1
        self.projection_measured = projection_measured
        self.numOfProjection = len(projection_measured)
        self.relaxation_coef = relaxation_coef
        assert self.numOfProjection == (len(self.angles) * self.numOfLineIndex)
        
        self.weight_matrix = np.zeros((self.numOfProjection, self.image_flat_length))
        self.weight_norm = np.zeros(self.numOfProjection)
        self.haveWeightMatrixAndNorm = False
    
    def initializeImage(self):
        if self.initial_image_array is None:
            self.current_image = np.ones(self.image_flat_length)
        else: self.current_image = self.initial_image_array.flatten(order=self.image_array_order)

    def _check_constraint_on_2d_image_array(self, image):
        assert isinstance(image, np.ndarray)
        assert image.ndim == 2
        assert image.shape[0] == image.shape[1]  # check if the image is square

    def calculateWeightMatrixAndNorm(self):
        for idx_angle, angle in enumerate(self.angles):
            print("[%03d]for angle == " % idx_angle,angle)
            for idx_line_index, line_index in enumerate(self.line_index_list):
                idx_projection = idx_angle*self.numOfLineIndex + idx_line_index
                self.weight_matrix[idx_projection,:] \
                    = get_w_matrix(angle, line_index, self.image_size, flatten=True)
                self.weight_norm[idx_projection] = sum(self.weight_matrix[idx_projection,:]**2)
        self.haveWeightMatrixAndNorm = True
    
    def _getCorrectionFacter(self, idx_projection):
        projection_estimated = np.inner(self.weight_matrix[idx_projection,:], self.current_image)
        projection_difference = self.projection_measured[idx_projection] - projection_estimated
        correctionFacter = projection_difference / self.weight_norm[idx_projection]
        return correctionFacter
    
    def _projectCurrentImageOnHyperPlane(self, idx_projection):
        self.current_image \
            += self.relaxation_coef \
            * self.weight_matrix[idx_projection,:] \
            * self._getCorrectionFacter(idx_projection)
    
    def iterate(self, n_iter=1):
        for iter_idx in range(n_iter):
            for idx_projection in range(self.numOfProjection):
                self._projectCurrentImageOnHyperPlane(idx_projection)
    
    def getCurrentImage2D(self):
        return self.current_image.reshape(self.image_size, order=self.image_array_order)
    
    

class TomoART(Tomo):
    pass


class TomoMART(Tomo):
    # (SHOULD BE IMPLEMENTED LATER)
    def _projectCurrentImageOnHyperPlane(self, idx_projection):
        proj_measured = self.projection_measured[idx_projection]
        proj_estimated = np.inner(self.weight_matrix[idx_projection,:], self.current_image)
        multiplier = proj_measured / proj_estimated
        #print("proj_measured == ", proj_measured)
        #print("proj_estimated == ", proj_estimated)
        #print(multiplier)
        #print(multiplier)
        for idx in range(len(self.current_image)):
            #try:
            power = self.relaxation_coef * self.weight_matrix[idx_projection,idx]
            self.current_image[idx] *= pow(multiplier, power)
            #except:
            #   print("multiplier == ", multiplier)
            #    print("power == ", power)
    
    #pass
#    def _getCorrectionFacter(self, idx_projection):
#        projection_estimated = np.inner(self.weight_matrix[idx_projection,:], self.current_image)
#        projection_difference = self.projection_measured[idx_projection] - projection_estimated
#        correctionFacter \
#            = projection_difference \
#            / self.weight_norm[idx_projection] \
#            / (self.projection_measured[idx_projection])
#        return correctionFacter
#        #return super._getCorrectionFacter(idx_projection) / self.projection_measured[idx_projection]

