import numpy as np
import math

class Fuzzypool2:
  # A Max Pooling layer using a pool size of 2.

  def iterate_regions(self, image):
    '''
    Generates non-overlapping 2x2 image regions to pool over.
    - image is a 2d numpy array
    '''
    h, w, _ = image.shape
    new_h = h // 2
    new_w = w // 2

    for i in range(new_h):
      for j in range(new_w):
        im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
        yield im_region, i, j






  def forward(self, input):
    '''
    Performs a forward pass of the maxpool layer using the given input.
    Returns a 3d numpy array with dimensions (h / 2, w / 2, num_filters).
    - input is a 3d numpy array with dimensions (h, w, num_filters)
    '''
    self.last_input = input

    h, w, num_filters = input.shape
    output = np.zeros((h // 2, w // 2, num_filters))

    

    for im_region, i, j in self.iterate_regions(input):
      r,c, _ = im_region.shape

      # print(r,c)
      s=0
      sd = 0
      for m in range(0,r):
        for n in range(0,c):
          s+= im_region[m][n]

      mean = s/r*c

      for m in range(0,r):
        for n in range(0,c):
          sd+= (im_region[m][n] - mean)*(im_region[m][n] - mean)

      sd = sd/(r*c)
      sd = np.sqrt(sd)

      for m in range(0,r):
        for n in range(0,c):
          im_region[m][n] = -((im_region[m][n]-mean)*(im_region[m][n]-mean))/2*(sd)*(sd)

      im_region = np.exp(im_region)
       
       
      th = np.amax(im_region, axis=(1))
      th = np.amin(im_region, axis=(0))

      output[i, j] = np.median(im_region, axis=(0, 1))

    return output





  def backprop(self, d_L_d_out):
    '''
    Performs a backward pass of the maxpool layer.
    Returns the loss gradient for this layer's inputs.
    - d_L_d_out is the loss gradient for this layer's outputs.
    '''
    d_L_d_input = np.zeros(self.last_input.shape)

    for im_region, i, j in self.iterate_regions(self.last_input):
      h, w, f = im_region.shape
      
      amax = np.median(im_region, axis=(0, 1))

      for i2 in range(h):
        for j2 in range(w):
          for f2 in range(f):
            # If this pixel was the max value, copy the gradient to it.
            if im_region[i2, j2, f2] == amax[f2]:
              d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]

    return d_L_d_input
