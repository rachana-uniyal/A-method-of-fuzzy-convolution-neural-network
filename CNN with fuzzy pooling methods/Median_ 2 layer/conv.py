import numpy as np

'''
Note: In this implementation, we assume the input is a 2d numpy array for simplicity, because that's
how our MNIST images are stored. This works for us because we use it as the first layer in our
network, but most CNNs have many more Conv layers. If we were building a bigger network that needed
to use Conv3x3 multiple times, we'd have to make the input be a 3d numpy array.
'''

class Conv3x3:
  # A Convolution layer using 3x3 filters.

  def __init__(self, num_filters):

    self.num_filters = num_filters

    # print("jp ",self.num_filters)

    # filters is a 3d array with dimensions (num_filters, 3, 3)
    # We divide by 9 to reduce the variance of our initial values
    self.filters = np.random.randn(num_filters, 3, 3) / 9
    # print(self.filters.shape)


  
  def set_filter(self, num_filters):
    self.num_filters = num_filters

    # filters is a 3d array with dimensions (num_filters, 3, 3)
    # We divide by 9 to reduce the variance of our initial values
    self.filters = np.random.randn(num_filters, 3, 3) / 9



  


  def iterate_regions(self, image):
    '''
    Generates all possible 3x3 image regions using valid padding.
    - image is a 2d numpy array.
    '''

    # print(image.shape)
    # h, w = image.shape

    h = image.shape[0]
    w = image.shape[1]


    c = 0
    for i in range(h - 2):
      for j in range(w - 2):
        im_region = image[i:(i + 3), j:(j + 3)]
        # print("aaya ",im_region.shape)

        yield im_region, i, j


  def iterate_regions2(self, image):
    '''
    Generates all possible 3x3 image regions using valid padding.
    - image is a 2d numpy array.
    '''


    # print("haa ",image.shape)
    h, w, z = image.shape

    for i in range(h - 2):
      for j in range(w - 2):
        im_region = image[i:(i + 3), j:(j + 3)]
        # print("aaya ab",im_region.shape)
        yield im_region, i, j

  def forward(self, input):
    '''
    Performs a forward pass of the conv layer using the given input.
    Returns a 3d numpy array with dimensions (h, w, num_filters).
    - input is a 2d numpy array
    '''
    self.last_input = input

    h, w = input.shape
    output = np.zeros((h - 2, w - 2, self.num_filters))
    # print("Ye filter ",output.shape)
    # print("filter shape ",self.filters.shape)

    for im_region, i, j in self.iterate_regions(input):
        # print("im_region ",im_region.shape,self.filters.shape,"   i ",i,"   j",j)
        output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

    # print("Ye lool ",output.shape)
    # print("Ye lool ",output[2][2][3])
    # print("Ye lool ",output[3][2][3])

    return output



  def backprop(self, d_L_d_out, learn_rate):
    '''
    Performs a backward pass of the conv layer.
    - d_L_d_out is the loss gradient for this layer's outputs.
    - learn_rate is a float.
    '''
    d_L_d_filters = np.zeros(self.filters.shape)

    # print("yo ",self.filters.shape)

    for im_region, i, j in self.iterate_regions(self.last_input):
      for f in range(self.num_filters):
        # print("done")
        # print("sad ",d_L_d_filters[f].shape)
        # print(im_region.T)
        # print("hello ",d_L_d_out[i, j, f].shape, im_region.T.shape)
        try :
            d_L_d_filters[f] += np.multiply(d_L_d_out[i, j, f] , im_region.T[f])
        
        except:
            pass
    # Update filters
    self.filters -= learn_rate * d_L_d_filters

    # We aren't returning anything here since we use Conv3x3 as the first layer in our CNN.
    # Otherwise, we'd need to return the loss gradient for this layer's inputs, just like every
    # other layer in our CNN.
    return None





  def forward2(self, input):
    
    self.last_input = input
    h, w, z = input.shape
    self.set_filter(z)
    output = np.zeros((h - 2, w - 2,self.num_filters))
    # print("Ye filter ",output.shape)
    # print("Ye input ",input.shape)

    for it in range(0, self.num_filters):
        for im_region, i, j in self.iterate_regions2(input):
            # print("im_region ",im_region.shape,self.filters.shape,"   i ",i,"   j",j)

            # print("im_region ",im_region.shape,"   i ",i,"   j",j)
            # print("laa",im_region.shape)
            output[i, j, it] = np.sum(im_region[:,:,it] * self.filters[it])
            # , axis=(1, 2))

    # print("Ye lool ",output.shape)

    return output
