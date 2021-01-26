import numpy as np

class Softmax:
  # A standard fully-connected layer with softmax activation.

  def __init__(self, input_len, nodes):
    # We divide by input_len to reduce the variance of our initial values
    nodes=200
    self.weights = np.random.randn(input_len, nodes) / input_len
    self.biases = np.zeros(input_len)

    self.biases = self.biases.reshape(input_len,1)



  def forward(self, input):
    '''
    Performs a forward pass of the softmax layer using the given input.
    Returns a 1d numpy array containing the respective probability values.
    - input can be any array with any dimensions.
    '''
    self.last_input_shape = input.shape

    input = input.flatten()
    self.last_input = input

    input = np.array(input)
    input = input.reshape(200,1)

    input_len, nodes = self.weights.shape

    # print("d ",input.shape)
    # print("w ",self.weights.shape)
    # print("w ",self.biases.shape)
    totals = np.dot(self.weights, input) + self.biases
    self.last_totals = totals

    exp = np.exp(totals)
    return exp / np.sum(exp, axis=0)

  def backprop(self, d_L_d_out, learn_rate):
    '''
    Performs a backward pass of the softmax layer.
    Returns the loss gradient for this layer's inputs.
    - d_L_d_out is the loss gradient for this layer's outputs.
    - learn_rate is a float.
    '''
    # We know only 1 element of d_L_d_out will be nonzero
    for i, gradient in enumerate(d_L_d_out):
      if gradient == 0:
        continue

      # e^totals
      t_exp = np.exp(self.last_totals)

      # Sum of all e^totals
      S = np.sum(t_exp)

      # Gradients of out[i] against totals
      d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
      d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

      # Gradients of totals against weights/biases/input
      d_t_d_w = self.last_input
      d_t_d_b = 1
      d_t_d_inputs = self.weights

      # Gradients of loss against totals
      d_L_d_t = gradient * d_out_d_t

      # Gradients of loss against weights/biases/input


      # print("yooo ",d_t_d_w[np.newaxis].T.shape)
      # print("noo",d_L_d_t[np.newaxis][:][:][0].T.shape)

      d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis][:][:][0].T
      d_L_d_b = d_L_d_t * d_t_d_b

      # print("yooo ",d_L_d_t.T.shape)
      # print("noo", d_t_d_inputs.shape)

      d_L_d_inputs =  d_L_d_t.T @ d_t_d_inputs

      d_L_d_inputs = d_L_d_inputs.T
      # print("done ",d_L_d_inputs.shape)

      # print(self.weights.shape)
      # print(d_L_d_w.shape)

      # Update weights / biases
      self.weights -= learn_rate * d_L_d_w.T
      self.biases -= learn_rate * d_L_d_b

      return d_L_d_inputs.reshape(self.last_input_shape)
