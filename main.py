# Let's try to implement a transformer from scratch JUST using numpy! 

# Patrick Ming, 
#     started: 3/30/26, 20:15
#     ended:   x/xx/xx, xx:xx   


import numpy as np
np.set_printoptions(linewidth=200)

# d_model is the hidden dimension of the tokens, we will use d_model = 8 for checking the math, and d_model = 512 for actual implementation
class LayerNorm:
    def __init__(self, d_model, eps = 1e-6):
        # avoid dividing by 0 when normalizing
        self.eps = eps
        
        # if the gradients tell us to undo a normalization to keep from being at such a small scale, then we use these
        # y = g . x_hat + b
        self.gain = np.ones(d_model)
        self.bias = np.zeros(d_model)

        # saving values for backward pass. x_hat is the normalized vector x, and std is obvious
        self.x_hat = None
        self.std = None

    def forward(self, x):
        # normalize
        mu = np.mean(x)
        self.std = np.sqrt(np.var(x) + self.eps)
        self.x_hat = (x-mu)/self.std

        # scale and shift by gain and bias
        y = self.x_hat * self.gain + self.bias
        return y

    def backward(self):
        pass

ln = LayerNorm(d_model = 8)
test_var = np.array([10, 3, 4, 5, 3, 1, -4, -30])
print(test_var)
print(ln.forward(test_var))

