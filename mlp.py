from lib import *

class Sequential:
    def __init__(self, layers):
        self.layers = layers
    def forward(self, x):
        for layers in self.layers:
            x = layers.forward(x)
        return x
    
    def __call__(self, x):
        return self.forward(x)
    
class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features)
        self.b = np.zeros((1, out_features))

    def forward(self, x):
        return np.dot(x, self.W) + self.b
    
    def parameter(self):
        return(self.W, self.b)
    
class ReLU:
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)
    
class MaxPool2D:
    def __init__(self, kernel_size = 2, stride = 2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self. x):
        batch_size, height, width , channels = x.shape
        out_height = (height - self.kernel_size) // self.stride + 1
        out_width =  (width - self.kernel_size) // self.stride + 1
        out = np.zeros((batch_size, out_height, out_width, channels))

        for i in range(out_height):
            for j in range(out_width):
                x_path = x[:, i * self.stride:i * self.stride + self.kernel_size,
                           j * self.stride:j * self.stride + self.kernel_size, :]
                
                out[:, i, j, :] = np.max(x_path, axis=(1, 2))
        return out
    
class Flatten:
    def forward(self, x):
        return x.reshape(x.reshape[0], 1)
    
class Dropout:
    def __init__(self, p=0.5):
        assert 0 <= p <= 1, "Probability p in about [0, 1]"
        self.p = p
        self.training = True

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        
        self.mask = np.random.rand(*x.shape) > self.p
        return x * self.mask / (1 - self.p)

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        #Initialize weights and bias random
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.bias = np.random.randn(out_channels)

    def forward(self, x):
        # Import size input
        batch_size, in_channels = H, W = x.shape
        assert in_channels = self.in_channels,  "in_channels is not match"

        # Add padding into input
        x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        # Calculate size output
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Initialize output
        output = np.zero(batch_size, self.out_channels, H_out, W_out)

        # Perform convolution
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size

                        x_slice = x_padded[b, :, h_start:h_end, w_start:h_end]

                        output[b, oc, h, w] = np.sum(x_slice * self.weights[oc], self.bias[oc])

        return output

    def __call__(self, x):
        return self.forward(x)
    
     
    

