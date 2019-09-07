import numpy as np

class Activation:
    @staticmethod
    def sigmoid(Z, derivative, other):
        A = 1 / (1 + np.exp(-Z))

        if not derivative:
            return A
        else:
            return A * (1 - A)

    @staticmethod
    def tanh(Z, derivative, other):
        num = np.exp(Z) - np.exp(-Z)
        den = np.exp(Z) + np.exp(-Z)
        A = np.divide(num, den)

        if not derivative:
            return A
        else:
            return (1 - np.power(A, 2))

    # @param args [optional]
    #     float args['beta'] slope for all A > 0
    #     float args['leak'] slope for all A < 0
    @staticmethod
    def relu(Z, derivative, other):
        beta = 1
        leak = 0
        if other is not None:
            if 'beta' in other:
                beta = other['beta']
            if 'leak' in other:
                leak = other['leak']
                
        if not derivative:
            A = Z.copy().astype(float)
            A[A < 0] *= leak
            A[A > 0] *= beta
            return A
        else:
            dA = Z.copy().astype(float)
            dA[dA > 0] = beta
            dA[dA == 0] = beta / 2.
            dA[dA < 0] = leak
            return dA

class Layer:
    
    def __init__(self, units, activation='relu', relu=None):
        # Hyperparameters
        self.units = units
        self.activation = activation
        
        # Activation
        # self.g: Executable activation function
        # self.g_arg: passed to activation function
        if activation == 'relu':
            self.g = Activation.relu

            beta = 1
            leak = 0
            if relu is not None:
                if 'beta' in relu:
                    beta = relu['beta']
                if 'leak' in relu:
                    leak = relu['leak']
            self.g_args = {
                'beta': beta,
                'leak': leak
            }

        elif activation == 'sigmoid':
            self.g = Activation.sigmoid
            self.g_args = None

        elif activation == 'tanh':
            self.g = Activation.tanh
            self.g_args = None
        
        # Parameters W, b
        self.parameters = {}
        # Learning cache
        self.cache = {}
        
        return
    
    def activate(self, Z, derivative=False):
        return self.g(Z, derivative, self.g_args)
    
    # @param int n_j number of nodes in previous layer
    def initialize(self, n_j, random_init=False, random_scale=0.01):
        # Number of nodes in this layer
        n_k = self.getUnits()
        
        # Initialize parameters
        # Layers with more than one node must be randomized
        # Otherwise, they will be symmetrical (learn identically)
        if n_k > 1 or random_init:
            W = np.random.randn(n_k, n_j) * random_scale
        else:
            W = np.zeros((n_k, n_j))
        b = np.zeros((n_k, 1))

        # Remember untrained parameters
        self.setParameters({
            'W': W,
            'b': b
        })
    
    def linear_forward(self, A_j):
        # Linear forward pass, broadcast b
        W = self.getParameter('W')
        b = self.getParameter('b')
        Z_k = np.dot(W, A_j) + b
        
        return Z_k
    
    def update_parameters(self, learning_rate):
        W = self.getParameter('W')
        b = self.getParameter('b')
        dW = self.getCacheItem('dW')
        db = self.getCacheItem('db')
        
        #print ('W: '+str(W))
        #print ('b: '+str(b))
        #print ('dW: '+str(dW))
        #print ('db: '+str(db))
        #print ('learning_rate: '+str(learning_rate))
        
        self.setParameters({
            'W': W - learning_rate * dW,
            'b': b - learning_rate * db
        })
    
    def getActivation(self):
        return self.activation
    
    def getCacheItem(self, item):
        return self.cache[item]
    
    def getParameter(self, item):
        return self.parameters[item]
    
    def getUnits(self):
        return self.units
    
    def setActivation(self, activation):
        self.activation = activation
        return
    
    # Set multiple cache items
    # @param dict cache
    def setCache(self, cache):
        for key in cache.keys():
            self.cache[key] = cache[key]
        return

    # Set a single cache item
    # @param String item cache key
    # @param value new cache value
    def setCacheItem(self, item, value):
        # May pass key value pair
        self.cache[item] = value
        return
    
    # Set multiple parameters
    # @param dict parameters
    def setParameters(self, parameters):
        for key in parameters.keys():
            self.parameters[key] = parameters[key]
        return
    
    def setUnits(self, units):
        self.units = units
        return

class Model:
    
    def __init__(self, input_size):
        #np.random.seed(3)
        
        # Populated with Layer objects
        self.layers = []

        # Number of input features
        self.input_size = input_size
        return
    
    def fit(self, X, Y, verbose=False, compute_cost_every=0, learning_rate=0.005, random_init=True, random_scale=0.01, iterations=5):
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.random_init = random_init
        self.random_scale = random_scale
        self.iterations = iterations
        
        # Initialize parameters
        self._initialize_parameters()
        
        # Iterate to fit
        for i in range(0, iterations):
            
            # Run forward passes
            Y_hat = self._forward_propagate(X)
            
            # Compute cost
            if compute_cost_every > 0 and i % compute_cost_every == 0:
                self._compute_cost(Y_hat, Y, iteration=i, verbose=verbose)
            
            # Run backward passes and update parameters
            self._backward_propagate(X, Y, Y_hat)
        
        # Show final cost
        if compute_cost_every > 0:
            self._compute_cost(Y_hat, Y, iteration=i, verbose=verbose)
    
    def predict(self, X):
        return self._forward_propagate(X)
    
    def layer(self, units, activation='relu'):
        layer = Layer(units, activation)
        self.layers.append(layer)
        return
    
    # Run backward propagation and update parameters
    def _backward_propagate(self, X, Y, Y_hat):
        L = len(self.layers)
        m = X.shape[1]
        
        dZ_j = None
        W_j = None
        
        # Propagate through layers
        for l in reversed(range(L)):
            
            # Keep any previous info
            dZ_k = dZ_j
            W_k = W_j
            
            # This layer
            layer_j = self.layers[l]
            # Previous layer (left)
            layer_i = None if l == 0 else self.layers[l - 1]
            
            # Cache
            A_i = X if layer_i is None else layer_i.getCacheItem('A')
            W_j = layer_j.getParameter('W')
            Z_j = layer_j.getCacheItem('Z')
            
            # Intermediate values
            dG_j = layer_j.activate(Z_j, derivative=True)
            
            if l == (L - 1):
                # Output layer
                
                # Assume logistic cost
                dA_j = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))
                dZ_j = np.multiply(dA_j, dG_j)
            else:
                dZ_j = np.multiply(np.dot(W_k.T, dZ_k), dG_j)
            
            # Outputs
            dW_j = np.dot(dZ_j, A_i.T) / m
            db_j = np.sum(dZ_j, axis=1, keepdims=True) / m
            
            # Save outputs
            layer_j.setCache({
                'dW': dW_j,
                'db': db_j
            })
            
            # Update parameters for this layer
            layer_j.update_parameters(self.learning_rate)
        
        return
    
    # Compute logistic cost
    def _compute_cost(self, Y_hat, Y, iteration=-1, verbose=False):
        # Logistic cost
        m = Y_hat.shape[1]
        logprops = np.multiply(Y, np.log(Y_hat)) + np.multiply(1 - Y, np.log(1 - Y_hat))
        cost = - np.sum(logprops, axis=1, keepdims=True) / m
        cost = np.squeeze(cost)
        
        # Show current cost
        if verbose:
            print('Cost after {} iterations: {}'.format(iteration, cost))
        
        return cost
    
    def _forward_propagate(self, X):
        L = len(self.layers)
        
        # Init input variable
        A_k = X

        # Propagate through layers
        for l in range(L):
            # Retrieve last outputs
            A_j = A_k
            
            # Execute forward pass
            layer_k = self.layers[l]
            Z_k = layer_k.linear_forward(A_j)
            A_k = layer_k.activate(Z_k)
            
            # Remember forward pass for backward
            layer_k.setCache({
                'A': A_k,
                'Z': Z_k
            })
        
        # Final value
        Y_hat = A_k
        
        return Y_hat
    
    def _initialize_parameters(self):
        L = len(self.layers)
        
        # First layer dimension not in self.layers
        n_k = self.input_size
        
        for l in range(L):
            # Get previous layer dimension
            n_j = n_k
            
            # Init untrained parameters
            layer = self.layers[l]
            layer.initialize(n_j, random_init=self.random_init, random_scale=self.random_scale)
            
            # Remember this layer dimension
            n_k = layer.getUnits()
        
        return
    
    def _update_parameters(self):
        L = len(self.layers)
        
        for l in range(L):
            self.layers[l].update_parameters(learning_rate=self.learning_rate)
        
        return
        
        