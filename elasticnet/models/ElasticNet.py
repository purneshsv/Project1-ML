import numpy as np

class ElasticNetModel():
    def __init__(self,lambdas,thresh = 0.5,max_iter=1000, tol=1e-4):
        self.lambdas = lambdas
        self.thresh = thresh# between [0,1],0 means only L1 regularization, 1 means only L2 regularization      
        self.max_iter = max_iter#Maximum number of iterations
        self.tol = tol
        self.w = None
        pass


    def fit(self, X, y):
        def calculate_weights(X, residual):
            pass
        self.w = np.zeros(X.shape[1])  # Initialize weights including intercept
        max_iter = self.max_iter
        while max_iter > 0:
            max_iter -= 1
            # Compute predictions and residuals
            pred = X @ self.w
            residual = pred - y

            # Calculate weight gradients
            gradient = calculate_weights(X, residual)
            # Update weights
            w_old = self.w.copy()
            self.w -= self.learning_rate * gradient

            # Check convergence
            if np.linalg.norm(self.w - w_old) < self.tol:
                break

        self.coef = self.w[1:]
        return ElasticNetResults(self.coef)



class ElasticNetResults():
    def __init__(self):
        pass

    def predict(self, X):
        return X @ self.w 
