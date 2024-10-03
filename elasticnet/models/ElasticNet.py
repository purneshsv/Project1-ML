

class ElasticNetModel():
    def __init__(self,lambdas,thresh = 0.5,max_iter=1000, tol=1e-4):
        self.lambdas = lambdas
        self.thresh = thresh# between [0,1],0 means only L1 regularization, 1 means only L2 regularization      
        self.max_iter = max_iter#Maximum number of iterations
        self.tol = tol
        self.w = None
        pass


    def fit(self, X, y):
        return ElasticNetModelResults()


class ElasticNetModelResults():
    def __init__(self):
        pass

    def predict(self, X):
        return X @ self.w 
