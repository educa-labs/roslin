import numpy as np

class Output:

    def fit(self,X=None,y=None):
        return self

    def predict(self,X=None):
        return X

    @staticmethod
    def calculate_score(distance):
        return (1/(1+distance))

    def set_k(self,k):
        pass

# Temporal solution, take neighbors of first input
class DumbOutput(Output):

    def predict(self,X=None):
        
        scores = np.array(list(map(lambda d: DumbOutput.calculate_score(d),X[0][0])))
        indexes = X[1][0]
        return [scores,indexes]

#Return the best neighbors of each input.
class GreedyOutput(Output):

    def __init__(self,k=5):
        self.k = k

    def predict(self,X=None):
        scores = []
        indexes = []

        for j in range(len(X[0][0])):
            for i in range(len(X[0])):
                if X[1][i][j] not in indexes:
                    scores.append(GreedyOutput.calculate_score(X[0][i][j]))
                    indexes.append(X[1][i][j])
                if len(scores) == self.k:
                    return [scores,indexes]
        return [scores,indexes]

    def set_k(self,k):
        self.k = k