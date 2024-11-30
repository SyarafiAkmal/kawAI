import csv
import Model
from Model import Model
from statistics import mode

class KNearestNeighbor(Model):
    def score(self, params = None):
        """
        Scores performance based on KNN results to the test dataset

        returns: a score element of accuracy 
        """
        tp = 0
        k = 3
        for i in range(len(self.data_test_x)):
            result = self.KNN(k, self.data_test_x[i])
            if(result == self.data_test_y[i][0]):
                tp += 1
    
        return (tp/len(self.data_test_x))  

    def distance(self, query, row):
        """
        Calculates the distance (difference) between the query (a row from the test dataset) and the row from train dataset

        returns: the distance in a form of integer
        """
        dist = 0
        for i in range(len(query)):
            if query[i] != row[i]:
                dist += 1
        return dist

    def KNN(self, k, query):
        """
        Calculates the classification of a single query (row from test dataset) using the KNN algorithm

        returns: the result classification
        """
        dist_array = [[0, i] for i in range(len(self.data_train_x))]
        for i in range(len(self.data_train_x)):
            dist_array[i][0] = self.distance(query, self.data_train_x[i])
        
        sorted_dist = sorted(dist_array, key=lambda x: x[0])
        sorted_dist = sorted_dist[:k]
        result = [self.data_train_y[dist[1]][0] for dist in sorted_dist]
        return mode(result)


if __name__ == "__main__":
    with open('../data/dummy.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        data_csv = [row for row in reader]
        data_csv.pop(0)

    kNN = KNearestNeighbor(data_csv[:7], data_csv[7:], len(data_csv[0]) - 1)
    print(kNN.score())




