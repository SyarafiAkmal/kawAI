import csv, json
import Model
from Model import Model
import pandas as pd

class NaiveBayes(Model):
    def fit(self, params = None):
        """
        Trains the train dataset using NB algorithm

        returns: an object model (JSON)
        """
        df_unique = pd.DataFrame(self.data_train_x + self.data_test_x, columns=self.columns['x'])
        attribute_vals = []
        for col in df_unique.columns:
            attribute_vals += (df_unique[col].unique()).tolist()

        classes = list(set([row_y[0] for row_y in self.data_train_y]))
        model = {}
        for cls in classes:
            model[cls] = {
                "freq" : self.data_train_y.count([cls]),
                "p": round(self.data_train_y.count([cls]) / len(self.data_train_y), 2)
                }
            for att in attribute_vals:
                model[cls][att] = {
                    "freq" : 0,
                    "p": 0
                }  
        for i in range(len(self.data_train_x)):
            for att in attribute_vals:
                if att in self.data_train_x[i]:
                    model[self.data_train_y[i][0]][att]['freq'] += 1
        
        for cls in classes:
            for att in attribute_vals:
                model[cls][att]['p'] = round(model[cls][att]['freq'] / model[cls]['freq'], 2)
        
        return model
    
    def scoreRow(self, query):
        classes = list(set([row_y[0] for row_y in self.data_train_y]))
        result = []
        for cls in classes:
            p = 1
            for cell in query:
                p *= self.model[cls][cell]['p']
            result.append([cls, p * self.model[cls]['p']])
            

        return sorted(result, key=lambda x: x[1], reverse=True)[0][0] # returns the class with the highest

    def score(self, params = None):
        """
        Scores performance of model based on the test dataset

        returns: a score element of accuracy, precision, and recall 
        """
        self.model = self.fit()
        tp = 0
        for i in range(len(self.data_test_x)):
            if self.scoreRow(self.data_test_x[i]) == self.data_test_y[i][0]:
                tp += 1
        
        return round(tp / len(self.data_test_x), 2)


if __name__ == "__main__":
    with open('../data/dummy.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        data_csv = [row for row in reader]

    naiveB = NaiveBayes(data_csv, 9)
    # print(json.dumps(naiveB.fit({'query': ['sunny', 'cool', 'high', 'true']}), indent=4))
    print(naiveB.score())




