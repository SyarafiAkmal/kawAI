from sklearn.model_selection import train_test_split

class Model:
    def __init__(self, data, split_factor):
        """
        Basic constructor
        """
        cols = data.pop(0)
        split = round(len(data) * split_factor)
        class_offset = len(data[0]) - 1
        self.columns = {
            'x': cols[:class_offset],
            'y': cols[class_offset:],
            }
        # self.data_train_x, self.data_test_x, self.data_train_y, self.data_test_y = train_test_split([x_row[:class_offset] for x_row in data], [y_row[class_offset:] for y_row in data], test_size= split, random_state=50)
        self.data_train_x, self.data_test_x, self.data_train_y, self.data_test_y = [x_row[:class_offset] for x_row in data], [x_row[:class_offset] for x_row in data], [y_row[class_offset:] for y_row in data], [y_row[class_offset:] for y_row in data]


    def fit(self, params = None):
        """
        Trains model based on input data

        returns: an object model (JSON)
        """
        raise NotImplementedError("This method is not implemented by the class.")

    def score(self, params = None):
        """
        Scores the performance score based of the training model to the test dataset

        returns: a score element of accuracy
        """
        raise NotImplementedError("This method is not implemented by the class.")
    
    def predict(self, params = None):
        """
        Scores the performance score based of the training model to the test dataset

        returns: a score element of accuracy
        """
        raise NotImplementedError("This method is not implemented by the class.")