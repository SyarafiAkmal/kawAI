
class Model:
    def __init__(self, data_train, data_test, class_offset):
        """
        Basic constructor
        """
        self.data_train_x, self.data_train_y = [train_row[:class_offset] for train_row in data_train], [train_row[class_offset:] for train_row in data_train]
        self.data_test_x, self.data_test_y = [test_row[:class_offset] for test_row in data_test], [test_row[class_offset:] for test_row in data_test]

    def fit(self, params):
        """
        Trains model based on input data

        returns: an object model (JSON)
        """
        raise NotImplementedError("This method is not implemented by the class.")

    def score(self, params):
        """
        Scores the performance score based of the training model to the test dataset

        returns: a score element of accuracy
        """
        raise NotImplementedError("This method is not implemented by the class.")