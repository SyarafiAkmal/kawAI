import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

def KNNs(k, data : pd.DataFrame):
    """
    A function to create, train, and evaluate a KNN model.
    
    Parameters:
    - k: Number of neighbors for the KNN classifier (default is 5)
    - data: DataFrame containing features and target column
    """
    
    # Separate features (X) and target (y)
    X = data.iloc[:, :-1]  # get all column except last column
    label = data.iloc[:, -1] # get only last column
    
    
    # PREPROCESS START
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    
    # Apply OneHotEncoder to categorical columns
    encoder = OneHotEncoder()  # Remove sparse=False argument
    X_encoded = encoder.fit_transform(X[categorical_columns])
    
    # Convert sparse matrix to dense if using an older version
    X_encoded = X_encoded.toarray()  # Convert sparse matrix to dense array
    
    # Convert encoded features back to DataFrame with appropriate column names
    encoded_columns = encoder.get_feature_names_out(categorical_columns)
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_columns)
    
    # Drop original categorical columns and concatenate with the encoded columns
    X = X.drop(categorical_columns, axis=1)  # Drop original categorical columns
    X = pd.concat([X, X_encoded_df], axis=1)
    
    # PREPROCESS END
    
    
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.2, shuffle=False)
    
    # datanya udh di preprocess di luar
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    
    # Create the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Train the model
    knn.fit(X_train, y_train)
    
    # Make predictions
    y_pred = knn.predict(X_test)
    # print(y_pred)
    # print(y_test)
    print("Score : ", knn.score(X_test, y_test))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

    # print(y_test)
    
    # print(y_pred)
    
    # Evaluate the model
    # print("Classification Report:")
    # print(classification_report(y_test, y_pred))
    
    # print("\nConfusion Matrix:")
    # print(confusion_matrix(y_test, y_pred))





def NBs(data: pd.DataFrame):
    """
    A function to create, train, and evaluate a Naive Bayes model.
    
    Parameters:
    - data: DataFrame containing features and target column
    """
        
    # Separate features (X) and target (y)
    X = data.iloc[:, :-1]  # Get all columns except the last column
    label = data.iloc[:, -1]  # Get only the last column
    
    # PREPROCESS START
    
    # Identify categorical columns
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    
    # Apply OneHotEncoder to categorical columns
    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X[categorical_columns])
    
    # Convert sparse matrix to dense array
    X_encoded = X_encoded.toarray()
    
    # Convert encoded features back to DataFrame with appropriate column names
    encoded_columns = encoder.get_feature_names_out(categorical_columns)
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_columns)
    
    # Drop original categorical columns and concatenate with the encoded columns
    X = X.drop(categorical_columns, axis=1)
    X = pd.concat([X, X_encoded_df], axis=1)
    
    # PREPROCESS END
    
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.2, shuffle=False)
    
    # Create the Gaussian Naive Bayes classifier
    nb = GaussianNB()
    
    # Train the model
    nb.fit(X_train, y_train)
    
    # Make predictions
    y_pred = nb.predict(X_test)
    # print(y_pred)
    # print(y_test)
    print("Score:", nb.score(X_test, y_test))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    
    
    # Evaluate the model
    # print("\nClassification Report:")
    # print(classification_report(y_test, y_pred))
    
    # print("\nConfusion Matrix:")
    # print(confusion_matrix(y_test, y_pred))

def ID3s(data: pd.DataFrame):
    """
    A function to create, train, and evaluate an ID3 Decision Tree model.
    
    Parameters:
    - data: DataFrame containing features and target column
    """
    
    # Separate features (X) and target (y)
    X = data.iloc[:, :-1]  # Get all columns except the last column
    label = data.iloc[:, -1]  # Get only the last column
    
    # PREPROCESS START
    
    # Identify categorical columns
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    
    # Apply OneHotEncoder to categorical columns
    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X[categorical_columns])
    
    # Convert sparse matrix to dense array
    X_encoded = X_encoded.toarray()
    
    # Convert encoded features back to DataFrame with appropriate column names
    encoded_columns = encoder.get_feature_names_out(categorical_columns)
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_columns)
    
    # Drop original categorical columns and concatenate with the encoded columns
    X = X.drop(categorical_columns, axis=1)
    X = pd.concat([X, X_encoded_df], axis=1)
    
    # PREPROCESS END
    
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.2, shuffle=False)
    
    # Create the ID3 Decision Tree classifier (criterion='entropy' for ID3)
    id3 = DecisionTreeClassifier(criterion='entropy')
    
    # Train the model
    id3.fit(X_train, y_train)
    
    # Make predictions
    y_pred = id3.predict(X_test)
    
    # Evaluate the model
    print("Score : ", id3.score(X_test, y_test))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    # print("\nClassification Report:")
    # print(classification_report(y_test, y_pred))
    # print("\nConfusion Matrix:")
    # print(confusion_matrix(y_test, y_pred))