import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


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
    
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.2, random_state=42)
    
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
    print(X_test)
    
    print(y_pred)
    
    # Evaluate the model
    # print("Classification Report:")
    # print(classification_report(y_test, y_pred))
    
    # print("\nConfusion Matrix:")
    # print(confusion_matrix(y_test, y_pred))

# Example usage (assuming you have a DataFrame `df` and target column 'Status')
# df = pd.read_csv('your_dataset.csv')
# knn_model(df, target_column='Status', k=3)
