import csv
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Example dataset: 2 features (X) and 1 target (y)
with open('../data/data.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    data_csv = [row for row in reader]
    data_csv.pop(0)  # Assuming first row is header

X = [train_row[:4] for train_row in data_csv]  # Features (first 4 columns)
y = [train_row[4:][0] for train_row in data_csv]  # Target (5th column)

# Encode categorical data to numeric values
feature_encoders = [LabelEncoder() for _ in range(len(X[0]))]  # Create a separate encoder for each feature
X_encoded = []

# Apply label encoding to each feature column
for col_idx, encoder in enumerate(feature_encoders):
    # Use the encoder to transform each feature column
    X_encoded.append(encoder.fit_transform([row[col_idx] for row in X]))

# Transpose the encoded data back to the original format
X_encoded = list(zip(*X_encoded))

# Encode the target variable (y)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=30)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)

# Decoding predictions back to original categories (using the label encoder for y)
y_pred_decoded = label_encoder.inverse_transform(y_pred)

# Decoding test features back to original categories
X_test_decoded = []
for col_idx, encoder in enumerate(feature_encoders):
    X_test_decoded.append(encoder.inverse_transform([row[col_idx] for row in X_test]))

# Reconstruct X_test_decoded to match the format
X_test_decoded = list(zip(*X_test_decoded))

# Calculate accuracy
tp = 0
for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:
        tp += 1

# Output decoded predictions and the test set
print("Decoded X_test:", X_test_decoded)
print("Original y_test:", y_test)
print("Decoded y_pred:", y_pred_decoded)
print("Accuracy:", round(tp / len(y_test), 2))
