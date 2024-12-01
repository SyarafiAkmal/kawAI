import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Step 1: Load dataset from CSV
df = pd.read_csv('src/data/data.csv')

# Step 2: Identify categorical columns and encode them
# Assuming 'attack_cat' is the target column, let's check for any other categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Label encode categorical features (other than the target column 'attack_cat')
label_encoder = LabelEncoder()

for col in categorical_cols:
    if col != 'attack_cat':  # We don't want to encode the target column
        df[col] = label_encoder.fit_transform(df[col])

# Step 3: Split the dataset into features (X) and target labels (y)
X = df.drop('attack_cat', axis=1)  # Drop the 'attack_cat' column to get the features
y = df['attack_cat']  # 'attack_cat' column is the labels

# Step 4: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Initialize the kNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Step 6: Train the model
knn.fit(X_train, y_train)

# Step 7: Make predictions on the test set
y_pred = knn.predict(X_test)

# Step 8: Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
