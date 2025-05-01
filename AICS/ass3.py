import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("carprices.csv")

# Display the first few rows
print("\nDataset preview:")
print(df.head())

# Plot 1: Mileage vs Sell Price
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(df['Mileage'], df['Sell Price($)'], color='blue')
plt.xlabel('Mileage')
plt.ylabel('Sell Price($)')
plt.title('Mileage vs Sell Price')

# Plot 2: Age vs Sell Price
plt.subplot(1, 2, 2)
plt.scatter(df['Age(yrs)'], df['Sell Price($)'], color='green')
plt.xlabel('Age (years)')
plt.ylabel('Sell Price($)')
plt.title('Age vs Sell Price')

# Show plots
plt.tight_layout()
plt.show()

# Prepare features and label
X = df[['Mileage', 'Age(yrs)']]
y = df['Sell Price($)']

# Split data
print("\nTRAIN TEST SPLIT")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
print("X_train:\n", X_train)
print("X_test:\n", X_test)

# Train model
print("\nLINEAR REGRESSION")
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
print("\nPredicted Sell Prices:")
print(predictions)

# Actual values
print("\nActual Sell Prices:")
print(y_test.values)

# Model accuracy
accuracy = model.score(X_test, y_test)
print(f"\nModel Accuracy (R² Score): {accuracy:.2f}")
