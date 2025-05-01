import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# Prepare the dummy data (same as before for demonstration)
# data=pd.read_csv()
data = {
 'packet_size': [1500, 1400, 1600, 200, 180, 220],
 'packet_frequency': [1000, 900, 1100, 50, 40, 60],
 'protocol_type': ['TCP', 'UDP', 'TCP', 'UDP', 'UDP', 'TCP'],
 'label': ['DDoS', 'Normal', 'DDoS', 'Normal', 'Normal', 'DDoS']
}
df = pd.DataFrame(data)
print(df)
df = pd.get_dummies(df, columns=['protocol_type'])

# Train the model
print(df)
X = df.drop('label', axis=1)
y = df['label']
print(X,y)
model = RandomForestClassifier()
print(model)
model.fit(X, y)
# Take input from the user
packet_size = int(input("Enter packet size: "))
packet_frequency = int(input("Enter packet frequency: "))
protocol_type_tcp = int(input("Is protocol type TCP? (1 for yes, 0 for no): "))
protocol_type_udp = 1 - protocol_type_tcp # complement
# Create a DataFrame with user input
user_data = pd.DataFrame({
 'packet_size': [packet_size],
 'packet_frequency': [packet_frequency],
 'protocol_type_TCP': [protocol_type_tcp],
 'protocol_type_UDP': [protocol_type_udp]
})
# Predict whether it's a DDoS attack or not
prediction = model.predict(user_data)
if prediction[0] == 'DDoS':
 print("The input is classified as a DDoS attack.")
else:
 print("The input is classified as normal traffic.")