# import pickle
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # Load the data
# with open('./data.pickle', 'rb') as f:
#     data_dict = pickle.load(f)

# # Inspect the data to find the maximum length
# max_length = max(len(x) for x in data_dict['data'])

# # Pad each sequence in the data to the maximum length
# # Here, we assume that the data is a list of sequences (e.g., lists or arrays)
# padded_data = np.array([np.pad(x, (0, max_length - len(x)), 'constant') if isinstance(x, list) else x for x in data_dict['data']], dtype=np.float64)

# # Convert labels to numpy array
# labels = np.array(data_dict['labels'])

# # Split the data
# x_train, x_test, y_train, y_test = train_test_split(padded_data, labels, test_size=0.2, shuffle=True, stratify=labels)

# # Initialize and train the model
# model = RandomForestClassifier()
# model.fit(x_train, y_train)

# # Make predictions
# y_predict = model.predict(x_test)

# # Calculate accuracy
# score = accuracy_score(y_predict, y_test)

# print(f'{score * 100:.2f}% of samples were classified correctly!')

# # Save the model
# with open('model.p', 'wb') as f:
#     pickle.dump({'model': model}, f)

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def process_data(item):
    if isinstance(item, list):
        return [subitem for sublist in item for subitem in (process_data(sublist) if isinstance(sublist, list) else [sublist])]
    return [item]

# Load the data
try:
    with open('./data.pickle', 'rb') as f:
        data_dict = pickle.load(f)
except FileNotFoundError:
    print("Error: data.pickle file not found. Make sure it's in the correct directory.")
    exit(1)
except pickle.UnpicklingError:
    print("Error: Unable to unpickle data.pickle. The file might be corrupted.")
    exit(1)

# Process the data
try:
    data = [process_data(d) for d in data_dict['data']]
    max_length = max(len(d) for d in data)
    data = [d + [0] * (max_length - len(d)) for d in data]  # Pad with zeros
    data = np.array(data)
    labels = np.array(data_dict['labels'])
except KeyError:
    print("Error: The data dictionary doesn't contain 'data' or 'labels' keys.")
    exit(1)
except ValueError as e:
    print(f"Error processing data: {e}")
    exit(1)

# Print data shape for debugging
print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")

# Split the data
try:
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
except ValueError as e:
    print(f"Error splitting data: {e}")
    exit(1)

# Create and train the model
model = RandomForestClassifier()

try:
    model.fit(x_train, y_train)
except ValueError as e:
    print(f"Error fitting the model: {e}")
    exit(1)

# Make predictions and calculate accuracy
try:
    y_predict = model.predict(x_test)
    score = accuracy_score(y_predict, y_test)
    print(f'{score * 100:.2f}% of samples were classified correctly!')
except ValueError as e:
    print(f"Error during prediction or scoring: {e}")
    exit(1)

# Save the model
try:
    with open('model.p', 'wb') as f:
        pickle.dump({'model': model}, f)
    print("Model saved successfully.")
except IOError:
    print("Error: Unable to save the model.")
    exit(1)