# train.py

import preprocessing
import cnn
from sklearn.model_selection import train_test_split

# Load data from preprocessing file
X = preprocessing.X
y = preprocessing.y

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load CNN model
cnn_model = cnn.model

# Train the model
history = cnn_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32
)

# Save trained model
cnn_model.save('mood_model.h5')

print("Training completed! Model saved as 'mood_model.h5'")