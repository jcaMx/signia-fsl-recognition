from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout

def build_static_model(num_classes, input_dim=63):
    """
    Builds and compiles the default MLP model for static signs.
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
