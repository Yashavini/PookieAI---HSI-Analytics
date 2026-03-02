import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Dropout, InputLayer

def create_3d_cnn(input_shape=(25, 25, 30, 1), num_classes=16):
    """
    Creates a 3D-CNN model for Hyperspectral Image Classification.
    input_shape: (window_size, window_size, spectral_bands, channels)
    num_classes: Number of ground truth categories in Indian Pines (default 16)
    """
    model = Sequential()

    # 1st 3D Convolutional Layer
    # Extracts spatial and spectral features simultaneously
    model.add(InputLayer(input_shape=input_shape))
    model.add(Conv3D(filters=8, kernel_size=(3, 3, 7), activation='relu', padding='same'))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))

    # 2nd 3D Convolutional Layer
    model.add(Conv3D(filters=16, kernel_size=(3, 3, 5), activation='relu', padding='same'))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))

    # 3rd 3D Convolutional Layer
    model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same'))

    # Flatten the 3D volume into a 1D vector for the Dense layers
    model.add(Flatten())

    # Fully Connected (Dense) Layers
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.4))  # Prevents overfitting
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.4))

    # Output Layer
    # Using 'softmax' for multi-class classification
    model.add(Dense(units=num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    # Test if the model builds correctly
    model = create_3d_cnn()
    model.summary()
    print("\n✅ 3D-CNN Model architecture created successfully!")