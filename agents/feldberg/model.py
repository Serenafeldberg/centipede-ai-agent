import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.optimizers.legacy import Adam

states = np.load('agents/feldberg/states/39501_states_new.npy')
actions = np.load('agents/feldberg/actions/39501_actions_new.npy')
# X_train, X_test, y_train, y_test = train_test_split(states, actions, test_size=0.2, random_state=42)
X_train = states
y_train = actions

# 2. Preprocesa los datos (ajusta según tus necesidades)
X_train = X_train / 255.0  # Normaliza los valores de píxeles

# 3. Diseña la arquitectura de la red neuronal
output = 18  # Número de acciones posibles
HEIGHT = 184
WIDTH = 160
CHANNELS = 6
model = Sequential([
    Conv2D (32, (5, 5), activation = 'relu', input_shape = (HEIGHT, WIDTH, CHANNELS), strides = 5, padding='same'),
    MaxPooling2D(pool_size=(3, 3)),
    Conv2D (16, (3, 3), activation = 'relu', strides = 2, padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(output, activation='softmax')
])

# model.summary()

# # 4. Compila el modelo
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# # 5. Entrena el modelo
model.fit(X_train, y_train, epochs=50, batch_size=32) #, validation_data=(X_test, y_test))
model.save('agents/feldberg/models/model.h5')

# # 6. Evalúa el modelo en un conjunto de datos de prueba
# # (Asegúrate de tener datos de prueba etiquetados)
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f'Loss: {loss}, Accuracy: {accuracy}')
    
