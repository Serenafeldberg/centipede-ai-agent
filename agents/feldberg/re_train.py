from keras.models import load_model
import numpy as np
# from keras.optimizers.legacy import Adam

model = load_model('agents/feldberg/models/model9.h5')

states = np.load('agents/feldberg/states/start_states_new.npy')
# states = states[(len(states)//2):]
actions = np.load('agents/feldberg/actions/start_actions_new.npy')
# actions = actions[(len(actions)//2):]
# X_train, X_test, y_train, y_test = train_test_split(states, actions, test_size=0.2, random_state=42)
X_train = states
y_train = actions

# 2. Preprocesa los datos (ajusta según tus necesidades)
X_train = X_train / 255.0 

# model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=32) #, validation_data=(X_test, y_test))

model.save('agents/feldberg/models/model10.h5')

