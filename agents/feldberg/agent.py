
import numpy as np
from keras.models import load_model

class FeldbergAgent ():
    def __init__ (self):
        self.model = load_model('agents/feldberg/models/model10.h5')
        self.input_shape = self.model.input_shape
    
    def name (self):
        return {'nombre': "Serena", 'apellido': "Feldberg", 'legajo': 34759}

    def action (self, state):
        return np.argmax(self.model.predict(state))