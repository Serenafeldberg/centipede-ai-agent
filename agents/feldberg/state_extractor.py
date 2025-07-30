import numpy as np
import cv2
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

class StateExtractor:

    def __init__ (self):
        self.mushroom_kernel = self.kernel (3, 4, True)
        self.player_kernel = self.kernel (9, 4, True)
        self.lives_kernel = self.kernel (7, 4, True) 
        self.shot_kernel = self.kernel (9, 1, True)

        self.centipede_kernel = np.array([[-1, 1, -1, -1],
             [1, 1, 1, -1],
             [1, 1, 1, 1],
             [1, 1, 1, 1],
             [1, 1, 1, -1],
             [-1, 1, -1, -1]])
        
        self.spider_kernel = np.array([
            [-11,-11,-11,-11,-11,-11,-11,-11,-11,-11],
            [-11,80,-11,-11,-11,-11,-11,-11,80,-11],
            [-11,-11,50,-11,50,50,-11,50,-11,-11],
            [-11,-11,50,2,2,2,2,50,-11,-11],
            [-11,-11,-11,2,2,2,2,-11,-11,-11],
            [-11,50,-11,2,2,2,2,-11,50,-11],
            [-11,-11,50,2,2,2,2,50,-11,-11],
            [-11,-11,-11,-11,-11,-11,-11,-11,-11,-11]
        ])


    def kernel (self, rows, cols, border= False):
        kernel = np.ones((rows, cols))

        if border:
            new_kernel = np.zeros((rows+2, cols+2))
            new_kernel[1:-1, 1:-1] = kernel

            return (new_kernel-0.5)*2
        return (kernel-0.5)*2
    
    def extract (self, image):
        image = image[:184]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, obs = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY)

        conv_mushroom = convolve2d(obs, self.mushroom_kernel, mode='same')
        conv_player = convolve2d(obs, self.player_kernel, mode='same')
        conv_shot = convolve2d(obs, self.shot_kernel, mode='same')
        conv_centipede = convolve2d(obs, self.centipede_kernel, mode='same')
        conv_spider = convolve2d(obs, self.spider_kernel, mode='same')

        HEIGHT = 184
        WIDTH = 160

        matrix = np.concatenate([(conv_mushroom > 11).reshape((HEIGHT, 160, 1)),
                                 (conv_player > 35).reshape((HEIGHT, 160, 1)),
                                 (conv_shot > 8).reshape((HEIGHT, 160, 1)),
                                 (conv_centipede > 15).reshape((HEIGHT, 160, 1)),
                                 (conv_spider > 339).reshape((HEIGHT, 160, 1)),
                                 (image).reshape(HEIGHT, 160, 1)], axis=-1).astype(np.uint8)

        return np.expand_dims(matrix, axis=0)

