from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, InputLayer
from tensorflow.keras.optimizers import Adam

def build_srcnn(input_channels=3):
    SRCNN = Sequential()
    # add model layers
    SRCNN.add(Conv2D(filters = 128, kernel_size=  (9,9), kernel_initializer= 'glorot_uniform', activation = 'relu', padding = 'valid', use_bias = True, input_shape = (None, None, 1)))   
    SRCNN.add(Conv2D(filters = 64, kernel_size = (3,3), kernel_initializer='glorot_uniform', activation = 'relu', padding = 'same', use_bias = True))
    SRCNN.add(Conv2D(filters = 1, kernel_size = (5, 5), kernel_initializer='glorot_uniform', activation = 'linear', padding = 'valid', use_bias = True))
    
    # define optimizer
    adam = Adam(learning_rate=0.0003)
    
    SRCNN.compile(optimizer = adam, loss= 'mean_squared_error', metrics = ['mean_squared_error'])
    return SRCNN
