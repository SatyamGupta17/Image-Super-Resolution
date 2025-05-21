import tensorflow as tf

if tf.config.list_physical_devices('GPU'):
    print("Tensorflow is using the GPU")
    gpus = tf.config.list_physical_devices('GPU')
    print("Available GPUs: ", gpus)
    for gpu in gpus:
        details = tf.config.experimental.get_device_details(gpu)
        print("GPU Details : ", details)
    
else:
    print("Tensorflow is using the CPU")