import tensorflow as tf

x = tf.constant([[1., 2., 3.], [4., 5., 6.]])

if tf.config.list_physical_devices('GPU'):
  print("TensorFlow **IS** using the GPU")
else:
  print("TensorFlow **IS NOT** using the GPU")
