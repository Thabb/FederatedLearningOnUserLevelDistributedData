import tensorflow as tf


class GradientModel(tf.Module):
    def __init__(self, value):
        super().__init__()
        self.weight = tf.Variable(value)

    @tf.function
    def __call__(self, *args, **kwargs):
        return None
