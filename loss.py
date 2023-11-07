import tensorflow as tf
def loss_function(y_true, y_pred):
  y_true = tf.reshape(y_true, shape=(-1, 512))
  
  loss = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')(y_true, y_pred)

  mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
  loss = tf.multiply(loss, mask)

  return tf.reduce_mean(loss)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=2000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)  # Cast d_model to float32
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)  # Cast step to float32
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    
def accuracy(y_true, y_pred):
  y_true = tf.reshape(y_true, shape=(-1, 512))
  return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)