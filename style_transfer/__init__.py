import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

try:
    tf.config.optimizer.set_jit(True)
except Exception as e:
    print("JIT not set: ", e)
