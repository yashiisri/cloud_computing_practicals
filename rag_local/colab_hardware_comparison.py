import tensorflow as tf
import time

# Set computation device
device = "/device:CPU:0"  # Change to "/GPU:0" or "/TPU:0" for testing

with tf.device(device):
    start = time.time()

    # Dummy matrix multiplication (simulate workload)
    a = tf.random.normal([3000, 3000])
    b = tf.random.normal([3000, 3000])
    c = tf.matmul(a, b)

    end = time.time()

print(f"Time taken on {device}: {end - start:.4f} seconds")
