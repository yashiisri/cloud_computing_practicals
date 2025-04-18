import tensorflow as tf
import time

# Matrix size for multiplication
MATRIX_SIZE = 3000

# Create random matrices
a = tf.random.uniform([MATRIX_SIZE, MATRIX_SIZE])
b = tf.random.uniform([MATRIX_SIZE, MATRIX_SIZE])

# Function to benchmark device
def benchmark(device_name):
    with tf.device(device_name):
        print(f"\nRunning on {device_name}...")
        start = time.time()
        c = tf.matmul(a, b)
        _ = c.numpy()  # Force computation
        print(f"Time taken on {device_name}: {time.time() - start:.4f} seconds")

# CPU
benchmark('/CPU:0')

# GPU (only if running in GPU runtime)
if tf.config.list_physical_devices('GPU'):
    benchmark('/GPU:0')
else:
    print("\nNo GPU found.")

# TPU
try:
    print("\nInitializing TPU...")
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    
    with strategy.scope():
        print("Running on TPU...")
        start = time.time()
        result = tf.matmul(a, b)
        _ = result.numpy()
        print(f"Time taken on TPU: {time.time() - start:.4f} seconds")
except:
    print("\nTPU not available in this runtime. Make sure to select TPU from Runtime > Change runtime type.")
