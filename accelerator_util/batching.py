import time
import tensorflow as tf
import numpy as np

input_data = np.random.normal(0.,1.,(1000,1000,250)).astype(np.float32)
dataset = tf.data.Dataset.from_tensor_slices(input_data)
dataset = dataset.batch(1)


start_time = time.time()

# Iterate over the dataset (optional)
for batch in dataset:
    pass

# End the timer
end_time = time.time()

# Calculate the batching time
batching_time = end_time - start_time

print("Batching time: {:.4f} seconds".format(batching_time))
