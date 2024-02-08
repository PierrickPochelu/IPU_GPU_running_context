import tensorflow as tf
import time

# Define the constant tensors x and y
x = tf.constant([i for i in range(250_000_000)], dtype=tf.float32)
y = tf.constant([i for i in range(250_000_000)], dtype=tf.float32)

# Define a placeholder for p
p = tf.Variable(1,dtype=tf.float32)


# Define the computation graph
z = tf.concat([x, y], axis=0)
z_mean = tf.reduce_mean(z)
result = z_mean + p

# Create a MirroredStrategy for distributed training
strategy = tf.distribute.MirroredStrategy()

# Create a computation function for distributed training
@tf.function
def distributed_compute(p):
#    with strategy.scope():
        z = tf.concat([x, y], axis=0)
        z_mean = tf.reduce_mean(z)
        result = z_mean + p
        return result

# Create a session and run the computation using distributed training
#if True:
with strategy.scope():
    for i in range(3):
        start_time=time.time()
        result_val = strategy.run(distributed_compute, args=(p,))
        result_val = result_val.values[0].numpy()
        
        if(i!=0):
            print(1./(time.time()-start_time))

