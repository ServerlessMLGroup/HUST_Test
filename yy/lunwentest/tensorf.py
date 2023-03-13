import tensorflow as tf
import os

tf.debugging.set_log_device_placement(True)
os.system("./get2Size")
#Place tensors on the GPU
#下面这行会引起如下图一至图N一系列操作
with tf.device('/GPU:2'):
    a=tf.constant([[1.0,2.0,3.0],[4.0,5.0,6.0]])
    b=tf.constant([[1.0,2.0],[3.0,4.0],[5.0,6.0]])
os.system("./get2Size")
c=tf.matmul(a,b)
print(c)
os.system("./get2Size")