import tensorflow as tf
import os

a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
c = a + b
os.system("./get2Size")
# 通过log_device_placement参数来输出运行每一个运算的设备。
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
print
os.system("./get2Size")
#sess.run(c)
os.system("./get2Size")