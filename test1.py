import tensorflow.compat.v1 as tf
import numpy as np
#import matplotlib.pyplot as plt 

tf.disable_v2_behavior()

v1= tf.Variable(0.0)
p1= tf.placeholder(tf.float32)
new_val = tf.add(v1,p1)
update = tf.assign(v1, new_val)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(5):
        sess.run(update, feed_dict={p1:1.0})
    print(sess.run(v1))
