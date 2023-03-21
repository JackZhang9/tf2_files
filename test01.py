import tensorflow as tf
import time
x=tf.random.normal([2,4,4,3],0.,1.,seed=10)

x9=tf.reshape(x,[4,-1])
# print(x,x9)

e1 = time.perf_counter()
x0=tf.pow(x,2)
e2 = time.perf_counter()
print(e2 - e1)

e1 = time.perf_counter()
x2=tf.sqrt(x)
e2 = time.perf_counter()
print(e2 - e1)

e1 = time.perf_counter()
x1=tf.square(x)
e2 = time.perf_counter()
print(e2 - e1)
w1=tf.Variable(tf.random.normal([60000,784],0.,0.1,seed=10))

w2=tf.Variable(tf.random.normal([784,256],0.,0.1,seed=10))
print(w1@w2)
# tf.Tensor([1.6368568 1.0018815 1.2575399 0.4501197], shape=(4,), dtype=float32)




