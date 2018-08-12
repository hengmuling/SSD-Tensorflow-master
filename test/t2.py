import tensorflow as tf
import numpy as np
batch = np.array(np.random.randint(1, 100, [10, 5]),dtype=np.float64)
mm, vv=tf.nn.moments(batch,axes=[0])#按维度0求均值和方差
#mm, vv=tf.nn.moments(batch,axes=[0,1])求所有数据的平均值和方差
sess = tf.Session()

print( sess.run([mm, vv]))#一定要注意参数类型
sess.close()




print(batch)
