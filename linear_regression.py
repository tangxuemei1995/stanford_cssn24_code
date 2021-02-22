import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
def generate_dataset():
    #data is generate by y = 2x+e
    #'e' is sample from a normal distribution
    x_batch = np.linspace(-1,1,101)
    y_batch = 2*x_batch + np.random.randn(*x_batch.shape)*0.3
    return x_batch, y_batch
def linear_regression():
    #y = Wx
    x = tf.placeholder(tf.float32,shape=(None,),name='x')
    y = tf.placeholder(tf.float32,shape=(None,),name='y')
    with tf.variable_scope('lreg') as scope:
        w = tf.Variable(np.random.normal(),name ='W')
        y_pred = tf.multiply(w,x)
        loss = tf.reduce_mean(tf.square(y_pred-y))
    return x,y,loss,y_pred

def run():
    x_batch,y_batch = generate_dataset()
    x,y,loss,y_pred = linear_regression()
    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    init = tf.global_variables_initializer() #定义函数初始化图中的变量
    with tf.Session() as session: #建立会话
        session.run(init)
        feed_dict  = {x:x_batch,y:y_batch}
        for _ in range(30):#迭代次数
            loss_val, _ = session.run([loss,optimizer],feed_dict)#更新参数
            print('loss:',loss_val.mean()) #这里对损失求了平均
        y_pred_batch = session.run(y_pred,{x:x_batch})
    plt.figure(1)
    plt.scatter(x_batch,y_batch)
    plt.plot(x_batch,y_pred_batch)
    plt.savefig('./plot.png')
if __name__=='__main__':
    run()
