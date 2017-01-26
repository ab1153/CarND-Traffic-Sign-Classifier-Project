import tensorflow as tf

def logits(x, keep):    
    mu = 0
    sigma = 0.01
    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x16.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 16), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(16))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)
    # Pooling.  Output = 14x14x16.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional+Inception. Output = 14 x 14 x 112
    conv2_1_W = tf.Variable(tf.truncated_normal(shape=(1, 1, 16, 16), mean = mu, stddev = sigma))
    conv2_1_b = tf.Variable(tf.zeros(16))
    conv2_1   = tf.nn.conv2d(conv1, conv2_1_W, strides=[1, 1, 1, 1], padding='SAME') + conv2_1_b

    conv2_2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 16, 32), mean = mu, stddev = sigma))
    conv2_2_b = tf.Variable(tf.zeros(32))
    conv2_2   = tf.nn.conv2d(conv1, conv2_2_W, strides=[1, 1, 1, 1], padding='SAME') + conv2_2_b

    conv2_3_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 64), mean = mu, stddev = sigma))
    conv2_3_b = tf.Variable(tf.zeros(64))
    conv2_3   = tf.nn.conv2d(conv1, conv2_3_W, strides=[1, 1, 1, 1], padding='SAME') + conv2_3_b    
    
    conv2 = tf.concat(3, [conv2_1, conv2_2, conv2_3])
    conv2 = tf.nn.relu(conv2)
    # additional drop
    conv2 = tf.nn.dropout(conv2, keep_prob = keep)
    # Pooling.  Output = 7x7x112
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 3: Convolutional. Output = 5x5x43.
    conv3_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 112, 43), mean = mu, stddev = sigma))
    conv3_b = tf.Variable(tf.zeros(43))
    conv3   = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_b
    conv3 = tf.nn.relu(conv3)
    # global avg pooling Output = 1x1x43
    conv3 = tf.nn.avg_pool(conv3, ksize=[1, 5, 5, 1], strides=[1, 1, 1, 1], padding='VALID')
    # flatten 
    logits = tf.reshape(conv3, [-1, 43])
    return logits