import numpy as np
import tensorflow as tf
import time
from datetime import datetime

NUM_CLASSES = 200
INITIAL_LEARNING_RATE = 1e-2
ITERATIONS = 500 * 500
WEIGHTS_STDEV = 0.01
BIAS_CONST = 0.01
BATCH_SIZE = 100
LR_ITERATIONS = [20000, 50000]

# def prepare_cifar_data():
#     # set data folder in cifar_utils.py
#     import cifar_utils
#     train_images, train_cls_res, train_cls_vec = cifar_utils.load_training_data()
#     test_images, test_cls_res, test_cls_vec = cifar_utils.load_test_data()
#     train_images = train_images.astype('float32')
#     test_images = test_images.astype('float32')
#     return train_images, train_cls_vec, test_images, test_cls_vec

def prepare_imagenet_data():
    from load_images import load_images
    from sklearn import preprocessing
    num_classes = NUM_CLASSES # max number
    path = '/home/student-2/Dropbox/Work/AXON/DL/data/tiny-imagenet-200'
    X_train, y_train, X_test, y_test = load_images(path, num_classes)
    lb = preprocessing.LabelBinarizer()
    y_train_onehot = lb.fit_transform(y_train)
    y_test_onehot = lb.fit_transform(y_test)

    X_train = np.transpose(X_train,(0, 2, 3, 1))
    X_test = np.transpose(X_test,(0, 2, 3, 1))

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    return X_train, y_train_onehot, X_test, y_test_onehot

def RGB_normalize(X):
    R_mean = np.mean(X[:,:,:,0], keepdims=True, axis=0)
    G_mean = np.mean(X[:,:,:,1], keepdims=True, axis=0)
    B_mean = np.mean(X[:,:,:,2], keepdims=True, axis=0)

    # print np.min(X[:,:,:,0]), np.max(X[:,:,:,0])
    # print np.min(X[:,:,:,1]), np.max(X[:,:,:,1])
    # print np.min(X[:,:,:,2]), np.max(X[:,:,:,2])
    #
    # print R_mean.shape, G_mean.shape, B_mean.shape

    X[:,:,:,0] -= R_mean
    X[:,:,:,1] -= G_mean
    X[:,:,:,2] -= B_mean

    # print X.shape
    #
    # print np.min(X[:,:,:,0]), np.max(X[:,:,:,0])
    # print np.min(X[:,:,:,1]), np.max(X[:,:,:,1])
    # print np.min(X[:,:,:,2]), np.max(X[:,:,:,2])

    return X

def get_logdir():
    """Return unique logdir based on datetime"""
    now = datetime.utcnow().strftime("%m%d%H%M%S")
    logdir = "run-{}".format(now)

    return logdir

def variable_summaries(var):
    pass
    # with tf.name_scope('summaries'):
    #     mean = tf.reduce_mean(var)
    #     tf.summary.scalar('mean', mean)
    # with tf.name_scope('stddev'):
    #     stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    # tf.summary.scalar('stddev', stddev)
    # tf.summary.scalar('max', tf.reduce_max(var))
    # tf.summary.scalar('min', tf.reduce_min(var))
    # tf.summary.histogram('histogram', var)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=WEIGHTS_STDEV)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(BIAS_CONST, shape=[shape])
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def fc_layer(input_tensor, shape, name, act = tf.nn.relu):
    with tf.name_scope(name):
        with tf.name_scope('weights'):
            weights = weight_variable(shape)
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable(shape[-1])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations, weights

def conv_layer(input_tensor, shape, name, act = tf.nn.relu):
    with tf.name_scope(name):
        with tf.name_scope('weights'):
            weights = weight_variable(shape)
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable(shape[-1])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = conv2d(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations, weights

def max_pool_layer(input_tensor, name):
    with tf.name_scope(name):
        h_pool = max_pool_2x2(input_tensor)
    return h_pool

def get_batch_data(x,y,batch_size):

    # perm = np.random.permutation(range(x.shape[0]), batch_size)
    batch_idxs = np.random.choice(range(x.shape[0]), batch_size, replace=False)
    batch_x = x[batch_idxs,:,:,:]
    batch_y = y[batch_idxs,:]
    return batch_x, batch_y


def vgg_C(X, y_):
    # conv1_shape = [3,3,3,64]
    # conv2_shape = [3,3,64,64]
    # # maxpool
    # conv3_shape = [3,3,64,128]
    # conv4_shape = [3,3,128,128]
    # # maxpool
    # conv5_shape = [3,3,128,256]
    # conv6_shape = [3,3,256,256]
    # conv7_shape = [1,1,256,256]
    # # maxpool
    # conv8_shape = [3, 3, 256, 512]
    # conv9_shape = [3, 3, 512, 512]
    # conv10_shape = [1, 1, 512, 512]
    # # maxpool
    # # conv11_shape = [3, 3, 512, 512]
    # # conv12_shape = [3, 3, 512, 512]
    # # conv13_shape = [1, 1, 512, 512]
    # # # maxpool
    # fc1_shape = [4*4*512,4096]
    # fc2_shape = [4096,4096]
    # fc3_shape = [4096,NUM_CLASSES]
    #
    # #block 0
    # conv1_activation, conv1_weights = conv_layer(X, conv1_shape, 'conv1')
    # conv2_activation, conv2_weights = conv_layer(conv1_activation, conv2_shape, 'conv2')
    # max_pool_1 = max_pool_layer(conv2_activation,'maxpool1')
    # #block 1
    # conv3_activation, conv3_weights = conv_layer(max_pool_1, conv3_shape, 'conv3')
    # conv4_activation, conv4_weights = conv_layer(conv3_activation, conv4_shape, 'conv4')
    # max_pool_2 = max_pool_layer(conv4_activation, 'maxpool2')
    # #block 2
    # conv5_activation, conv5_weights = conv_layer(max_pool_2, conv5_shape, 'conv5')
    # conv6_activation, conv6_weights = conv_layer(conv5_activation, conv6_shape, 'conv6')
    # conv7_activation, conv7_weights = conv_layer(conv6_activation, conv7_shape, 'conv7')
    # max_pool_3 = max_pool_layer(conv7_activation, 'maxpool3')
    # #block 3
    # conv8_activation, conv8_weights = conv_layer(max_pool_3, conv8_shape, 'conv8')
    # conv9_activation, conv9_weights = conv_layer(conv8_activation, conv9_shape, 'conv9')
    # conv10_activation, conv10_weights = conv_layer(conv9_activation, conv10_shape, 'conv10')
    # max_pool_4 = max_pool_layer(conv10_activation, 'maxpool4')
    # #fully connected layers
    # max_pool_4_flattened = tf.reshape(max_pool_4, [-1, fc1_shape[0]])
    # fc1_activation, fc1_weights = fc_layer(max_pool_4_flattened,fc1_shape,'fc1')
    # fc2_activation, fc2_weights = fc_layer(fc1_activation,fc2_shape,'fc2')
    # fc3_activation, fc3_weights = fc_layer(fc2_activation,fc3_shape,'fc3')
    #
    # y = tf.nn.softmax(fc3_activation)

    # A configuration:

    conv1_shape = [3,3,3,64]

    # maxpool

    conv4_shape = [3,3,64,128]
    # maxpool
    conv5_shape = [3,3,128,256]
    conv6_shape = [3,3,256,256]
    # maxpool
    conv8_shape = [3, 3, 256, 512]
    conv9_shape = [3, 3, 512, 512]
    # # maxpool
    fc1_shape = [4*4*512,4096]
    fc2_shape = [4096,4096]
    fc3_shape = [4096,NUM_CLASSES]

    #block 0
    conv1_activation, conv1_weights = conv_layer(X, conv1_shape, 'conv1')
    max_pool_1 = max_pool_layer(conv1_activation,'maxpool1')
    #block 1
    conv4_activation, conv4_weights = conv_layer(max_pool_1, conv4_shape, 'conv4')
    max_pool_2 = max_pool_layer(conv4_activation, 'maxpool2')
    #block 2
    conv5_activation, conv5_weights = conv_layer(max_pool_2, conv5_shape, 'conv5')
    conv6_activation, conv6_weights = conv_layer(conv5_activation, conv6_shape, 'conv6')
    max_pool_3 = max_pool_layer(conv6_activation, 'maxpool3')
    #block 3
    conv8_activation, conv8_weights = conv_layer(max_pool_3, conv8_shape, 'conv8')
    conv9_activation, conv9_weights = conv_layer(conv8_activation, conv9_shape, 'conv9')
    max_pool_4 = max_pool_layer(conv9_activation, 'maxpool4')
    #fully connected layers
    max_pool_4_flattened = tf.reshape(max_pool_4, [-1, fc1_shape[0]])
    fc1_activation, fc1_weights = fc_layer(max_pool_4_flattened,fc1_shape,'fc1')
    fc2_activation, fc2_weights = fc_layer(fc1_activation,fc2_shape,'fc2')
    fc3_activation, fc3_weights = fc_layer(fc2_activation,fc3_shape,'fc3')

    y = tf.nn.softmax(fc3_activation)


    with tf.name_scope('cross_entropy_loss'):
        reg_const = 5e-4
        cross_entropy_logits = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=fc3_activation)
        # cross_entropy_regularized = cross_entropy + reg_const * (tf.nn.l2_loss(conv1_weights) +tf.nn.l2_loss(conv2_weights)
        #                                                          + tf.nn.l2_loss(conv3_weights) + tf.nn.l2_loss(conv4_weights)
        #                                                          + tf.nn.l2_loss(conv5_weights) + tf.nn.l2_loss(conv6_weights)
        #                                                          + tf.nn.l2_loss(conv7_weights) + tf.nn.l2_loss(conv8_weights)
        #                                                          + tf.nn.l2_loss(conv9_weights) + tf.nn.l2_loss(conv10_weights)
        #                                                          + tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc3_weights))
        cross_entropy_regularized = cross_entropy_logits + reg_const * (tf.nn.l2_loss(conv1_weights)
                                                                 + tf.nn.l2_loss(conv4_weights)
                                                                 + tf.nn.l2_loss(conv5_weights) + tf.nn.l2_loss(conv6_weights)
                                                                 + tf.nn.l2_loss(conv8_weights)
                                                                 + tf.nn.l2_loss(conv9_weights)
                                                                 + tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc3_weights))
        cross_entropy_regularized = tf.reduce_mean(cross_entropy_regularized)
        # variable_summaries(cross_entropy_regularized)
        tf.summary.scalar('cross_entropy',cross_entropy_regularized)

    return y, cross_entropy_regularized


#TODO:
# compare to CIFAR (different folders?)
# arcitecture
# shuffle batches once every epoch



def main(_):
    train_images, train_cls_vec, test_images, test_cls_vec = prepare_imagenet_data()
    print train_images.shape,train_cls_vec.shape,test_images.shape,test_cls_vec.shape
    train_images = RGB_normalize(train_images)
    test_images = RGB_normalize(test_images)
    print train_images.shape,train_cls_vec.shape,test_images.shape,test_cls_vec.shape

    x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3]) #cifar is different indices

    y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

    lr = tf.placeholder(tf.float32)

    with tf.name_scope('VGG'):
        y, cross_entropy = vgg_C(x, y_)
        with tf.name_scope('adam_optimizer'):
            tf.summary.scalar('learning_rate', lr)
            train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            accuracy = tf.reduce_mean(correct_prediction)
            # variable_summaries(accuracy)
            tf.summary.scalar('accuracy', accuracy)


        merged = tf.summary.merge_all()

        tolerance = 1e-7
        t0 = time.time()

        saver = tf.train.Saver()

        saver_path = "./checkpoints/" + get_logdir() + "/" + get_logdir() + ".ckpt"

        logs_path = "./logs/" + get_logdir() + "/VGG/"

        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(logs_path + "train", sess.graph)
            test_writer = tf.summary.FileWriter(logs_path + "test")
            sess.run(tf.global_variables_initializer())
            old_cost = 0.0
            learning_rate = INITIAL_LEARNING_RATE
            for i in range(ITERATIONS):
                batch_x, batch_y = get_batch_data(train_images, train_cls_vec, BATCH_SIZE)
                sess.run(train_step, feed_dict={x: batch_x, y_: batch_y, lr: learning_rate})
                new_cost = sess.run(cross_entropy, feed_dict={x: batch_x, y_: batch_y})

                # if(np.square(new_cost-old_cost) < tolerance):
                #     batch_x_test, batch_y_test = test_images, test_cls_vec
                #     cost_empirical, accuracy_empirical = sess.run([cross_entropy, accuracy], feed_dict={x: batch_x_test, y_: batch_y_test})
                #     summary = sess.run(merged, feed_dict={x: batch_x_test, y_: batch_y_test})
                #     test_writer.add_summary(summary, i)
                #     print("Iteration {}, Test cost: {}, Test accuracy: {}".format(i,cost_empirical, accuracy_empirical))
                #     print("tolerance {} reached, stopping".format(tolerance))
                #     break

                if i % 100 == 0:
                    cost_empirical, accuracy_empirical = sess.run([cross_entropy, accuracy], feed_dict={x: batch_x, y_: batch_y})
                    summary = sess.run(merged, feed_dict={x: batch_x, y_: batch_y, lr: learning_rate})
                    train_writer.add_summary(summary, i)
                    print("Iteration {}, time passed: {}, train cost: {}, train accuracy: {}".format(i, time.time() - t0 , cost_empirical, accuracy_empirical))

                if i % 500 == 0:
                    # batch_x_test, batch_y_test = test_images, test_cls_vec
                    batch_x_test, batch_y_test = get_batch_data(test_images, test_cls_vec,BATCH_SIZE)
                    cost_empirical, accuracy_empirical = sess.run([cross_entropy, accuracy], feed_dict={x: batch_x_test, y_: batch_y_test})
                    summary = sess.run(merged, feed_dict={x: batch_x_test, y_: batch_y_test, lr: learning_rate})
                    test_writer.add_summary(summary, i)
                    print("Test cost: {}, Test accuracy: {}".format(cost_empirical, accuracy_empirical))

                if i % 10000 ==0:
                    save_path = saver.save(sess, saver_path)
                    print("Model saved in file: %s" % save_path)

                if i in LR_ITERATIONS:
                    print("Changed learning rate, iteration {}".format(i))
                    learning_rate /= 10.0

if __name__ == '__main__':
    tf.app.run(main=main)