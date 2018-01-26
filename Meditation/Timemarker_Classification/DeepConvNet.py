import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from Meditation.Timemarker_Classification.DataStructureParser import dataStructureParser
from sklearn.metrics import confusion_matrix
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm


def optimize(num_iterations):
    global total_iterations
    start_time = time.time()
    current_batch_index = 0
    for i in range(total_iterations,
                   total_iterations + num_iterations):

        x_batch = np.array(trainX[current_batch_index : current_batch_index + train_batch_size])
        y_true_batch = np.array(trainY[current_batch_index : current_batch_index + train_batch_size])
        current_batch_index += train_batch_size
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        session.run(optimizer, feed_dict = feed_dict_train)
        if i % 1 == 0:
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
            print(msg.format(i + 1, acc))

    total_iterations += num_iterations
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))



def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))
def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size_x,        # Width and height of each filter.
                   filter_size_y,
                   num_filters,        # Number of filters.
                   activation_function,
                   pooling_filter_size_x,
                   pooling_filter_size_y,
                   use_pooling=True):  # Use 2x2 max-pooling.


    shape = [filter_size_x, filter_size_y, num_input_channels, num_filters]

    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME', data_format="NHWC")

    layer += biases
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, pooling_filter_size_x, pooling_filter_size_y, 1],
                               strides=[1, pooling_filter_size_x, pooling_filter_size_y, 1],
                               padding='SAME')

    # TODO: Add a probability value in order to do the testing
    layer = tf.nn.dropout(layer, 0.5)
    # TODO: Create a Bool for the Batch Normalizer (Training/Testing)
    layer = batch_norm(layer)

    if (activation_function == 'relu'):
        layer = tf.nn.relu(layer)
    if (activation_function == 'elu'):
        layer = tf.nn.elu(layer)
    return layer, weights


def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features
def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 activation_function):

    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases
    if (activation_function == 'relu'):
        layer = tf.nn.relu(layer)
    if (activation_function == 'elu'):
        layer = tf.nn.elu(layer)

    return layer

#Data to train and test
trainX, trainY, testX, testY = dataStructureParser().getFeaturesAndLabels()


#Convolutional Layer 1
filter_size_1_x = 10
filter_size_1_y = 1
num_filters_1 = 25

#Convolutional Layer 2
filter_size_2_x = 25
filter_size_2_y = 6
num_filters_2 = 25

#Convolutional Layer 3
filter_size_3_x = 10
filter_size_3_y = 25
num_filters_3 = 50

#Convolutional Layer 4
filter_size_4_x = 10
filter_size_4_y = 50
num_filters_4 = 100

#Convolutional Layer 5
filter_size_5_x = 10
filter_size_5_y = 100
num_filters_5 = 200

#Fully Connected Layer
num_fully_connected_1 = 400

#data.test.cls = np.argmax(data.test.labels, axis=1)


img_size_x = 6
img_size_y = 500
img_size_flat = img_size_x * img_size_y
img_shape = (img_size_x, img_size_y)
num_channels = 1
num_classes = 5

#Placeholder variable for input
x = tf.placeholder(tf.float32, shape=[None, img_size_x, img_size_y], name='x')
x_image = tf.reshape(x, [-1, img_size_x, img_size_y, num_channels])

#Placeholder Variable for true output of the image
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)


layer_conv1, weights_conv1 = new_conv_layer(input=x_image,
                    num_input_channels = num_channels,
                    filter_size_x = filter_size_1_x,
                    filter_size_y = filter_size_1_y,
                    num_filters = num_filters_1,
                    activation_function= 'linear',
                    pooling_filter_size_x = 0,
                    pooling_filter_size_y = 0,
                    use_pooling = False)



layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
                    num_input_channels=num_filters_1,
                    filter_size_x = filter_size_2_x,
                    filter_size_y = filter_size_2_y,
                    num_filters = num_filters_2,
                    activation_function= 'elu',
                    pooling_filter_size_x = 3,
                    pooling_filter_size_y = 1,
                    use_pooling = True)

layer_conv3, weights_conv3 = new_conv_layer(input=layer_conv2,
                    num_input_channels=num_filters_2,
                    filter_size_x = filter_size_3_x,
                    filter_size_y = filter_size_3_y,
                    num_filters = num_filters_3,
                    activation_function= 'elu',
                    pooling_filter_size_x=3,
                    pooling_filter_size_y=1,
                    use_pooling = True)

layer_conv4, weights_conv4 = new_conv_layer(input=layer_conv3,
                    num_input_channels=num_filters_3,
                    filter_size_x = filter_size_4_x,
                    filter_size_y = filter_size_4_y,
                    num_filters = num_filters_4,
                    activation_function= 'elu',
                    pooling_filter_size_x=3,
                    pooling_filter_size_y=1,
                    use_pooling = True)

layer_conv5, weights_conv5 = new_conv_layer(input=layer_conv4,
                    num_input_channels=num_filters_4,
                    filter_size_x = filter_size_5_x,
                    filter_size_y = filter_size_5_y,
                    num_filters = num_filters_5,
                    activation_function= 'elu',
                    pooling_filter_size_x=3,
                    pooling_filter_size_y=1,
                    use_pooling = True)

layer_flat, num_features = flatten_layer(layer_conv5)

layer_fc_1 = new_fc_layer(input=layer_flat,
                          num_inputs=num_features,
                          num_outputs=num_fully_connected_1,
                          activation_function='elu')

layer_fc_2 = new_fc_layer(input=layer_fc_1,
                         num_inputs=num_fully_connected_1,
                         num_outputs=num_classes,
                         activation_function='elu')


y_pred = tf.nn.softmax(layer_fc_2)
y_pred_cls = tf.argmax(y_pred, axis=1)


cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc_2,
                                                        labels=y_true)

cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())

train_batch_size = 64
total_iterations = 0




def plot_confusion_matrix(cls_pred):

    cls_true = testY
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)
    print(cm)
    plt.matshow(cm)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

test_batch_size = 128

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    num_test = len(testY)
    #num_test = 256
    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    i = 0

    while i < num_test:
        j = min(i + test_batch_size, num_test)
        images = testX[i:j]
        labels = testY[i:j]
        feed_dict = {x: images,
                     y_true: labels}
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        i = j

    cls_true = testY
    cls_true = np.argmax(cls_true, axis=1)
    correct = (cls_true == cls_pred)
    correct_sum = correct.sum()
    acc = float(correct_sum) / num_test
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))
    if show_example_errors:
        print("Example errors:")
        #plot_example_errors(cls_pred=cls_pred, correct=correct)
    if show_confusion_matrix:
        print("Confusion Matrix:")
        #plot_confusion_matrix(cls_pred=cls_pred)



#optimize(3000)
saver = tf.train.Saver()
saver.restore(session, "tmp/deep_convnet_weights.ckpt")
print_test_accuracy()
#save_path = saver.save(session, "tmp/deep_convnet_weights.ckpt")
#print_test_accuracy()
