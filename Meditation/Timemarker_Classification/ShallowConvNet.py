import math
import time
from sklearn.metrics import confusion_matrix
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from Meditation.Timemarker_Classification.DataStructureParser import dataStructureParser
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm


def optimize(num_iterations):
    global total_iterations
    global best_validation_accuracy
    global last_improvement

    start_time = time.time()
    current_batch_index = 0

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        x_batch = np.array(trainX[current_batch_index : current_batch_index + batch_size])
        y_true_batch = np.array(trainY[current_batch_index : current_batch_index + batch_size])
        current_batch_index += batch_size
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        session.run(optimizer, feed_dict = feed_dict_train)
        if i % 100 == 0:
            acc_train = session.run(accuracy, feed_dict=feed_dict_train)
            acc_validation, _ = validation_accuracy()
            if (acc_validation > best_validation_accuracy):
                best_validation_accuracy = acc_validation
                last_improvement = total_iterations
                saver.save(session, save_path_name)
                improved_str = '*'
            else:
                improved_str = ''
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}, Validation Acc: {2:>6.1%} {3}"
            print(msg.format(i, acc_train, acc_validation, improved_str))

        if total_iterations - last_improvement > require_improvement:
            print("No improvement has been found after a large number of iterations, process stopped.")
            break
    total_iterations += num_iterations
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

def validation_accuracy():
    correct, _ = predict_cls_validation()
    return cls_accuracy(correct)

def predict_cls_validation():
    return predict_cls(validationX, validationY, validation_class)

def predict_cls(input, one_hot_classification, classification):
    input_length = len(input)
    cls_pred = np.zeros(shape=input_length, dtype=np.int)
    i = 0
    while i < input_length:
        j = min(i + batch_size, input_length)
        feed_dict = {x: input[i : j],
                     y_true: one_hot_classification[i : j]}
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        i = j
    correct = (classification == cls_pred)
    return correct, cls_pred

def cls_accuracy(correct):
    correct_sum = correct.sum()
    acc = float(correct_sum) / len(correct)
    return acc, correct_sum

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input,  # The previous layer.
                   num_input_channels,  # Num. channels in prev. layer.
                   filter_size_x,  # Width and height of each filter.
                   filter_size_y,
                   num_filters,  # Number of filters.
                   activation_function,
                   pooling_filter_size_x,
                   pooling_filter_size_y,
                   use_pooling=True,
                   pooling_stride_x = 0,
                   pooling_stride_y = 0,
                   pooling_type = "max_pooling",
                   dropout_keep_prob = 1.0):

    shape = [filter_size_x, filter_size_y, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    layer += biases
    if use_pooling:
        if pooling_type == "max_pooling":
            layer = tf.nn.max_pool(value=layer,
                                   ksize=[1, pooling_filter_size_x, pooling_filter_size_y, 1],
                                   strides=[1, pooling_stride_x, pooling_stride_y, 1],
                                   padding='SAME')
        if pooling_type == "mean_pooling":
            layer = tf.nn.avg_pool(value=layer,
                                   ksize=[1, pooling_filter_size_x, pooling_filter_size_y, 1],
                                   strides=[1, pooling_stride_x, pooling_stride_y, 1],
                                   padding='SAME')
    layer = tf.nn.dropout(layer, dropout_keep_prob)
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

def plot_conv_weights(weights, input_channel=0):
    w = session.run(weights)
    print("Mean: {0:.5f}, Stdev: {1:.5f}".format(w.mean(), w.std()))
    w_min = np.min(w)
    w_max = np.max(w)
    num_filters = w.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            # Get the weights for the i'th filter of the input channel.
            # The format of this 4-dim tensor is determined by the
            # TensorFlow API.
            img = w[:, :, input_channel, i]
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def plot_confusion_matrix(cls_pred):

    cls_true = np.argmax(testY, axis = 1)
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

#Data to train and test
trainX, trainY, testX, testY, validationX, validationY = dataStructureParser().getFeaturesAndLabels()

test_class = np.argmax(testY, axis = 1)
validation_class = np.argmax(validationY, axis = 1)

#Convolutional Layer 1
filter_size_1_x = 6
filter_size_1_y = 25
num_filters_1 = 40

#Convolutional Layer 2
filter_size_2_x = 6
filter_size_2_y = 6
num_filters_2 = 40

#Fully Connected Layer
num_fully_connected_1 = 300

img_size_x = 6
img_size_y = 500
img_size_flat = img_size_x * img_size_y
img_shape = (img_size_x, img_size_y)
num_channels = 1
num_classes = 2

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
                                            activation_function= 'elu',
                                            pooling_filter_size_x = 0,
                                            pooling_filter_size_y = 0,
                                            use_pooling = False,
                                            dropout_keep_prob=0.5)

layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
                                            num_input_channels=num_filters_1,
                                            filter_size_x = filter_size_2_x,
                                            filter_size_y = filter_size_2_y,
                                            num_filters = num_filters_2,
                                            activation_function= 'elu',
                                            pooling_filter_size_x = 6,
                                            pooling_filter_size_y = 1,
                                            pooling_stride_x = 15,
                                            pooling_stride_y = 1,
                                            use_pooling = True,
                                            pooling_type="mean_pooling",
                                            dropout_keep_prob=0.5)

layer_flat, num_features = flatten_layer(layer_conv2)

layer_fc_1 = new_fc_layer(input=layer_flat,
                          num_inputs=num_features,
                          num_outputs=num_fully_connected_1,
                          activation_function='None')

layer_fc_2 = new_fc_layer(input=layer_fc_1,
                         num_inputs=num_fully_connected_1,
                         num_outputs=num_classes,
                         activation_function='None')

y_pred = tf.nn.softmax(layer_fc_2)
y_pred_cls = tf.argmax(y_pred, axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc_2,
                                                        labels=y_true)

cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
session.run(tf.global_variables_initializer())

batch_size = 128
total_iterations = 0
best_validation_accuracy = 0.0
last_improvement = 0
require_improvement = 1000



def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False,
                        size= len(testY)):

    num_test = size
    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    i = 0

    while i < num_test:
        j = min(i + batch_size, num_test)
        images = testX[i:j]
        labels = testY[i:j]
        feed_dict = {x: images,
                     y_true: labels}
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        i = j
        if i % 10000 == 0:
            print(i)

    cls_true = testY[0:size]
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
        plot_confusion_matrix(cls_pred=cls_pred)

def train_network(num_iterations = 3000):
    optimize(num_iterations)

def test_network(num_test_iterations = len(testY)):
    saver.restore(session, save_path_name)
    print_test_accuracy(size = num_test_iterations, show_confusion_matrix=True)


saver = tf.train.Saver()
save_path_name = "tmp/shallow_convnet_meditation_weights_01.ckpt"

#######################################

mode = input("Please Enter either 'Train' to train the Network, or 'Test' to test it. You should only test once you have trained it.")
if (mode == "Train"):
    print("Just stop the software at any time, the graph will be saved at the previous best validation set.")
    train_network(num_iterations=150000)
if (mode == "Test"):
    print("Testing the network, the confusion matrix and the weights will be posted. ")
    test_network()

#######################################

# TODO: What about an Ensemble CNN, where each one takes in a filter bank of the Wavelet Packet Decomposition
# TODO: Implement a RCNN?


# TODO: Make a real time visualizer of the activity levels, as well as synchrony levels
# TODO: Receive feature visualization for Control and Meditation Visualization, visualize the time band closely (with time markers, colors, etc)
# TODO: Make a multitude of classifiers, and visualize in realtime how activated they are. Have a VAGUE(!) idea of what they try to track
