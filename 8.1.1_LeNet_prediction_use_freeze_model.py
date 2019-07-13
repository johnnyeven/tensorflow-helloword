import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import utils.input_data as input_data

freeze_model_path = './models/8.1.1/frozen_model.pb'
input_node_name = 'mnist/placeholders/input:0'
prediction_node_name = 'mnist/prediction/prediction_index:0'
mnist = input_data.read_data_sets('./mnist_data', one_hot=True)


def visualization(image_array):
    plt.imshow(image_array, cmap="gray")
    plt.show()


with tf.gfile.GFile(freeze_model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(
        graph_def,
        input_map=None,
        return_elements=None,
        name='mnist',
        op_dict=None,
        producer_op_list=None
    )

    for op in graph.get_operations():
        print(op.name, op.values())

    x = graph.get_tensor_by_name(input_node_name)
    y = graph.get_tensor_by_name(prediction_node_name)

    with tf.Session(graph=graph) as sess:
        rand = random.randint(1, 4999)
        mnist.validation.next_batch(rand)
        valid_x, valid_y = mnist.validation.next_batch(1)
        valid_x_img = np.reshape(valid_x * 255, (28, 28))
        visualization(valid_x_img)
        valid_x_reshaped = np.reshape(valid_x, (1, 28, 28, 1))
        valid_feed = {x: valid_x_reshaped}

        print(sess.run(y, feed_dict=valid_feed))
