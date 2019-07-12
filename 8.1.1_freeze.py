import tensorflow as tf
from tensorflow.python.framework import graph_util

model_path = './models/8.1.1/'

checkpoint = tf.train.get_checkpoint_state(model_path)
input_checkpoint = checkpoint.model_checkpoint_path

absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
output_graph = absolute_model_folder + "/frozen_model.pb"

output_node_names = 'prediction/prediction_index'


saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

graph = tf.get_default_graph()
input_graph_ref = graph.as_graph_def()

with tf.Session() as sess:
    saver.restore(sess, input_checkpoint)
    output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_ref, output_node_names.split(','))

    with tf.gfile.GFile(output_graph, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
