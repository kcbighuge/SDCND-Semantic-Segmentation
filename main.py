import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # DONE: Implement function
    #  Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)

    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)

    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)

    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # DONE: Implement function
    layer7_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, 
                                  padding='same', 
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=.01),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    output = tf.layers.conv2d_transpose(layer7_1x1, num_classes, 4, 
                                        strides=(2,2), padding='same', 
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=.01),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    # scale output of layers 3 & 4 before feeding to 1x1 convolution
    pool3_out_scaled = tf.multiply(vgg_layer3_out, 0.0001, name='pool3_out_scaled')
    pool4_out_scaled = tf.multiply(vgg_layer4_out, 0.01, name='pool4_out_scaled')

    layer4_1x1 = tf.layers.conv2d(pool4_out_scaled, num_classes, 1, 
                                  padding='same', 
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=.01),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    output = tf.add(output, layer4_1x1)
    output = tf.layers.conv2d_transpose(output, num_classes, 4, 
                                        strides=(2,2), 
                                        padding='same', 
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=.01),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    layer3_1x1 = tf.layers.conv2d(pool3_out_scaled, num_classes, 1, 
                                  padding='same', 
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=.01),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    output = tf.add(output, layer3_1x1)
    output = tf.layers.conv2d_transpose(output, num_classes, 16, 
                                        strides=(8,8), 
                                        padding='same', 
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=.01),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    return output

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # DONE: Implement function
    # Convert 4D output tensor to 2D with shape (num pixels, num classes)
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    # Define loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_loss = tf.reduce_mean(cross_entropy)

    # Need to include additional term for regularization loss
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    cross_entropy_loss = cross_entropy_loss + sum(reg_losses)

    # Define optimization
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # DONE: Implement function
    for e in range(epochs):

        i = 0
        for images, labels in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss], 
                               feed_dict={input_image: images, 
                                          correct_label: labels, 
                                          keep_prob: 0.5, 
                                          learning_rate: 1e-4})
            if i % 4 == 0:
                print("Epoch {}/{}. Training loss: {:.4f}\n".format(e+1, epochs, loss))
            i += 1

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    # Download data: http://kitti.is.tue.mpg.de/kitti/data_road.zip
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # TODO: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # DONE: Build NN using load_vgg, layers, and optimize function
        # Define placeholders
        correct_label = tf.placeholder(dtype=tf.float32, shape=[None,None,None,num_classes])
        learning_rate = tf.placeholder(dtype=tf.float32)

        # Load pre-trained VGG model
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        
        # Build skp-layers using VGG layers
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # DONE: Train NN using the train_nn function
        # Initialize variables
        sess.run(tf.global_variables_initializer())

        # Set training parameters
        epochs = 20
        batch_size = 8

        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, 
            correct_label, keep_prob, learning_rate)

        # DONE: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # TODO: Apply the trained model to a video


if __name__ == '__main__':
    run()
