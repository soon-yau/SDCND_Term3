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
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer3_out)
    """
    # TODO: Implement function
    model_path=vgg_path+'/saved_model.pb'

    with sess.as_default():
        tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
        layer3_out=sess.graph.get_tensor_by_name("layer3_out:0")
        layer4_out=sess.graph.get_tensor_by_name("layer4_out:0")
        layer7_out=sess.graph.get_tensor_by_name("layer7_out:0")
        image_input = sess.graph.get_tensor_by_name("image_input:0")
        keep_prob = sess.graph.get_tensor_by_name("keep_prob:0")

        # to display in tensorboard
        file_writer=tf.summary.FileWriter('./logs/1')
        file_writer.add_graph(sess.graph)        
    print("Load VGG successful!")    

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    #print("layer 7", vgg_layer7_out.get_shape())
    #print("layer 4", vgg_layer4_out.get_shape())
    #print("layer 3", vgg_layer3_out.get_shape())

    # upsample layer 3 by 8x   
    l3_x1=tf.layers.conv2d_transpose(vgg_layer3_out, \
                                     num_classes, 16, strides=1*8, padding='same')

    # upsample layer 4 by 16x   
    l4_x2=tf.layers.conv2d_transpose(vgg_layer4_out, \
                                     num_classes, 8, strides=2*8, padding='same')
                                     
    # upsample layer 7 by 32x    
    l7_x4=tf.layers.conv2d_transpose(vgg_layer7_out, \
                                     num_classes, 8, strides=4*8, padding='same')     

    # Fuse all layers                                     
    output = tf.add(l7_x4, l4_x2)
    output = tf.add(output, l3_x1)                                     
    
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
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer,(-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=correct_label))
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
    # TODO: Implement function
    #saver = tf.train.Saver()
    steps = 0
    print_every=1

    lr = 1e-5
    prob = 0.5
    
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            for images, labels in get_batches_fn(batch_size):
               
                _,loss = sess.run([train_op,cross_entropy_loss], \
                                    feed_dict={ input_image:images, \
                                                correct_label:labels,\
                                                learning_rate:lr,\
                                                keep_prob:prob})
                steps+=1
                
                if steps % print_every == 0:
                    print("Epoch {}/{}...".format(e+1, epochs),
                            "Loss: {:.4f}...".format(loss))
                
                    # Save losses to view after training
            #saver.save(sess, './checkpoints/generator.ckpt')

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'                        
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        #for images, labels in get_batches_fn(1):
        #    print(type(labels))
        # TODO: Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = \
                load_vgg(sess, os.path.join(data_dir,'vgg'))

        nn_last_layer= layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        
        correct_label = tf.placeholder(tf.float32)
        learning_rate = tf.placeholder(tf.float32)

        logits, train_op, cross_entropy_loss = \
                optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        epochs = 10
        batch_size = 1
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, image_input,
                    correct_label, keep_prob, learning_rate)
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

        # OPTIONAL: Apply the trained model to a video
if __name__ == '__main__':
    run()
