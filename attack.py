# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from cleverhans.attacks import Model
from tensorflow.contrib.slim.nets import inception
import os
import csv
from scipy.misc import imread
from scipy.misc import imresize
from PIL import Image
import numpy as np
tf.logging.set_verbosity(tf.logging.ERROR)


tf.flags.DEFINE_string(
    'checkpoint_path_inception', '', 'Path to checkpoint for inception network.')
tf.flags.DEFINE_string(
    'checkpoint_path_resnet', '', 'Path to checkpoint for resnet network.')
tf.flags.DEFINE_string(
    'checkpoint_path_vgg', '', 'Path to checkpoint for vgg network.')
tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')
tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')
tf.flags.DEFINE_integer(
    'image_width', 224, 'Width of each input images.')
tf.flags.DEFINE_integer(
    'image_height', 224, 'Height of each input images.')
tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')
tf.flags.DEFINE_integer(
    'num_classes', 110, 'Number of Classes')
FLAGS = tf.flags.FLAGS

def load_images(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    labels = np.zeros(batch_shape[0], dtype=np.int32)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    with open(os.path.join(input_dir, 'dev.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepath = os.path.join(input_dir, row['filename'])
            with open(filepath,'rb') as f:
                raw_image = imread(f, mode='RGB').astype(np.float)
                image = imresize(raw_image, [FLAGS.image_height, FLAGS.image_width]) / 255.0
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            images[idx, :, :, :] = image * 2.0 - 1.0
            labels[idx] = int(row['targetedLabel'])
            filenames.append(os.path.basename(filepath))
            idx += 1
            if idx == batch_size:
                yield filenames, images, labels
                filenames = []
                images = np.zeros(batch_shape)
                labels = np.zeros(batch_shape[0], dtype=np.int32)
                idx = 0
        if idx > 0:
            yield filenames, images, labels


def save_images(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with open(os.path.join(output_dir, filename), 'wb') as f:
            img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
            # resize back to [299, 299]
            r_img = imresize(img, [299, 299])
            Image.fromarray(r_img).save(f, format='PNG')


#saver=tf.train.import_meta_graph('resnet_v1_50/model.ckpt-49800.meta')
#saver=tf.train.import_meta_graph('resnet_v1_50/model.ckpt-49800.meta')
class InceptionModel(Model):
    """Model class for CleverHans library."""
    def __init__(self, nb_classes):
        super(InceptionModel, self).__init__(nb_classes=nb_classes,
                                             needs_dummy_fprop=True)
        self.built = False

    def __call__(self, x_input, return_logits=False):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_v1_arg_scope()):
            _, end_points = inception.inception_v1(
                x_input, num_classes=self.nb_classes, is_training=False,
                reuse=reuse,scope='InceptionV1')
        self.built = True
        self.logits = end_points['Logits']
        # Strip off the extra reshape op at the output
        self.probs = end_points['Predictions'].op.inputs[0]
        if return_logits:
            return self.logits
        else:
            return self.probs

    def get_logits(self, x_input):
        return self(x_input, return_logits=True)

    def get_probs(self, x_input):
        return self(x_input)

class Resnet(Model):
    """Model class for CleverHans library."""
    def __init__(self, nb_classes):
        super(Resnet, self).__init__(nb_classes=nb_classes,
                                             needs_dummy_fprop=True)
        self.built = False

    def __call__(self, x_input, return_logits=False):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
            logits, end_points = nets.resnet_v1.resnet_v1_50(
                x_input, num_classes=self.nb_classes, is_training=False,
                reuse=reuse,scope='resnet_v1_50')
        self.built = True
        self.logits = logits
        # Strip off the extra reshape op at the output
        self.probs = end_points['predictions'].op.inputs[0]
        if return_logits:
            return self.logits
        else:
            return self.probs

    def get_logits(self, x_input):
        return self(x_input, return_logits=True)

    def get_probs(self, x_input):
        return self(x_input)
#saver=tf.train.import_meta_graph('resnet_v1_50/model.ckpt-49800.meta')
class Vgg_16(Model):
    """Model class for CleverHans library."""
    def __init__(self, nb_classes):
        super(Vgg_16, self).__init__(nb_classes=nb_classes,
                                             needs_dummy_fprop=True)
        self.built = False

    def __call__(self, x_input, return_logits=False):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(nets.vgg.vgg_arg_scope()):
            logits,_= nets.vgg.vgg_16(x_input, num_classes=self.nb_classes, is_training=False,scope='vgg_16')
        self.built = True
        self.logits = logits
        # Strip off the extra reshape op at the output
        self.probs = logits
        if return_logits:
            return self.logits
        else:
            return self.probs

    def get_logits(self, x_input):
        return self(x_input, return_logits=True)

    def get_probs(self, x_input):
        return self(x_input)

def main(_):

    sess=tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))

    batch_shape= [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    batch_size=FLAGS.batch_size
    nb_classes = FLAGS.num_classes
    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    tf.logging.set_verbosity(tf.logging.INFO)
    
    image=tf.Variable(tf.zeros(batch_shape))
    x = tf.placeholder(tf.float32, (batch_shape))

    x_hat = image # our trainable adversarial input
    assign_op = tf.assign(x_hat, x)
    x_scale_resnet=((x_hat + 1.0) * 0.5) * 255.0 #重新从-1--1到0--255
    x_resnet_0=x_scale_resnet[:,:,:,0]-_R_MEAN
    x_resnet_1=x_scale_resnet[:,:,:,1]-_G_MEAN
    x_resnet_2=x_scale_resnet[:,:,:,2]-_B_MEAN
    x_resnet_input=tf.stack([x_resnet_0,x_resnet_1,x_resnet_2],3)

    model_inception=InceptionModel(nb_classes)
    model_resnet=Resnet(nb_classes)
    model_vgg=Vgg_16(nb_classes)

    logits_resnet=model_resnet.get_logits(x_resnet_input)
    logits_inception=model_inception.get_logits(image)
    logits_vgg=model_vgg.get_logits(x_resnet_input)

    #模型融合
    logits=(tf.reshape(logits_resnet,(-1,nb_classes))+logits_vgg+logits_inception)/3.0

    #加载模型
    saver0 = tf.train.Saver(slim.get_model_variables(scope='InceptionV1'))
    saver1 = tf.train.Saver(slim.get_model_variables(scope='resnet_v1_50'))
    saver2 = tf.train.Saver(slim.get_model_variables(scope='vgg_16'))

    saver0.restore(sess, FLAGS.checkpoint_path_inception)
    saver1.restore(sess, FLAGS.checkpoint_path_resnet)
    saver2.restore(sess, FLAGS.checkpoint_path_vgg)

    learning_rate = tf.placeholder(tf.float32, ())
    y_hat = tf.placeholder(tf.int32, (batch_size,))
    #momentum=0.5
    labels = tf.one_hot(y_hat, nb_classes)
   
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=[labels])

    optim_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, var_list=[x_hat])


    #optim_step = tf.train.MomentumOptimizer(learning_rate,momentum).minimize(loss, var_list=[x_hat])
    #init_op=tf.global_variables_initializer()

    epsilon = tf.placeholder(tf.float32, ())

    below = x - epsilon
    above = x + epsilon
    projected = tf.clip_by_value(tf.clip_by_value(x_hat, below, above), -1, 1)#clip x_hat,并将其约束到0，1之间作为输入。
    with tf.control_dependencies([projected]):#此函数指定某些操作执行的依赖关系，即执行完projected,才能再执行project_step
        project_step = tf.assign(x_hat, projected)


    demo_epsilon = 32.0/255.0 # 一个很小的扰动
    demo_lr = 6e-1
    demo_steps =80

    #sess.run(init_op)
    for filenames, images, tlabels in load_images(FLAGS.input_dir, batch_shape):
        # initialization step #先初始化x_hat为x
        sess.run(assign_op, feed_dict={x: images})
        # projected gradient descent
        for i in range(demo_steps):
            # gradient descent step
            _, loss_value = sess.run(
                [optim_step, loss],
                feed_dict={learning_rate: demo_lr, y_hat: tlabels})
            # project step
            sess.run(project_step, feed_dict={x: images, epsilon: demo_epsilon})
        adv = x_hat.eval() # retrieve the adversarial example
        #classify(adv,tlabels)
        save_images(adv,filenames,FLAGS.output_dir)

if __name__=='__main__':
    tf.app.run()

