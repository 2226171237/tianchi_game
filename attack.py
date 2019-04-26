# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim import nets
from scipy.misc import imread
from scipy.misc import imresize
from cleverhans import attacks
from cleverhans.attacks import MomentumIterativeMethod
from cleverhans.attacks import ElasticNetMethod
from cleverhans.attacks import ProjectedGradientDescent
from cleverhans.attacks import Model
from PIL import Image
slim = tf.contrib.slim

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
    'batch_size', 1, 'How many images process at one time.')
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


class InceptionModel(Model):
    """Model class for CleverHans library."""
    def __init__(self, nb_classes):
        super(InceptionModel, self).__init__(nb_classes=nb_classes,
                                             needs_dummy_fprop=True)
        self.built = False

    def __call__(self, x_input, return_logits=False):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(nets.inception.inception_v1_arg_scope()):
            _, end_points = nets.inception.inception_v1(
                x_input, num_classes=self.nb_classes, is_training=False,
                reuse=reuse)
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

class EnsembleModel(Model):
    '''
    三个模型融合
    '''
    def __init__(self, nb_classes):
        super(EnsembleModel, self).__init__(nb_classes=nb_classes,
                                             needs_dummy_fprop=True)
        self.built = False
        self.inception_model=InceptionModel(nb_classes)
        self.resnet_model=Resnet(nb_classes)
        self.vgg_model=Vgg_16(nb_classes)
        self.model_name=['inception','resnet','vgg']
        self.n_classes=nb_classes

    def init_input(self,x_input):
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        x_scale=((x_input + 1.0) * 0.5) * 255.0 #重新从-1--1到0--255
        temp0=x_scale[:,:,:,0]-_R_MEAN
        temp1=x_scale[:,:,:,1]-_G_MEAN
        temp2=x_scale[:,:,:,2]-_B_MEAN
        x_init_input=tf.stack([temp0,temp1,temp2],3)
        return x_init_input
        
    def __call__(self, x_input, return_logits=False):

        reuse = True if self.built else None
        with tf.variable_scope('',reuse=reuse):
            logits_inception=self.inception_model.get_logits(x_input)
            x_init_input=self.init_input(x_input)
            logits_resnet=self.resnet_model.get_logits(x_init_input)
            logits_vgg=self.vgg_model.get_logits(x_init_input)
            self.logits=(tf.reshape(logits_resnet,(-1,self.n_classes))+logits_vgg+logits_inception)/3.0
            self.probs=tf.nn.softmax(self.logits)
        self.built = True
        if return_logits:
            return self.logits
        else:
            return self.probs

    def get_logits(self, x_input):
        return self(x_input, return_logits=True)

    def get_probs(self, x_input):
        return self(x_input) 

def main(_):
    """Run the sample attack"""
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    nb_classes = FLAGS.num_classes
    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        target_class_input = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
        one_hot_target_class = tf.one_hot(target_class_input, nb_classes)
        
        model = EnsembleModel(nb_classes)
        # Run computation
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            
            #攻击方法1，MomentumIterativeMethod
            
            mim = MomentumIterativeMethod(model, sess=sess)
    
            #parse_params(eps=0.3, eps_iter=0.06, nb_iter=10, y=None, ord=inf, decay_factor=1.0,
            #   clip_min=None, clip_max=None, y_target=None, sanity_checks=True, **kwargs)
        
            attack_params = {"eps": 0.2, "eps_iter": 0.01, "clip_min": -1.0, "clip_max": 1.0, \
                             "nb_iter": 10, "decay_factor": 1.0, "y_target": one_hot_target_class}
            
            #攻击方法2，ProjectedGradientDescent
           
            mim2=ProjectedGradientDescent(model,sess=sess)
            #parse_params(eps=0.3, eps_iter=0.05, nb_iter=10, y=None, ord=inf, clip_min=None,
            #              clip_max=None, y_target=None, rand_init=None, rand_minmax=0.3, sanity_checks=True, **kwargs)
            attack_params2={"eps":0.3,"y_target":one_hot_target_class,"nb_iter":10,"clip_min":-1.0,"clip_max":1.0}
            
           
            x_adv1= mim.generate(x_input, **attack_params) #第一生成阶段

            x_adv2=mim2.generate(x_input,**attack_params2) #第二生成阶段

            saver0 = tf.train.Saver(slim.get_model_variables(scope='InceptionV1'))
            saver1 = tf.train.Saver(slim.get_model_variables(scope='resnet_v1_50'))
            saver2 = tf.train.Saver(slim.get_model_variables(scope='vgg_16'))
            saver0.restore(sess, FLAGS.checkpoint_path_inception)
            saver1.restore(sess, FLAGS.checkpoint_path_resnet)
            saver2.restore(sess, FLAGS.checkpoint_path_vgg)

            for filenames, images, tlabels in load_images(FLAGS.input_dir, batch_shape):
                adv_1 = sess.run(x_adv1,
                                      feed_dict={x_input: images,target_class_input: tlabels}) 
                adv_images = sess.run(x_adv2,
                                      feed_dict={x_input: adv_1,target_class_input: tlabels}) 
                save_images(adv_images, filenames, FLAGS.output_dir)

if __name__ == '__main__':
    tf.app.run()
