'''
Implementation Vanilla GAN using Tensorlfow 
This code is written for learning GAN purpose
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

class VanillaGAN:
    def __init__(self):
        '''
        define input placeholders, weights
        '''
        # inputs
        self.Z = tf.placeholder(tf.float32, [None, 100], "noise")
        self.X = tf.placeholder(tf.float32, [None, 784])
        
        # weights
        self.G_weight_1 = tf.Variable(self.xavier_init([100, 128]))
        self.G_bias_1 = tf.Variable(tf.zeros([128]))

        self.G_weight_2 = tf.Variable(self.xavier_init([128, 784]))
        self.G_bias_2 = tf.Variable(tf.zeros([784]))

        self.D_weight_1 = tf.Variable(self.xavier_init([784, 128]))
        self.D_bias_1 = tf.Variable(tf.zeros([128]))

        self.D_weight_2 = tf.Variable(self.xavier_init([128, 1]))
        self.D_bias_2 = tf.Variable(tf.zeros([1]))

        # generated output from noise
        self.G_generated_output = self.generator(self.Z)

        # build loss function
        self.D_real_output, self.D_real_logit = self.discriminator(self.X)
        self.D_fake_output, self.D_fake_logit = self.discriminator(self.G_generated_output)

        # these bad loss function make NAN while training!!!
        self.D_loss = -tf.reduce_mean(tf.log(self.D_real_output) + tf.log(1. - self.D_fake_output))
        self.G_loss = -tf.reduce_mean(tf.log(self.D_fake_output))

        # prepare dataset MNIST
        self.mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

    def generator(self, z):
        '''
        this function takes noise as input and then generate the 28x28 image output
        '''
        # ReLU(input * weight + bias) of layer 1
        G_h1 = tf.nn.relu(tf.matmul(z, self.G_weight_1) + self.G_bias_1)

        # layer 2 sigmoid(input * weight + bias)
        G_output = tf.sigmoid(tf.matmul(G_h1, self.G_weight_2) + self.G_bias_2)

        return G_output


    def discriminator(self, x):
        '''
        This function take a real/fake image then discriminate whether it is real or fake
        '''
        # layer 1: relu(input * weight + bias)
        D_h1 = tf.nn.relu(tf.matmul(x, self.D_weight_1) + self.D_bias_1)

        # layer 2
        D_logit_2 = tf.matmul(D_h1, self.D_weight_2) + self.D_bias_2
        D_output = tf.sigmoid(D_logit_2)

        return D_output, D_logit_2

    def xavier_init(self, size):
        xavier_stddev = 1. / tf.sqrt(size[0] / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)


    def generate_Z(self, m, n):
        return np.random.uniform(-1., 1., size=[m, n])

    def plot(self, samples):
        fig = plt.figure(figsize=(5, 5))
        gs = gridspec.GridSpec(5, 5)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        return fig

    def train(self, num_it = 1000000, batch_size=128):
        '''
        this function for training our GAN model
        '''
        # build optimizer 
        D_optimizer = tf.train.AdamOptimizer().minimize(self.D_loss, var_list=[self.D_weight_1, self.D_bias_1, self.D_weight_2, self.D_bias_2])
        G_optimizer = tf.train.AdamOptimizer().minimize(self.G_loss, var_list=[self.G_weight_1, self.G_bias_1, self.G_weight_2, self.G_bias_2])

        # prepare output folder for saving output images while training
        if not os.path.isdir('./output'):
            os.mkdir('./output')

        # init sess
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        i = 0
        
        for it in range(num_it):
            if it % 1000 == 0:
                # generate sample from current trained model
                samples = sess.run(self.G_generated_output, feed_dict={self.Z: self.generate_Z(25, 100)})

                # save the image
                fig = self.plot(samples)
                plt.savefig('output/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                i += 1
                plt.close(fig)
            
            train_batch, _ = self.mnist.train.next_batch(batch_size)

            _, D_current_loss = sess.run([D_optimizer, self.D_loss], feed_dict={self.X: train_batch, self.Z: self.generate_Z(batch_size, 100)})
            _, G_current_loss = sess.run([G_optimizer, self.G_loss], feed_dict={self.X: train_batch, self.Z: self.generate_Z(batch_size, 100)})

            if it % 100 == 0:
                print('=============> Iter: %d' % it)
                print('Discriminator loss: {:.4}'. format(D_current_loss))
                print('Generator loss: {:.4}'.format(G_current_loss))
                


if __name__ == "__main__":
    print('Preparing dataset and model...')
    gan = VanillaGAN()

    print('Start training GAN...')
    gan.train()

