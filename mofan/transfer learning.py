# author: viaeou
from urllib.request import urlretrieve
import os
import numpy as np
import tensorflow as tf
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt

def download():
    categories = ['tiger', 'kittycat']
    for category in categories:
        os.makedirs('./for_transfer_learning/data/%s'%category, exist_ok=True)
        with open('./for_transfer_learning/imagenet_%s.txt'%category, 'r') as file:
            urls = file.readlines()
            n_urls = len(urls)
            cnt = 0
            for i, url in enumerate(urls):
                if os.path.exists('./for_transfer_learning/data/%s/%s' % (category, url.strip().split('/')[-1])):
                    cnt += 1
                    print('file has exists %d'%cnt)
                    continue
                try:
                    urlretrieve(url.strip(), './for_transfer_learning/data/%s/%s' % (category, url.strip().split('/')[-1]))
                    print('%s %i/%i' % (category, i, n_urls))
                except:
                    print('%s %i/%i' % (category, i, n_urls), 'no_image')


def load_image(path):
    img = skimage.io.imread(path)
    img = img / 255
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    if len(crop_img.shape) == 3:
        resized_image = skimage.transform.resize(crop_img, (224, 224))[None, :, :, :]
        #print(resized_image.shape)
        return resized_image
    else:
        print(path, 'error')


def load_data():
    imgs = {'tiger': [], 'kittycat': []}
    for k in imgs.keys():
        dir = './for_transfer_learning/data/' + k
        for file in os.listdir(dir):
            if not file.lower().endswith('.jpg'):
                continue
            try:
                resized_img = load_image(os.path.join(dir, file))
            except OSError:
                continue
            imgs[k].append(resized_img)
            if len(imgs[k]) == 400:
                break
    tigers_y = np.maximum(20, np.random.randn(len(imgs['tiger']), 1) * 30 + 100)
    cat_y = np.maximum(10, np.random.randn(len(imgs['kittycat']), 1) * 8 + 40)
    return imgs['tiger'], imgs['kittycat'], tigers_y, cat_y


class Vgg16:
    vgg_mean = [103.939, 116.779, 123.68]

    def __init__(self, vgg_npy_path=None, restore_from=None):
        try:
            self.data_dict = np.load(vgg_npy_path, encoding='latin1').item()
        except FileNotFoundError:
            print('Please download VGG16 parameters from ...')

        self.tfx = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.tfy = tf.placeholder(tf.float32, [None, 1])

        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=self.tfx * 255.0)
        bgr = tf.concat(axis=3, values=[
            blue - self.vgg_mean[0],
            green - self.vgg_mean[1],
            red - self.vgg_mean[2]
        ])

        conv1_1 = self.conv_layer(bgr, 'conv1_1')
        conv1_2 = self.conv_layer(conv1_1, 'conv1_2')
        pool1 = self.max_pool(conv1_2, 'pool1')

        conv2_1 = self.conv_layer(pool1, 'conv2_1')
        conv2_2 = self.conv_layer(conv2_1, 'conv2_2')
        pool2 = self.max_pool(conv2_2, 'pool2')

        conv3_1 = self.conv_layer(pool2, 'conv3_1')
        conv3_2 = self.conv_layer(conv3_1, 'conv3_2')
        conv3_3 = self.conv_layer(conv3_2, 'conv3_3')
        pool3 = self.max_pool(conv3_3, 'pool3')

        conv4_1 = self.conv_layer(pool3, 'conv4_1')
        conv4_2 = self.conv_layer(conv4_1, 'conv4_2')
        conv4_3 = self.conv_layer(conv4_2, 'conv4_3')
        pool4 = self.max_pool(conv4_3, 'pool4')

        conv5_1 = self.conv_layer(pool4, 'conv5_1')
        conv5_2 = self.conv_layer(conv5_1, 'conv5_2')
        conv5_3 = self.conv_layer(conv5_2, 'conv5_3')
        pool5 = self.max_pool(conv5_3, 'pool5')

        print('pool5: shape', pool5.shape)
        self.flatten = tf.reshape(pool5, [-1, 7*7*512])
        self.fc6 = tf.layers.dense(self.flatten, 256, tf.nn.relu, name='fc6')
        self.out = tf.layers.dense(self.fc6, 1, name='out')

        self.sess = tf.Session()
        if restore_from:
            saver = tf.train.Saver()
            saver.restore(self.sess, restore_from)
        else:
            self.loss = tf.losses.mean_squared_error(labels=self.tfy, predictions=self.out)
            self.train_op = tf.train.RMSPropOptimizer(0.01).minimize(self.loss)
            self.sess.run(tf.global_variables_initializer())

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            conv = tf.nn.conv2d(bottom, self.data_dict[name][0], [1, 1, 1, 1], padding='SAME')
            lout = tf.nn.relu(tf.nn.bias_add(conv, self.data_dict[name][1]))
        return lout

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def train(self, x, y):
        loss, _ = self.sess.run([self.loss, self.train_op], {self.tfx: x,
                                                             self.tfy: y})
        return loss

    def predict(self, paths):
        fig, axs = plt.subplots(1, 2)
        for i, path in enumerate(paths):
            x = load_image(path)
            length = self.sess.run(self.out, {self.tfx: x})
            axs[i].imshow(x[0])
            axs[i].set_title('Len: %.1f cm' % length)
            axs[i].set_xticks(())
            axs[i].set_yticks(())
        plt.show()

    def save(self, path='./for_transfer_learning/model/transfer_learn'):
        saver = tf.train.Saver()
        saver.save(self.sess, path, write_meta_graph=False)


def train():
    tigers_x, cats_x, tigers_y, cats_y = load_data()
    #print(tigers_y)
    plt.hist(tigers_y, bins=20, label='Tigers')
    plt.hist(cats_y, bins=10, label='Cats')
    plt.legend()
    plt.xlabel('length')
    plt.show()

    xs = np.concatenate(tigers_x + cats_x, axis=0)
    ys = np.concatenate((tigers_y, cats_y), axis=0)
    print(xs.shape)
    print(ys.shape)
    vgg = Vgg16(vgg_npy_path='./for_transfer_learning/vgg16.npy')
    print('Net built')
    for i in range(100):
        b_idx = np.random.randint(0, len(xs), 6)
        print(b_idx)
        train_loss = vgg.train(xs[b_idx], ys[b_idx])
        print(i, 'train loss: ', train_loss)

    vgg.save('./for_transfer_learning/model/transfer_learn')


def eval():
    vgg = Vgg16(vgg_npy_path='./for_transfer_learning/vgg16.npy',
                restore_from='./for_transfer_learning/model/transfer_learn')
    vgg.predict(
        ['./for_transfer_learning/data/kittycat/020.jpg', './for_transfer_learning/data/tiger/02027.jpg']
    )


if __name__ == '__main__':
    # download()
    # train()
    # eval()
    data_dict = np.load('./for_transfer_learning/vgg16.npy', encoding='latin1').item()
    print('conv1_1: ', data_dict['conv1_1'][0].shape, data_dict['conv1_1'][1].shape)
    print('conv1_2: ', data_dict['conv1_2'][0].shape, data_dict['conv1_2'][1].shape)
    print('conv2_1: ', data_dict['conv2_1'][0].shape, data_dict['conv2_1'][1].shape)
    print('conv2_2: ', data_dict['conv2_2'][0].shape, data_dict['conv2_2'][1].shape)
    print('conv3_1: ', data_dict['conv3_1'][0].shape, data_dict['conv3_1'][1].shape)
    print('conv3_2: ', data_dict['conv3_2'][0].shape, data_dict['conv3_2'][1].shape)
    print('conv3_3: ', data_dict['conv3_3'][0].shape, data_dict['conv3_3'][1].shape)
    print('conv4_1: ', data_dict['conv4_1'][0].shape, data_dict['conv4_1'][1].shape)
    print('conv4_2: ', data_dict['conv4_2'][0].shape, data_dict['conv4_2'][1].shape)
    print('conv4_3: ', data_dict['conv4_3'][0].shape, data_dict['conv4_3'][1].shape)
    print('conv5_1: ', data_dict['conv5_1'][0].shape, data_dict['conv5_1'][1].shape)
    print('conv5_2: ', data_dict['conv5_2'][0].shape, data_dict['conv5_2'][1].shape)
    print('conv5_3: ', data_dict['conv5_3'][0].shape, data_dict['conv5_3'][1].shape)
    print('fc6: ', data_dict['fc6'][0].shape, data_dict['fc6'][1].shape)
    print('fc7: ', data_dict['fc7'][0].shape, data_dict['fc7'][1].shape)
    print('fc8: ', data_dict['fc8'][0].shape, data_dict['fc8'][1].shape)


