import argparse
import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")
import random
random.seed(10)

import sys

from tensorflow import keras
import numpy as np
from fawkes.utils import init_gpu, load_extractor, load_victim_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Model

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
num_classes = 20

def load_data(datapath):
    data = np.load(datapath)
    images = data['images']
    fawkes = data['fawkes']
    labels = data['labels']
    return images, fawkes, labels

class Face(object):
    #load feature dataset
    def __init__(self, args, test_size = 0.3):
        super(Face, self).__init__()

        if args.lowkey == 1:
            self.dir = 'faces/lowkey' 
        else:
            self.dir = 'faces/fawkes'

        #base dataset
        self.base = self.dir + '.npz'
        self.datapath = os.path.join(self.dir, args.datapath)
        print('=====loading eval data at:', self.datapath)

        self.origin,_,_ = load_data(self.base)
        self.images, self.fawkes, self.labels = load_data(self.datapath)
        #o_recon, f_recon, _ = load_data(self.datapath)
        #self.labels = to_categorical(self.labels, num_classes=num_classes)

        #partition the dataset into training and testing for each label with same test size
        image_train = np.copy(self.images)
        label_train = np.copy(self.labels)
        fawkes_train = np.copy(self.fawkes)
        origin = np.copy(self.origin)

        #for each class
        label_test = []
        image_test = []

        for i in range(num_classes):
            #get the index array of label i
            idx = np.where(label_train == i )[0]

            test_idx = random.sample(range(idx[0],idx[-1]+1), int(test_size * idx.shape[0]))
            
            #test images are orignal, no fawkes and no recon
            image_test.append(origin[test_idx])
            label_test.append(label_train[test_idx])

            image_train = np.delete(image_train, test_idx, axis = 0)
            label_train = np.delete(label_train, test_idx, axis = 0)
            fawkes_train = np.delete(fawkes_train, test_idx, axis = 0)
            origin = np.delete(origin, test_idx, axis = 0)
        
        self.label_test = np.concatenate(label_test, axis = 0)
        self.image_test = np.concatenate(image_test, axis = 0)   
        
        print(self.image_test.shape)

        self.image_train = image_train
        self.label_train = label_train
        self.fawkes_train = fawkes_train
        self.label_train_hot = to_categorical(self.label_train, num_classes=num_classes)
        self.label_test_hot = to_categorical(self.label_test, num_classes=num_classes)


    def get_label(self):
        return self.label_train_hot, self.label_test_hot

    def get_image(self):
        return self.image_train, self.image_test

    #form dataset and replace the specific label with cloaked images feature
    def replace(self, user = 0):
        image_train = np.copy(self.image_train)
        idx_train = np.where(self.label_train == user)
        image_train[idx_train] = self.fawkes_train[idx_train]

        image_test = np.copy(self.image_test)
        idx_test = np.where(self.label_test == user)

        return image_train, image_test, idx_train, idx_test

def filt_user(image, label, idx):
    #get testset for cloaked user
    label_user = label[idx]
    image_user = image[idx]

    label_clean = np.delete(label, idx, axis = 0)
    image_clean = np.delete(image, idx, axis = 0)

    return image_user, label_user, image_clean, label_clean


def main():
    sess = init_gpu(args.gpu)
    face = Face(args)

    train_clean = 0
    test_clean = 0
    train_user = 0
    test_user = 0

    #for user in range(num_classes):
    for user in [0]:
        base_model = load_extractor('extractor_0')
        model = load_victim_model(num_classes, base_model.model, True )
        label_train,label_test = face.get_label()
        image_train,image_test, idx_train, idx_test= face.replace(user)

        #get user dataset
        image_test_user, label_test_user, image_test_clean, label_test_clean = filt_user(image_test, label_test, idx_test)
        image_train_user, label_train_user, image_train_clean, label_train_clean = filt_user(image_train, label_train, idx_train)
        
        model.fit(image_train, label_train, batch_size = args.batch_size,
                        epochs=100, verbose=0)
        
        _, train_accu_clean = model.evaluate(image_train_clean, label_train_clean)
        _, test_accu_clean = model.evaluate(image_test_clean, label_test_clean)

        _, train_accu_user = model.evaluate(image_train_user, label_train_user)
        _, test_accu_user = model.evaluate(image_test_user, label_test_user)

        train_clean += train_accu_clean
        test_clean += test_accu_clean

        train_user += train_accu_user
        test_user += test_accu_user

        print("==============USER {}================".format(user))
        print("Train accu (clean): {:.4f}".format(train_accu_clean))
        print("Test accu (clean): {:.4f}".format(test_accu_clean))
        print("Train acc (user cloaked): {:.4f}".format(train_accu_user))
        print("Test acc (user cloaked): {:.4f}".format(test_accu_user))
        print("Protection rate: {:.4f}".format(1 - test_accu_user))
    '''
    train_clean /= num_classes
    test_clean /= num_classes

    train_user /= num_classes
    test_user /= num_classes

    print("==============SUMMARY================")
    print("Train accu (clean): {:.2f}".format(train_clean))
    print("Test accu (clean): {:.2f}".format(test_clean))
    print("Train acc (user cloaked): {:.2f}".format(train_user))        
    print("Test acc (user cloaked): {:.2f}".format(test_user))
    print("Protection rate: {:.2f}".format(1 - test_user))
    '''

        
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str,
                        help='GPU id', default='0')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--datapath', '-d', type=str,
                        help='the file name for test data', default='fawkes.npz')
    parser.add_argument('--mode', '-m', type=int,
                        help='0 for nn, 1 for linear', default = 0)
    parser.add_argument('--denoise', '-de', type= bool, default = False)
    parser.add_argument('--feature', '-f', type=int,
                        help ='feature extractor, 0 for fawkes, 1 for lowkey', default = 0)
    parser.add_argument('--lowkey', '-l', type=int,
                        help ='test data, 0 for fawkes, 1 for lowkey', default = 0)

    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()
