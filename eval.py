import argparse
import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")
import random
import sys

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
from utils import get_feature
from resnet import get_feature_resnet
import random

num_class = 20

class Feature(object):
    #load feature dataset
    def __init__(self, args, test_size = 0.3):
        super(Feature, self).__init__()
        self.datapath = args.datapath

        if args.feature == 0:
            self.images, self.fawkes, self.labels = get_feature(self.datapath, denoise = args.denoise)
        else:
            self.images, self.fawkes, self.labels = get_feature_resnet(self.datapath)

        #partition the dataset into training and testing for each label with same test size
        image_train = np.copy(self.images)
        label_train = np.copy(self.labels)
        fawkes_train = np.copy(self.fawkes)

        #for each class
        label_test = []
        image_test = []

        for i in range(num_class):
            #get the index array of label i
            idx = np.where(label_train == i )[0]

            test_idx = random.sample(range(idx[0],idx[-1]+1), int(test_size * idx.shape[0]))
            
            image_test.append(image_train[test_idx])
            label_test.append(label_train[test_idx])

            image_train = np.delete(image_train, test_idx, axis = 0)
            label_train = np.delete(label_train, test_idx, axis = 0)
            fawkes_train = np.delete(fawkes_train, test_idx, axis = 0)

        
        self.label_test = np.concatenate(label_test, axis = 0)
        self.image_test = np.concatenate(image_test, axis = 0)   
        print(self.image_test.shape)

        self.image_train = image_train
        self.label_train = label_train
        self.fawkes_train = fawkes_train

    def get_label(self):
        return self.label_train, self.label_test

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

#filt the data out(for specific user)
def filt_user(image, label, idx):
    #get testset for cloaked user
    label_user = label[idx]
    image_user = image[idx]

    label_clean = np.delete(label, idx, axis = 0)
    image_clean = np.delete(image, idx, axis = 0)

    return image_user, label_user, image_clean, label_clean

#run face recognitiom model over the feature
#0 for nn, 1 for linear
def recognition(feature, mode = 0):
    if mode == 1:
        model = LogisticRegression(random_state=0, n_jobs=-1, warm_start=False) 
        model = make_pipeline(StandardScaler(), model)
    else:
        model = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)

    train_clean = 0
    test_clean = 0
    train_user = 0
    test_user = 0

    #train and test model for each user
    for user in range(num_class):
    #for user in [19]:
        label_train,label_test = feature.get_label()
        image_train,image_test, idx_train, idx_test= feature.replace(user)

        #get user dataset
        image_test_user, label_test_user, image_test_clean, label_test_clean = filt_user(image_test, label_test, idx_test)
        image_train_user, label_train_user, image_train_clean, label_train_clean = filt_user(image_train, label_train, idx_train)
            
        model = model.fit(image_train, label_train)

        train_accu_clean = model.score(image_train_clean, label_train_clean)
        test_accu_clean = model.score(image_test_clean, label_test_clean)

        train_accu_user = model.score(image_train_user, label_train_user)
        test_accu_user = model.score(image_test_user, label_test_user)

        train_clean += train_accu_clean
        test_clean += test_accu_clean

        train_user += train_accu_user
        test_user += test_accu_user

        print("==============USER {}================".format(user))
        print("Train accu (clean): {:.2f}".format(train_accu_clean))
        print("Test accu (clean): {:.2f}".format(test_accu_clean))
        print("Train acc (user cloaked): {:.2f}".format(train_accu_user))
        print("Test acc (user cloaked): {:.2f}".format(test_accu_user))
        print("Protection rate: {:.2f}".format(1 - test_accu_user))

    train_clean /= num_class
    test_clean /= num_class

    train_user /= num_class
    test_user /= num_class

    print("==============SUMMARY================")
    print("Train accu (clean): {:.2f}".format(train_clean))
    print("Test accu (clean): {:.2f}".format(test_clean))
    print("Train acc (user cloaked): {:.2f}".format(train_user))        
    print("Test acc (user cloaked): {:.2f}".format(test_user))
    print("Protection rate: {:.2f}".format(1 - test_user))


def main(*argv):
    if not argv:
        argv = list(sys.argv)

    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', '-d', type=str,
                        help='the path of feature set', default='faces/fawkes.npz')
    parser.add_argument('--mode', '-m', type=int,
                        help='0 for nn, 1 for linear', default = 0)
    parser.add_argument('--denoise', '-de', type= bool, default = False)
    parser.add_argument('--feature', '-f', type=int,
                        help ='feature extractor, 0 for fawkes, 1 for lowkey', default = 0)
    args = parser.parse_args(argv[1:])

    dataset = Feature(args)
    recognition(dataset, args.mode)


if __name__ == '__main__':
    main(*sys.argv)
