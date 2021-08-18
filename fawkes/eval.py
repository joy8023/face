import argparse
import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")
import random
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
#from fawkes.align_face import aligner
#from fawkes.utils import init_gpu, load_extractor, load_victim_model, preprocess, Faces, load_image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

num_class = 20

class Feature(object):
    #load feature dataset
    def __init__(self, datapath,test_size = 0.3 ):
        super(Feature, self).__init__()
        self.datapath = datapath
        self.data = np.load(self.datapath)
        self.images = self.data['images']
        self.fawkes = self.data['fawkes']
        self.labels = self.data['labels']

        #partition the dataset into training and testing for each label with same test size

        image_train = np.copy(self.images)
        label_train = np.copy(self.labels)
        #for each class
        label_test = []
        image_test = []
        for i in range(num_class):
            #get the index array of label i
            idx = np.where(label_train == i )[0]
            idx_start = idx[0]
            idx_end = idx_start + int(test_size* idx.shape[0])+1
            #print(idx_start,idx_end)

            image_test.append(image_train[idx_start:idx_end])
            label_test.append(label_train[idx_start:idx_end])

            image_train = np.delete(image_train, range(idx_start, idx_end), axis = 0)
            label_train = np.delete(label_train, range(idx_start, idx_end), axis = 0)

        
        self.label_test = np.concatenate(label_test, axis = 0)
        self.image_test = np.concatenate(image_test, axis = 0)   
        print(self.image_test.shape)
        print(self.label_test.shape)
        self.image_train = image_train
        self.label_train = label_train

    def get_label(self):
        return self.label_train, self.label_test

    def get_image(self):
        return self.image_train, self.image_test

    #form dataset and replace the specific label with cloaked images feature
    def replace(self, user = 0):
        image_train = np.copy(self.image_train)
        idx_train = np.where(self.label_train == user)
        image_train[idx_train] = self.fawkes[idx_train]

        image_test = np.copy(self.image_test)
        idx_test = np.where(self.label_test == user)
        
        image_test[idx_test] = self.fawkes[idx_test]

        return image_train,image_test, idx_train, idx_test

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
    parser.add_argument('--path', '-p', type=str,
                        help='the path of feature set', default='faces/fawkes_feature.npz')
    parser.add_argument('--mode', '-m', type=int,
                        help='0 for nn, 1 for linear', default= 0)
    args = parser.parse_args(argv[1:])

    dataset = Feature(args.path)
    recognition(dataset, args.mode)


if __name__ == '__main__':
    main(*sys.argv)
















'''        
def load_data(datapath):
    data = np.load(datapath)
    images = data['images']
    fawkes = data['fawkes']
    labels = data['labels']
    return images, fawkes， labels
def filter_image_paths(image_paths):
    new_image_paths = []
    new_images = []
    for p in image_paths:
        img = load_image(p)
        if img is None:
            continue
        new_image_paths.append(p)
        new_images.append(img)
    return new_image_paths, new_images
def get_features(model, paths, ali, batch_size=16):
    paths, images = filter_image_paths(paths)
    faces = Faces(paths, images, ali, verbose=0, eval_local=True, no_align=True)
    faces = faces.cropped_faces
    features = model.predict(faces, verbose=0)
    return features
def get_feature_extractor(base_model="low_extract", custom_weights=None):
    base_model = load_extractor(base_model)
    features = base_model.layers[-1].output
    model = Model(inputs=base_model.input, outputs=features)

    if custom_weights is not None:
        model.load_weights(custom_weights, by_name=True, skip_mismatch=True)

    return model
def get_class(data_dir):
    folders_arr = data_dir.split('/')
    for i in range(len(folders_arr)-1):
        if folders_arr[i+1] == 'face':
            class_name = folders_arr[i]
            return class_name
    return None

def get_facescrub_features(model, ali, dataset_path):
    # get features for all facescrub users
    data_dirs = sorted(glob.glob(os.path.join(dataset_path, "*")))

    classes_train = []
    features_train = []

    classes_test = []
    features_test = []

    for data_dir in data_dirs:
        data_dir += "/face/"
        cls = get_class(data_dir)
        all_pathes = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))

        f = get_features(model, all_pathes, ali)

        test_len = int(0.3 * len(all_pathes))
        test_idx = random.sample(range(len(all_pathes)), test_len)

        f_test = f[test_idx]
        f_train = np.delete(f, test_idx, axis=0)
        features_train.append(f_train)
        classes_train.extend([cls] * len(f_train))
        features_test.append(f_test)
        classes_test.extend([cls] * len(f_test))

    classes_train = np.asarray(classes_train)
    features_train = np.concatenate(features_train, axis=0)

    classes_test = np.asarray(classes_test)
    features_test = np.concatenate(features_test, axis=0)

    return features_train, features_test, classes_train, classes_test

def main():
    sess = init_gpu("0")
    ali = aligner(sess)
    model = get_feature_extractor("low_extract", custom_weights=args.robust_weights)

    random.seed(10)
    print("Extracting features...", flush=True)
    X_train_all, X_test_all, Y_train_all, Y_test_all = get_facescrub_features(model, ali, args.facescrub_dir)

    val_people = args.names_list
    print(val_people)

    base_dir = args.attack_dir

    for name in val_people:
        directory = f"{base_dir}/{name}/face/"
        print(directory)
        image_paths = glob.glob(directory + "*.png") + glob.glob(directory + "*.jpg")
        
        all_pathes_uncloaked = sorted([path for path in image_paths if args.unprotected_file_match in path.split("/")[-1]])
        all_pathes_cloaked = sorted([path for path in image_paths if args.protected_file_match in path.split("/")[-1]])  

        print(name, len(all_pathes_cloaked), len(all_pathes_uncloaked))
        assert len(all_pathes_cloaked) == len(all_pathes_uncloaked)
        
        f_cloaked = get_features(model, all_pathes_cloaked, ali)
        f_uncloaked = get_features(model, all_pathes_uncloaked, ali)

        random.seed(10)
        test_frac = 0.3
        test_idx = random.sample(range(len(all_pathes_cloaked)), int(test_frac * len(all_pathes_cloaked)))

        f_train_cloaked = np.delete(f_cloaked, test_idx, axis=0)
        f_test_cloaked = f_cloaked[test_idx]
 
        f_train_uncloaked = np.delete(f_uncloaked, test_idx, axis=0)
        f_test_uncloaked = f_uncloaked[test_idx]

        if args.classifier == "linear":
            clf1 = LogisticRegression(random_state=0, n_jobs=-1, warm_start=False) 
            clf1 = make_pipeline(StandardScaler(), clf1)
        else:
            clf1 = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)

        idx_train = np.asarray([y != name for y in Y_train_all])
        idx_test = np.asarray([y != name for y in Y_test_all])
        print(np.sum(idx_train), np.sum(idx_test))

        # with cloaking
        X_train = np.concatenate((X_train_all[idx_train], f_train_cloaked))
        Y_train = np.concatenate((Y_train_all[idx_train], [name] * len(f_train_cloaked)))
        clf1 = clf1.fit(X_train, Y_train)

        print("Test acc: {:.2f}".format(clf1.score(X_test_all[idx_test], Y_test_all[idx_test])))
        print("Train acc (user cloaked): {:.2f}".format(clf1.score(f_train_cloaked, [name] * len(f_train_cloaked))))
        print("Test acc (user cloaked): {:.2f}".format(clf1.score(f_test_uncloaked, [name] * len(f_test_uncloaked))))
        print("Protection rate: {:.2f}".format(1-clf1.score(f_test_uncloaked, [name] * len(f_test_uncloaked))))
        print(flush=True)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_model', type=str,
                        help='the feature extractor', default='low_extract')
    parser.add_argument('--classifier', type=str,
                        help='the classifier', default='NN')
    parser.add_argument('--robust-weights', type=str, 
                        help='robust weights', default=None)
    parser.add_argument('--names-list', nargs='+', default=[], help="names of attacking users")
    parser.add_argument('--facescrub-dir', help='path to unprotected facescrub directory', default="facescrub/download/")
    parser.add_argument('--attack-dir', help='path to protected facescrub directory', default="facescrub_attacked/download/")
    parser.add_argument('--unprotected-file-match', type=str,
                        help='pattern to match protected pictures', default='.jpg')
    parser.add_argument('--protected-file-match', type=str,
                        help='pattern to match protected pictures', default='high_cloaked.png')
    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()
'''