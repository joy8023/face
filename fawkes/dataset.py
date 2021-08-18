import argparse
import glob
import numpy as np
import os
import sys
from PIL import Image
import tensorflow as tf
import tensorflow.keras as keras

def l2_norm(x, axis=1):
    """l2 norm"""
    norm = tf.norm(x, axis=axis, keepdims=True)
    output = x / norm
    return output

class Extractor(object):
    def __init__(self, model):
        self.model = model

    def predict(self, imgs ):
        imgs = imgs / 255.0
        embeds = l2_norm(self.model.predict(imgs,batch_size = 64))
        return embeds

    def __call__(self, x):
        print('flag')
        return self.predict(x,batch_size = 32)

def load_extractor(name):

    model = keras.models.load_model("model/{}.h5".format(name))
    model = Extractor(model)

    return model

def load_data(datapath):
    data = np.load(datapath)
    images = data['images']
    fawkes = data['fawkes']
    labels = data['labels']
    return images, fawkes, labels

def save_feature(datapath, save_path, model):
    images, fawkes, labels = load_data(datapath)
    image_features = model.predict(images)
    fawkes_features = model.predict(fawkes)
    print(image_features.shape)
    np.savez(save_path, images = image_features, fawkes = fawkes_features, labels = labels)
    print('done!')


def load_image(path):
    try:
        img = Image.open(path)
    except PIL.UnidentifiedImageError:
        return None
    except IsADirectoryError:
        return None

    try:
        info = img._getexif()
    except OSError:
        return None

    if info is not None:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = dict(img._getexif().items())
        if orientation in exif.keys():
            if exif[orientation] == 3:
                img = img.rotate(180, expand=True)
            elif exif[orientation] == 6:
                img = img.rotate(270, expand=True)
            elif exif[orientation] == 8:
                img = img.rotate(90, expand=True)
            else:
                pass
    img = img.convert('RGB')
    img = img.resize((112,112))
    image_array = np.asarray(img)

    return image_array

def to_array(l):
    a = np.array(l)
    print(a.shape)
    return a

#save single file for one person images
def save_dataset(image_paths, save_path):
    print("Identify {} files in the directory".format(len(image_paths)))
    new_image_paths = []
    new_images = []
    cloaked_img = []
    #for p in image_paths:
    for i in range(0, len(image_paths), 2):

        img = load_image(image_paths[i])
        cloaked = load_image(image_paths[i+1])
        if img is None:
            print("{} is not an image file, skipped".format(p.split("/")[-1]))
            continue
        file_name = image_paths[i].split("/")[-1].split(".")[0]
        cloaked_name = image_paths[i+1].split("/")[-1].split(".")[0]

        if file_name+'_cloaked' == cloaked_name:
            new_images.append(img)
            cloaked_img.append(cloaked)

    print("Identify {} images in the directory".format(len(new_images)))
    new_images = to_array(new_images)
    cloaked_img = to_array(cloaked_img)
    np.savez(save_path+'.npz', images = new_images, fawkes = cloaked_img)
    #return new_images, cloaked_img

#combine single dataset into one dataset
def combine(data_path):

    data_paths = glob.glob(os.path.join(data_path, "*.npz"))
    labels = []
    data = np.load(data_paths[0])
    images = data['images']
    fawkes = data['fawkes']
    labels.extend([0]*data['images'].shape[0])

    for i in range(1,len(data_paths)):
        print(data_paths[i])
        data = np.load(data_paths[i])

        images = np.concatenate((images,np.squeeze(data['images'])),axis = 0)
        fawkes = np.concatenate((fawkes,np.squeeze(data['fawkes'])), axis = 0)
        labels.extend([i]*len(data['images']))

    print(images.shape)
    print(fawkes.shape)
    labels = to_array(labels)
    print(labels)

    np.savez(os.path.join(data_path, 'fawkes.npz'), images = images, fawkes = fawkes, labels = labels)
    print('done!')


def main(*argv):
    if not argv:
        argv = list(sys.argv)

    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', '-d', type=str,
                        help='the directory that contains images', default='imgs/')
    parser.add_argument('--model_name', '-mn', type=str,
                        help='Extractor,0 or 2', default='extractor_0')
    args = parser.parse_args(argv[1:])


    image_paths = glob.glob(os.path.join(args.directory, "*"))
    #image_paths = [path for path in image_paths if "_cloaked" not in path.split("/")[-1]]
    #save_dataset(sorted(image_paths), args.directory)

    #combine(args.directory)
    model = load_extractor(args.model_name)
    save_feature(args.directory+'/fawkes.npz', args.directory +'/fawkes_feature.npz', model)

if __name__ == '__main__':
    main(*sys.argv)