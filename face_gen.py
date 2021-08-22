#for generating the dataset from the original images files
import time
import numpy as np
import os
from PIL import Image

#generate Facesrcub dataset
def face(path):

	if not os.path.exists(path):
		print("no such directory")
		return

	images = []
	labels = []
	names = []

	test_images = []
	test_labels = []

	C = 0

	with os.scandir(path) as dirs:
		for actor in dirs:
			if actor.is_dir():

				img_list = os.listdir(actor)
				total = len(img_list)
				counter = 0
				for item in img_list:
					#print(item)
					if os.path.splitext(item)[1] == '.jpeg':

						img = Image.open(os.path.join(actor,item))
						img = img.resize((112,112))
						img = img.convert('RGB')
						img = np.asarray(img)

						if counter < total * 0.8:
							images.append(img)
							labels.append(C)
						else:
							test_images.append(img)
							test_labels.append(C)

						counter += 1

				names.append(actor.name)
				C += 1
				print(actor.name,C)

		images = np.array(images)
		labels = np.array(labels)
		print(images.shape)
		test_images = np.array(test_images)
		test_labels = np.array(test_labels)
		print(test_images.shape)
		#print(labels)
		#print(len(names))
		#convert labels to one hot vector
		#one_hot = np.eye(len(names))[labels]

		np.savez("face.npz", images = images, labels = labels, names = names)
		np.savez("face_test.npz", images = test_images, labels = test_labels, names = names)


train_size = 30000
test_size = 10000
#generate CelebA dataset
def celeb(path):

	if not os.path.exists(path):
		print("no such directory")
		return

	train = []
	test = []
	C = 0

	with os.scandir(path) as files:
		
		for file in files:
			if os.path.splitext(file)[1] == '.jpg':
				img = Image.open(os.path.join(file))
				#print(img.shape)
				img = img.crop((35,70,143,178))
				#img = cv2.resize(img[70:178,35:143],(112,112))
				img = img.resize((112,112))
				img = img.convert('RGB')
				#img.show()
				#img.close()
				#time.sleep(5)
				img = np.asarray(img)
				if C < train_size:
					train.append(img)
				else:
					test.append(img)
				C += 1

			if C % 500 == 0:
				print("already read {} pictures".format(C))

			if C == train_size + test_size:
				break

	train = np.array(train)
	test = np.array(test)

	print(train.shape)
	print(test.shape)
	np.save("celeba_3w.npy", train)
	np.save("celeba_1w.npy", test)

celeb("./celeba")
#face("./faces")




