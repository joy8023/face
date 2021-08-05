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
						#test = Image.fromarray(img,'RGB')
						#test.save('test.jpeg')
						#return

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

#generate CelebA dataset
def celeb(path):

	if not os.path.exists(path):
		print("no such directory")
		return

	images = []
	C = 0

	with os.scandir(path) as files:
		
		for file in files:
			if os.path.splitext(file)[1] == '.jpg' and C < 50000:
				img = cv2.imread(os.path.join(file), cv2.IMREAD_GRAYSCALE)
				#print(img.shape)
				img = cv2.resize(img[70:178,35:143],(64,64))
				#cv2.imshow("img",img)
				#cv2.waitKey(0)
				images.append(img)
				C += 1
	
	images = np.array(images)
	#images = images / 255.0

	print(images.shape)
	#print(images.max())
	np.save("celeba_5w_255.npy", images)



#celeb("./celeba")
face("./faces")

'''
input = np.load("./facescrub.npz")
#print(input)
data = input['images']
print(data.max())
data= data/255.0
print(data[0])
'''
#labels = input['labels']
#names = input['names']
#print(labels)
#one = np.where(labels == 2)
#print(one)
#one_hot = np.eye(len(names))[labels]

#np.savez("facescrub.npz", images = data, labels = one_hot, names = names)




