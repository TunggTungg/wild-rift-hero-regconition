import cv2
import numpy as np
import pickle
from PIL import Image
from os import walk
from load_model import load_embb_model
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input
from numpy.linalg import norm
import pandas as pd
from tqdm import tqdm

def crop_circle(path, size):
	img = cv2.imread(path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = img[:,:img.shape[1]//2,:]
	# Convert to grayscale.
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Blur using 3 * 3 kernel.
	gray_blurred = cv2.blur(gray, (3, 3))

	# Apply Hough transform on the blurred image.
	detected_circles = cv2.HoughCircles(gray_blurred, 
										cv2.HOUGH_GRADIENT, 1.2, 15, param1 = 50,
										param2 = 35, minRadius = 1, maxRadius = 80)

	# Draw circles that are detected.
	if detected_circles is not None:
	# Convert the circle parameters a, b and r to integers.
		detected_circles = np.uint16(np.around(detected_circles))
		for pt in detected_circles[0, :]:
			a, b, r = pt[0], pt[1], pt[2]
			gap = int(r/4)
			roi = img[b-r+gap:b+r-gap, a-r+gap:a+r-gap]
			roi = cv2.resize(roi, (size,size), interpolation = cv2.INTER_AREA)
			return roi
	else: 
		return None


if __name__ == '__main__':
	
	input_size = 70
	model = load_embb_model(input_size)

	with open('faces_emb_data.pkl', 'rb') as f:
	    (emb_images, name_images) = pickle.load(f)

	emb_images = np.array(emb_images).reshape(len(name_images),model.output.shape[-1])

	output = {"Path": [], "Prediction": []}
	root_path = input('Input your image_path: ')
	filenames = next(walk(root_path), (None, None, []))[2]
	filenames.sort()

	for path in tqdm(filenames):
		img = crop_circle(os.path.join(root_path,path), input_size)
		try:
			img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
			# preprocess the image
			img = preprocess_input(img)
			# extract features
			feature = model.predict(img, verbose=0)
			cosine = np.dot(feature,emb_images.T)/(norm(emb_images, axis=1)*norm(feature))
			predict = name_images[np.argmax(cosine)]
			output["Path"].append(path)
			output["Prediction"].append(predict)
		except:
			output["Path"].append(path)
			output["Prediction"].append("Cannot find the circle in the image (hero's face). Maybe the image is too blurry!")
	
	output = pd.DataFrame(data=output)
	output.to_csv("output.csv")
	print("Check file output.csv for the result!")