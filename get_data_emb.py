import glob, pickle
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg19 import  preprocess_input
from load_model import load_embb_model
from os import walk

images_name = []
images_emb = []

input_size = 70

model = load_embb_model(input_size)

filenames = next(walk("data/"), (None, None, []))[2] 

for image in filenames:
    name = image.split("/")[-1].split(".")[0]
    print(name)
    # load the image
    img = load_img('data/' + image, target_size=(input_size, input_size))

    # convert pixel to numpy array
    img = img_to_array(img)
    # reshape the image for the model
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

    # preprocess the image
    img = preprocess_input(img)
    # extract features
    feature = model.predict(img, verbose=0)

    images_emb.append(feature)
    images_name.append(name)

with open('faces_emb_data.pkl', 'wb') as f:
    pickle.dump((images_emb, images_name), f)