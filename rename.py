# Read image.
from os import walk
import os 
path = "data"

filenames = next(walk(path), (None, None, []))[2] 

count = 0
for name in filenames:
	p = os.path.join(path, name)
	new_name = os.path.join(path, name.split(".")[0]+".jpg")
	os.rename(p, new_name)
	count = count + 1