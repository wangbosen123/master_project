import numpy as np
from PIL import Image
import os
import termcolor

def mkdir(path):
	if not os.path.exists(path):
		os.mkdir(path)

def imread(fname):
	return imageio.imread(fname) / 255.

def imsave(fname, array):
	img = Image.fromarray(np.uint8(array))
	img.save(fname)

# convert to colored strings
def toRed(content):
	return termcolor.colored(content, "red", attrs=["bold"])

def toGreen(content):
	return termcolor.colored(content,"green",attrs=["bold"])

def toBlue(content):
	return termcolor.colored(content,"blue",attrs=["bold"])

def toCyan(content):
	return termcolor.colored(content,"cyan",attrs=["bold"])

def toYellow(content):
	return termcolor.colored(content,"yellow",attrs=["bold"])

def toMagenta(content):
	return termcolor.colored(content,"magenta",attrs=["bold"])