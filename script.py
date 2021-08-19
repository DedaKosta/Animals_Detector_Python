import numpy as np
import cv2 as cv
import imutils
import argparse
import time

def pyramid(image, scale=1.5, minSize=(30, 30)):
	yield image
	while True:
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
		yield image
def sliding_window(image, stepSize, windowSize):
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
image = cv.imread(args["image"])
image = image[22:742, 252:1692]
(winW, winH) = (180, 180)



for resized in pyramid(image, scale=1.5):
	for (x, y, window) in sliding_window(resized, stepSize=180, windowSize=(winW, winH)):
		if window.shape[0] != winH or window.shape[1] != winW:
			continue

		blob = cv.dnn.blobFromImage(window, 1, (224, 224), (104, 117, 123))

		net = cv.dnn.readNetFromCaffe("bvlc_googlenet.prototxt", "bvlc_googlenet.caffemodel")

		blob = cv.dnn.blobFromImage(window, 1, (224, 224), (104, 117, 123))
		net.setInput(blob)
		start = time.time()
		preds = net.forward()
		end = time.time()

		rows = open("synset_words.txt").read().strip().split("\n")
		classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

		idxs = np.argsort(preds[0])[::-1][:5]
		for (i, idx) in enumerate(idxs):
			if preds[0][idx] == "cat" or preds[0][idx] == "dog":
				text = "{}, {:.2f}%".format(classes[idx], preds[0][idx] * 100)
				cv.putText(window, text, (5, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		clone = resized.copy()
		cv.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
		cv.imshow("Window", clone)
		cv.waitKey(1)
		time.sleep(0.025)

cv.imshow("Output", image)
cv.imwrite("output.jpg", image)
cv.waitKey(0)
cv.destroyAllWindows()