from imutils.perspective import four_point_transform
import imutils
from imutils import contours
import cv2
from flask import Flask, request, render_template, url_for, jsonify
import numpy as np
from PIL import Image
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html', appName="Intel Image Classification")


@app.route('/predictApi', methods=["POST"])
def api():
    # Get the image from post request
    try:
        if 'fileup' not in request.files or 'fileup2' not in request.files:
            return "Please try again. The Image doesn't exist"
        image = request.files.get('fileup')
        image2 = request.files.get('fileup2')
        score = readIamge(image, image2)
        prediction = score
        return jsonify({'prediction': prediction})
    except:
        return jsonify({'Error': 'Error occur'})


def readIamge(img, img2):
    image1 = Image.open(img)
    image2 = Image.open(img2)
    image_arr = np.array(image1.convert('RGB'))
    image_arr2 = np.array(image2.convert('RGB'))
    ANSWER_KEY_one = procssing(cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY))
    ANSWER_KEY_tow = procssing(cv2.cvtColor(image_arr2, cv2.COLOR_BGR2GRAY))
    correct = []
    for i in range(len(ANSWER_KEY_one)):
        if ANSWER_KEY_one[i] == ANSWER_KEY_tow[i]:
            correct.append(1)
    score = (sum(correct)/5)*100
    return score


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("run code")
    if request.method == 'POST':
        print("image loading....")
        image1 = request.files['fileup2']
        image2 = request.files['fileup']
        score = readIamge(image1, image2)
        prediction = score
        return render_template('index.html', prediction=f' your grade is : {prediction}')
    else:
        return render_template('index.html', appName="Intel Image Classification")
# eXtract correct from orgnal image


def procssing(gray):
    ANSWER_KEY = []
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    cnts = cv2.findContours(
        edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    docCnt = None
    if len(cnts) > 0:
        # sorting the contours according to their size in descending order
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        # looping over the sorted contours
        for c in cnts:
            # approximating the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # if our approximated contour has four points, then we can assume we have found the paper
            if len(approx) == 4:
                docCnt = approx
                break
    #paper = four_point_transform(image, docCnt.reshape(4, 2))
    warped = four_point_transform(gray, docCnt.reshape(4, 2))
    thresh = cv2.threshold(
        warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []
    # looping over the contours
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
            questionCnts.append(c)
    # sorting the question contours top-to-bottom, then initializing the total number of correct answers
    questionCnts = contours.sort_contours(
        questionCnts, method="top-to-bottom")[0]
    # each question has 5 possible answers, to loop over the question in batches of 5
    for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
        cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
        bubbled = None
        # loop over the sorted contours
        for (j, c) in enumerate(cnts):
            # construct a mask that reveals only the current
            # "bubble" for the question
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)
            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)
        ANSWER_KEY.append(bubbled[1])
    return ANSWER_KEY


if __name__ == '__main__':
    app.run(debug=True)
