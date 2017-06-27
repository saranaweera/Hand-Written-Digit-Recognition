from flask import Flask, render_template, request
import io
import re
import base64
import flask

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from scipy.misc import imread, imresize
import numpy as np

app = Flask(__name__)

#read model from pickle
clf = joblib.load('model.full.pkl')


def findEdges(img,xStart):
    started = False
    leftX = 0
    rightX = 0
    for x in range(xStart,img.shape[1]):
        if((np.sum(img[:,x]) != 0) and (started == False)):
            #print('here')
            leftX = x
            started = True
        if((np.sum(img[:,x]) == 0) and (started == True)):
            rightX = x
            break

    started = False
    topY = 0
    bottomY = 0
#     if(rightX <1):
#         return((0,0,0,0))

    for y in range(0,img.shape[0]):
        if((np.sum(img[y,leftX:rightX+1]) != 0) and (started == False)):
            #print('here')
            topY = y
            started = True
        if((np.sum(img[y,leftX:rightX+1]) == 0) and (started == True)):
            bottomY = y
            break

    return((leftX,rightX, topY, bottomY))

def getImage(data):

    # get img data
    imgstr = re.search(b'base64,(.*)', data).group(1)
    #print(imgstr)
    # convert from base64 to bytes and save image
    with open('output3.png','wb') as output:
        output.write(base64.decodebytes(imgstr))

    # reread image
    img = imread('output3.png', mode='L')

    # invert so that black-white switches
    return(np.invert(img))

def breakDownToMultipleImages(img):
    lstDigits = []
    xStart = 0

    while(True):
        edges = findEdges(img,xStart)
        if np.sum(edges) < 1:
            break
        else:
            leftX,rightX, topY, bottomY = edges
            lstDigits.append(img[topY:bottomY, leftX:rightX])
            xStart = rightX

    return(lstDigits)

def addPadding(img):
    j =np.max(img.shape) * 1.8
    #print(j)

    paddingY = int((j - img.shape[0]) // 2)

    topBottomPadding = np.zeros((paddingY, img.shape[1]))
    tempImg = np.concatenate((topBottomPadding, img, topBottomPadding),axis =0)

    paddingX = int((j - tempImg.shape[1]) // 2)

    leftRightPadding = np.zeros((tempImg.shape[0], paddingX))

    #print(leftRightPadding.shape)

    tempImg = np.concatenate((leftRightPadding, tempImg, leftRightPadding),axis =1)

    imgDim = np.min(tempImg.shape)

    tempImg = tempImg[:imgDim,:imgDim]

    return(tempImg)

def getDigitImages(img):
    imgs = breakDownToMultipleImages(img)
    imgs2 = [ imresize(addPadding(im), (28,28)) for im in imgs]
    return(imgs2)


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict/', methods=['GET','POST'])
def predict():

    dataUri = request.get_data()

    img = getImage(dataUri)

    predList = []

    for im in getDigitImages(img):
    # resize image so that it is same size as training samples
    #resizedImg = imresize(img,(28,28))

        #increase contrast
        im[im>(255*.3)] = 255
        im[im<=(255*.3)] = 0

        pred = clf.predict(im.reshape(1,-1))

        predList.append(int(pred[0]))

    results = {'predictions' : " ".join(map(str, predList))}

    return flask.jsonify(results)
    #return('test')



if __name__ == '__main__':
    app.run(debug=True)
