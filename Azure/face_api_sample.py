import http.client
import urllib.request, urllib.parse, urllib.error
import base64
import cv2
import numpy as np
import json
import sys

# APIキー直書きなのでこのままコミットしないこと！
headers = {
    # Request headers
    'Content-Type': 'application/octet-stream',
    'Ocp-Apim-Subscription-Key': '36d3c371750047a38d5d4e0c0463f65f',
}

params = urllib.parse.urlencode({
    # Request parameters
    'returnFaceId': 'true',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': 'age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise',
})

def display_expression(data,img):
    font = cv2.FONT_HERSHEY_PLAIN
    font_size = 2

    data = json.loads(data)
    for face in data:
        # 顔の範囲
        f_rec  =  face['faceRectangle']
        width  =  f_rec['width']
        height =  f_rec['height']
        left   =  f_rec['left']
        top    =  f_rec['top']
        cv2.rectangle(img,(left,top),(left+width,top+height),(0,200,0),2)

        # 顔の属性
        f_attr = face['faceAttributes']
        gender = f_attr['gender']
        age = f_attr['age']

        # 感情なども取得できる

        cv2.putText(img, gender, (left, 30+top+height), font, font_size, (0, 200, 0), 2)
        cv2.putText(img, str(age), (left, 60+top+height), font, font_size, (0, 200, 0), 2)

if __name__ == '__main__':
    if len(sys.argv) < 1:
        quit()
    # argv[0]は実行しているpythonプログラム名が入っている
    file_path = sys.argv[1]
    #print(file_path)
    
    f = open(file_path, 'rb')
    body = f.read()
    f.close()

    conn = http.client.HTTPSConnection('westus.api.cognitive.microsoft.com')
#    conn.request("POST", "/face/v1.0/detections?%s" % params, open(file_path, 'rb'), headers)
    conn.request("POST", "/face/v1.0/detect?%s" % params, body, headers)
    response = conn.getresponse()
    data = response.read()

    print(data)

    img = cv2.imread(file_path)
    display_expression(data, img)

    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    conn.close()