import cv2
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    # 画像ファイルを受け取る
    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # 前処理：グレースケール＆2値化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 輪郭抽出
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return jsonify({'error': 'No object detected'}), 400

    # 最大輪郭を料理とみなす
    food_contour = max(contours, key=cv2.contourArea)

    # 重心を計算
    M = cv2.moments(food_contour)
    if M['m00'] == 0:
        cx, cy = 0, 0
    else:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

    # 外接矩形
    x, y, w, h = cv2.boundingRect(food_contour)

    # 結果を返す
    result = {
        'center': {'x': cx, 'y': cy},
        'bounding_box': {'x': x, 'y': y, 'width': w, 'height': h}
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
