from flask import Flask, request
from flask_cors import CORS
from flask import render_template
import os,sys,random,string
#import operator
#import pymysql

basedir = os.path.abspath(os.path.dirname(__file__))  # 定义一个根目录 用于保存图片用

app = Flask(__name__)
CORS(app, supports_credentials=True)  # 设置跨域

# http://127.0.0.1:8095
@app.route('/')
def hello_world():
    return render_template('maskcheck-index.html')

# http://127.0.0.1:8095/test
@app.route('/test')
def testhtml():
    return render_template('uploadtest.html')

# http://127.0.0.1:8095/information
@app.route('/information', methods=['GET', 'POST'])
def register():
    dict = request.args
    print(dict)
    return '<script language="javascript" type="text/javascript"> window.location.href=\'https://www.cnblogs.com/Amb1tion100\'; </script>'

def return_img_stream(img_local_path):
    """
    工具函数:
    获取本地图片流
    :param img_local_path:文件单张图片的本地绝对路径
    :return: 图片流
    """
    import base64
    img_stream = ''
    with open(img_local_path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream).decode()
    return img_stream

# http://127.0.0.1:8095/upload
@app.route('/upload', methods=['GET', 'POST'])
def editorData():
    # 获取图片文件 name = upload
    img = request.files.get('image')
    # 定义一个图片存放的位置 存放在static下面
    path = basedir + "/static/img/"
    # 图片名称 使用随机数防止重复
    imgName = str(random.randint(0, 999999)) + img.filename
    # 图片path和名称组成图片的保存路径
    file_path = path + imgName
    # 保存图片
    img.save(file_path)
    # url是图片的路径
    url = '/static/img/' + imgName

    img_path = url
    img_stream = return_img_stream(file_path)
    return render_template('maskcheck-result.html', img_stream=img_stream)

# http://127.0.0.1:8095/assets
@app.route('/assets', methods=['GET', 'POST'])
def returnAssets():
    print(request)
    return "d"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8095, debug=True)
