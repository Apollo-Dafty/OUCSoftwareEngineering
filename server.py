from flask import Flask, request
from flask_cors import CORS
import os,sys,random,string

basedir = os.path.abspath(os.path.dirname(__file__))  # 定义一个根目录 用于保存图片用


#import operator
#import pymysql

app = Flask(__name__)
CORS(app, supports_credentials=True)  # 设置跨域

@app.route('/req', methods=['GET', 'POST'])
def register():
    dict = request.args
    print(dict)
    return '吕亮nb'
    #return '<script language="javascript" type="text/javascript"> window.location.href=\'https://www.cnblogs.com/Amb1tion100\'; </script>'

@app.route('/upload', methods=['GET', 'POST'])
def editorData():
    # 获取图片文件 name = upload
    img = request.files.get('image')

    # 定义一个图片存放的位置 存放在static下面
    path = basedir + "/static/img/"

    # 图片名称
    imgName = str(random.randint(0, 999999)) + img.filename

    # 图片path和名称组成图片的保存路径
    file_path = path + imgName

    # 保存图片
    img.save(file_path)

    # url是图片的路径
    url = '/static/img/' + imgName
    return url

if __name__ == '__main__':
    app.run(port=8095, debug=True)

