# -*- coding:utf-8 -*-
import cv2
import time

import argparse
import numpy as np
from PIL import Image
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from load_model.pytorch_loader import load_pytorch_model, pytorch_inference
from flask import Flask, request
from flask_cors import CORS
from flask import render_template
import os,sys,random,string
#import operator
#import pymysql

basedir = os.path.abspath(os.path.dirname(__file__))  # 定义一个根目录 用于保存图片用

app = Flask(__name__)
CORS(app, supports_credentials=True)  # 设置跨域


# model = load_pytorch_model('models/face_mask_detection.pth');
model = load_pytorch_model('models/model360.pth');
# anchor configuration
#feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
feature_map_sizes = [[45, 45], [23, 23], [12, 12], [6, 6], [4, 4]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}


def inference(image,
              conf_thresh=0.5,
              iou_thresh=0.4,
              target_shape=(160, 160),
              draw_result=True,
              show_result=True
              ):
    '''
    Main function of detection inference
    :param image: 3D numpy array of image
    :param conf_thresh: the min threshold of classification probabity.
    :param iou_thresh: the IOU threshold of NMS
    :param target_shape: the model input size.
    :param draw_result: whether to daw bounding box to the image.
    :param show_result: whether to display the image.
    :return:
    '''
    # image = np.copy(image)
    output_info = []
    height, width, _ = image.shape
    image_resized = cv2.resize(image, target_shape)
    image_np = image_resized / 255.0  # 归一化到0~1
    image_exp = np.expand_dims(image_np, axis=0)

    image_transposed = image_exp.transpose((0, 3, 1, 2))

    y_bboxes_output, y_cls_output = pytorch_inference(model, image_transposed)
    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                 bbox_max_scores,
                                                 conf_thresh=conf_thresh,
                                                 iou_thresh=iou_thresh,
                                                 )

    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)

        if draw_result:
            if class_id == 0:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
        output_info.append([class_id, conf, xmin, ymin, xmax, ymax])

    lastimg = 0
    if show_result:
        lastimg = Image.fromarray(image)
    return lastimg


def run_on_video(video_path, output_video_name, conf_thresh):
    cap = cv2.VideoCapture(video_path)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # writer = cv2.VideoWriter(output_video_name, fourcc, int(fps), (int(width), int(height)))
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not cap.isOpened():
        raise ValueError("Video open failed.")
        return
    status = True
    idx = 0
    while status:
        start_stamp = time.time()
        status, img_raw = cap.read()
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        read_frame_stamp = time.time()
        if (status):
            inference(img_raw,
                      conf_thresh,
                      iou_thresh=0.5,
                      target_shape=(360, 360),
                      draw_result=True,
                      show_result=False)
            cv2.imshow('image', img_raw[:, :, ::-1])
            cv2.waitKey(1)
            inference_stamp = time.time()
            # writer.write(img_raw)
            write_frame_stamp = time.time()
            idx += 1
            print("%d of %d" % (idx, total_frames))
            print("read_frame:%f, infer time:%f, write time:%f" % (read_frame_stamp - start_stamp,
                                                                   inference_stamp - read_frame_stamp,
                                                                   write_frame_stamp - inference_stamp))
    # writer.release()

def zyy(imgPath):
    file_path = imgPath + 'new'
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = inference(img, show_result=True, target_shape=(360, 360))
    img.save(file_path)
    return file_path

if __name__ == "__main123__":
    parser = argparse.ArgumentParser(description="Face Mask Detection")
    parser.add_argument('--img-mode', type=int, default=1, help='set 1 to run on image, 0 to run on video.')
    parser.add_argument('--img-path', type=str, help='path to your image.')
    parser.add_argument('--video-path', type=str, default='0', help='path to your video, `0` means to use camera.')

    # parser.add_argument('--hdf5', type=str, help='keras hdf5 file')
    args = parser.parse_args()
    if args.img_mode:
        imgPath = args.img_path
        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inference(img, show_result=True, target_shape=(360, 360))
    else:
        video_path = args.video_path
        if args.video_path == '0':
            video_path = 0
        run_on_video(video_path, '', conf_thresh=0.5)

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
    # img_path是图片的路径
    img_path = '/static/img/' + imgName

    output_path = zyy(img_path)
    img_stream = return_img_stream(output_path)
    return render_template('maskcheck-result.html', img_stream=img_stream)

# http://127.0.0.1:8095/assets
@app.route('/assets', methods=['GET', 'POST'])
def returnAssets():
    print(request)
    return "d"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8095, debug=True)
