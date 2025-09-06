#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

import face_detector as fd
import os

import multiprocessing

class Frame:
    """class to hold information about each frame

    """

    def __init__(self, id, diff):
        self.id = id
        self.diff = diff

    def __lt__(self, other):
        if self.id == other.id:
            return self.id < other.id
        return self.id < other.id

    def __gt__(self, other):
        return other.__lt__(self)

    def __eq__(self, other):
        return self.id == other.id and self.id == other.id

    def __ne__(self, other):
        return not self.__eq__(other)

def drawBoxes(im, boxes):
    x1 = boxes[0]
    y1 = boxes[1]
    x2 = boxes[2]
    y2 = boxes[3]

    cv2.putText(im, 'face', (int(x1), int(y1)), 5, 1.0, [255, 0, 0])
    cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
    return im

def rectify_bbox(bbox, img):
    max_height = img.shape[0]
    max_width = img.shape[1]
    bbox[0] = max(bbox[0], 0)
    bbox[1] = max(bbox[1], 0)
    bbox[2] = min(bbox[2], max_width)
    bbox[3] = min(bbox[3], max_height)
    return bbox

def process(video_dir, save_dir, train_file_list):
    minsize = 20
    total_face = 0
    with open(train_file_list, 'a+') as f_trainList:
        for dir_path, dir_names, _ in os.walk(video_dir):
            # frame_count = 0
            for dir_name in dir_names:
                print('processing directory: ' + dir_path + '/' + dir_name)
                video_dir_name = os.path.join(dir_path, dir_name)
                if dir_name in ['Normal']:
                    label = '0'
                elif dir_name in ['Talking']:
                    label = '2'
                elif dir_name in ['Yawning']:
                    label = '1'
                else:
                    print("Too bad, label invalid.")
                    continue
                # if dir_name in ['Talking']:
                #     label = '0'
                # else:
                #     print("Too bad, label invalid.")
                #     continue
                video_names = os.listdir(video_dir_name)

                for video_name in video_names:
                    # video = cv2.VideoCapture()
                    cap = cv2.VideoCapture(os.path.join(video_dir_name, video_name))

                    curr_frame = None
                    prev_frame = None
                    frame_diffs = []
                    frames = []
                    success, frame = cap.read()
                    i = 0
                    ed = []
                    rmses = []
                    while (success):
                        # 图像转换
                        luv = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                        curr_frame = luv
                        if curr_frame is not None and prev_frame is not None:
                            # # logic here
                            # 计算当前帧与背景的平方差。
                            diff = np.square(prev_frame - curr_frame)
                            # 求欧氏距离（ED）
                            diff_sum_mean = np.sqrt(np.sum(diff))
                            # 求均方根误差 (RMSE)
                            rmse = np.sqrt((np.sum(diff))/(diff.shape[0] * diff.shape[1]))
                            rmses.append(rmse)
                            frame_diffs.append(diff_sum_mean)
                            frame = Frame(i, diff_sum_mean)
                            frames.append(frame)
                        prev_frame = curr_frame
                        i = i + 1
                        success, frame = cap.read()
                    cap.release()

                    
                    ed = frame_diffs

                    rmse = np.sum(rmses)/(i-1)
                    diff_sum_mean = np.sum(frame_diffs) / (i - 1)
                    
                    frame_count = -1
                    cap = cv2.VideoCapture(os.path.join(video_dir_name, video_name))
                    if cap.isOpened():
                        ftotal = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        read_success = True
                    else:
                        read_success = False
                    while read_success:
                        read_success, img = cap.read()
                        try:
                            img.shape
                        except:
                            break

                        frame_count += 1
                        if frame_count % SAMPLE_STEP == 0:
                            # img = img.transpose(2, 0, 1)
                            conf, raw_boxes = fd.get_facebox(image=img, threshold=0.7)
                            if len(raw_boxes) == 0:
                                # if face can't be detected between n frames
                                print('***too sad, the face can not be detected!')
                                continue
                            elif len(raw_boxes) == 1:
                                bbox = raw_boxes[0]
                            elif len(raw_boxes) > 1:
                                #print('---multi detect!')
                                # idx = np.argsort(-raw_boxes[:, 4])
                                # TODO. sort
                                bbox = raw_boxes[0]
                            bbox = rectify_bbox(bbox, img)
                            x = int(bbox[0])
                            y = int(bbox[1])
                            w = int(bbox[2] - bbox[0])
                            h = int(bbox[3] - bbox[1])
                            face = img[y:y + h, x:x + w, :]

                            face_sp = np.shape(face)
                            #if face_sp[0] <= 16 or face_sp[1] <= 16:
                            #    continue
                            face_resize = face

                            face_resize = cv2.resize(face_resize, (CROPPED_WIDTH, CROPPED_HEIGHT))

                            if DEBUG:
                                cv2.imshow('face', face)
                                ch = cv2.waitKey(40000) & 0xFF
                                cv2.imshow('face_trans', face_resize)
                                ch = cv2.waitKey(40000) & 0xFF
                                img = drawBoxes(img, bbox)
                                cv2.imshow('img', img)
                                ch = cv2.waitKey(40000) & 0xFF
                                if ch == 27:
                                    break
                            try:
                                if (ed[frame_count]) > np.sum(frame_diffs) / (i - 1):
                                    print(total_face)
                                    total_face += 1
                                    img_file_name = video_name.split('.')[0]
                                    img_file_name = img_file_name + '-' + str(total_face) + '_' + str(label) + '.jpg'
                                    img_save_path = os.path.join(save_dir, img_file_name)
                                    f_trainList.write(img_save_path + ' ' + label + '\n')
                                    cv2.imwrite(img_save_path, face_resize)

                                elif (rmses[frame_count]) > np.sum(rmses)/(i-1) :
                                    total_face += 1
                                    img_file_name = video_name.split('.')[0]
                                    img_file_name = img_file_name + '-' + str(total_face) + '_' + str(label) + '.jpg'
                                    img_save_path = os.path.join(save_dir, img_file_name)
                                    f_trainList.write(img_save_path + ' ' + label + '\n')
                                    cv2.imwrite(img_save_path, face_resize)
                            except:
                                print("")


                    cap.release()
    print('Total face img: {}'.format(total_face))



if __name__ == "__main__":
    SAMPLE_STEP = 2
    NUM_WORKER = 1
    DEBUG = False
    CROPPED_WIDTH = 112
    CROPPED_HEIGHT = 112

    video_dir = '../dataset/train_lst'
    face_save_dir = 'D:/fatigue-drive-yawning-detection-master/extracted_face/face_image'
    if not os.path.exists(face_save_dir):
        os.makedirs(face_save_dir)
    train_file_list_dir = '../extracted_face/'
    file_list_name = 'test.txt'
    process(video_dir, face_save_dir, train_file_list_dir+file_list_name)

