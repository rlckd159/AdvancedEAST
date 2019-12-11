import argparse

import os
import numpy as np
from shapely.geometry import Polygon
#import predict
import cfg
#from network import East

def cal_score(gt_total, det_total, matched) :
    precision = matched/det_total
    recall = matched/gt_total
    f1_score = 2*(precision*recall)/(precision+recall)
    return precision, recall, f1_score

def get_iou(pg, pd):
    return (pg.intersection(pd).area)/pg.union(pd).area

def evaluation(gt_dir, det_dir):
    gt_files = [os.path.join(gt_dir, file_name) for file_name in os.listdir(gt_dir)]
    det_files = [os.path.join(det_dir, os.path.basename(gt_file)) for gt_file in gt_files]
    gt_total = 0
    det_total = 0
    matched = 0
    for gt_file_name, det_file_name in zip(gt_files, det_files):
        with open(gt_file_name, 'r', encoding='utf-8-sig') as gt_f:
            anno_list = gt_f.readlines()
        gt_polys = []
        for anno, i in zip(anno_list, range(len(anno_list))):
            if not anno.strip():
                continue
            anno_colums = anno.strip().split(',')
            anno_array = np.array(anno_colums)
            points = np.reshape(anno_array[:8].astype(np.int32), (4, 2))
            gt_poly = Polygon(points)
            gt_polys.append(gt_poly)

        with open(det_file_name, 'r', encoding='utf-8-sig') as det_f:
            anno_list = det_f.readlines()
        det_polys = []
        for anno, i in zip(anno_list, range(len(anno_list))):
            anno_colums = anno.strip().split(',')
            anno_array = np.array(anno_colums)
            points = np.reshape(anno_array[:8].astype(np.int32), (4, 2))
            det_poly = Polygon(points)
            det_polys.append(det_poly)
        lg = len(gt_polys)
        ld = len(det_polys)
        out_shape = [lg, ld]
        iouMat = np.empty(out_shape)
        for gt_num in range(lg) :
            for det_num in range(ld):
                pg = gt_polys[gt_num]
                pd = det_polys[det_num]
                iouMat[gt_num, det_num] = get_iou(pg, pd)
        for gt_num in range(lg):
            for det_num in range(ld):
                if (iouMat[gt_num, det_num] > cfg.iou_threshold) :
                    matched += 1
        gt_total += lg
        det_total += ld
    return cal_score(gt_total, det_total, matched)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p',
                        default='test/',
                        help='test path')
    parser.add_argument('--threshold', '-t',
                        default=cfg.pixel_threshold,
                        help='pixel activation threshold')
    parser.add_argument('--result', '-r',
                        default='result',
                        help='result path')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    test_dir = args.path
    threshold = float(args.threshold)
    result_path = args.result
    print(test_dir, threshold, result_path)


    test_image_dir = os.path.join(test_dir, cfg.test_image_dir_name)
    test_text_dir = os.path.join(test_dir, cfg.test_text_dir_name)
    test_images = [os.path.join(test_image_dir, file_name) for file_name in os.listdir(test_image_dir)]

    # east = East()
    # east_detect = east.east_network()
    # east_detect.load_weights(cfg.saved_model_weights_file_path)
    # for test_case in test_images :
    #     predict.predict(east_detect, test_case, threshold, result_path)

    precision, recall, f1_score = evaluation(test_text_dir, result_path)
    print(precision, recall, f1_score)