import argparse

import numpy as np
import predict
import cfg
from network import East


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
    print(img_path, threshold, result_path)

    test_image_dir = os.join(test_dir, cfg.test_image_dir_name)
    test_text_dir = os.join(test_dir, cfg.test_text_dir_name)
    backgrounds = [os.path.join(test_image_dir, file_name) for file_name in os.listdir(test_image_dir)]

    east = East()
    east_detect = east.east_network()
    east_detect.load_weights(cfg.saved_model_weights_file_path)
    for test_case in test_image_dir :
        predict.predict(east_detect, test_case, threshold, result_path)
