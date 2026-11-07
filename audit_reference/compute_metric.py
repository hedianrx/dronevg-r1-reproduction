import re
import math
from tqdm import tqdm
import os
from PIL import Image, ImageDraw
import numpy as np
import json
import argparse
from collections import defaultdict

def calculate_iou(box1, box2):
    """
    快速计算两个水平矩形框的 IoU（Intersection over Union）。

    参数:
    - box1, box2: (x1, y1, x2, y2) 矩形框，表示左上角和右下角的坐标。
    
    返回:
    - IoU (float): 交并比（Intersection over Union）
    """
    
    # 计算交集矩形的左上角和右下角坐标
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    
    # 计算交集的宽和高，如果没有交集则宽或高为0
    inter_width = xi2 - xi1
    inter_height = yi2 - yi1
    
    if inter_width <= 0 or inter_height <= 0:
        return 0.0  # 没有交集
    
    # 交集面积
    inter_area = inter_width * inter_height
    
    # 计算每个矩形框的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 并集面积 = 两个矩形的面积之和 - 交集面积
    union_area = box1_area + box2_area - inter_area
    
    # 计算IoU
    return inter_area / union_area

def extract_bboxes(output):
    """
    Extract bounding box coordinates from the given string using regular expressions.
    :param output: String containing bounding box coordinates in the format {<bx_left><by_top><bx_right><by_bottom>|θ}
    :return: List of bounding boxes, each in the format [bx_left, by_top, bx_right, by_bottom, θ]
    """
    # 修改正则表达式，确保最后一个数字和管道符号能够正确匹配
    pattern = r'\[([0-9., ]+)\]'
    matches = re.findall(pattern, output)
    bboxes = [list(map(float, match.split(","))) for match in matches]
    return bboxes

# 读取JSONL文件并将每行解析为Python字典，存入列表
def load_jsonl(filename):
    data = []
    with open(filename, 'r') as jsonl_file:
        for line in jsonl_file:
            data.append(json.loads(line.strip()))
    return data

def folder_creat_if_not_exist(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some paths.')
    parser.add_argument('--image-folder', required=True, help='Image directory')
    parser.add_argument('--answers-file', required=True, help='Target jsonl directory')
    parser.add_argument('--vis-dir', default=None, help='Base URL for the API')

    args = parser.parse_args()

    # 从 jsonl 文件中加载数据
    predictions = load_jsonl(args.answers_file)
    total_cnt = len(predictions)
    
    # 使用 defaultdict 来存储每个类别的统计信息
    class_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'format_error': 0})
    overall_stats = {'total': 0, 'correct': 0, 'format_error': 0}
    
    for i, predict in tqdm(enumerate(predictions), total=total_cnt):
        answer = predict['answer'].strip()
        gt_bbox = predict['bbox']
        pred_bbox = extract_bboxes(answer)
        class_name = predict.get('class', 'unknown').strip()  # 获取类别，如果没有则使用'unknown'

        # 更新总体统计
        overall_stats['total'] += 1
        class_stats[class_name]['total'] += 1

        ori_img_path = os.path.join(args.image_folder, predict['dataset'], predict['image_id'])
        image = Image.open(ori_img_path)
        width, height = image.size
        vis_dir = args.vis_dir

        if vis_dir:
            folder_creat_if_not_exist(vis_dir)
            draw = ImageDraw.Draw(image)
        
        try:
            # compute IoU
            iou_score = calculate_iou(gt_bbox, pred_bbox[0])
            if iou_score >= 0.5:
                overall_stats['correct'] += 1
                class_stats[class_name]['correct'] += 1

            if vis_dir:
                draw.rectangle(gt_bbox, outline="red", width=5)
                draw.rectangle(pred_bbox[0], outline="blue", width=5)
                image.save(os.path.join(vis_dir, f"{predict['image_id'].split('.')[0]}_{i}.jpg"))
        except Exception as e:
            print(f"Error processing image {predict['image_id']}: {e}")
            print(f"Predicted bbox: {pred_bbox}")
            overall_stats['format_error'] += 1
            class_stats[class_name]['format_error'] += 1
            continue
        
    print(f"Evaluating Overall Performance:")
    print(f'Total samples: {overall_stats["total"]}')
    print(f'Correct predictions: {overall_stats["correct"]}')
    print(f'Format errors: {overall_stats["format_error"]}')
    print(f'Precision @ 0.5: {overall_stats["correct"] / overall_stats["total"] if overall_stats["total"] > 0 else 0}')
    print(f'Format error ratio: {overall_stats["format_error"] / overall_stats["total"] if overall_stats["total"] > 0 else 0}')
    print('\n')

    print(f"Evaluating by Class:")
    for class_name, stats in class_stats.items():
        print(f"Class: {class_name}")
        print(f'  Total samples: {stats["total"]}')
        print(f'  Correct predictions: {stats["correct"]}')
        print(f'  Format errors: {stats["format_error"]}')
        print(f'  Precision @ 0.5: {stats["correct"] / stats["total"] if stats["total"] > 0 else 0}')
        print(f'  Format error ratio: {stats["format_error"] / stats["total"] if stats["total"] > 0 else 0}')
        print('\n')