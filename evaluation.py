import os
import argparse
from pathlib import Path
# from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS # 이 줄은 필요한 경우에 따라 주석을 해제하세요.

def get_file_list(directory):
    """지정된 디렉토리에서 txt 파일 목록을 반환합니다."""
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]

def calculate_metrics(true_files, pred_files):
    TP = FP = FN = 0  # TN 계산 제외

    for true_file, pred_file in zip(sorted(true_files), sorted(pred_files)):
        with open(true_file, 'r') as f:
            true_data = f.read().splitlines()
        with open(pred_file, 'r') as f:
            pred_data = f.read().splitlines()

        true_groups = sum(1 for line in true_data if line.startswith('1'))
        pred_groups = sum(1 for line in pred_data if line.startswith('1'))

        TP += min(pred_groups, true_groups)
        FP += max(0, pred_groups - true_groups)
        FN += max(0, true_groups - pred_groups)

    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

    return precision, recall, f1_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--true_dir', required=True, help="실제 파일이 있는 디렉토리 경로")
    parser.add_argument('--pred_dir', required=True, help="예측 파일이 있는 디렉토리 경로")

    args = parser.parse_args()

    true_files = get_file_list(args.true_dir)
    pred_files = get_file_list(args.pred_dir)

    precision, recall, f1_score = calculate_metrics(true_files, pred_files)
    print(f"Precision: {precision}\nRecall: {recall}\nF1 Score: {f1_score}")

if __name__ == "__main__":
    main()






