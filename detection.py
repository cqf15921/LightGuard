import numpy as np
import os

# 定义数据集的前缀名称，这与 preprocessing.py 生成的文件名一致
datasets = ['ustc_tfc2016', 'cic_iot_2023', 'ton_iot']
processed_dir = "data/processed"

for ds in datasets:
    train_path = os.path.join(processed_dir, f"{ds}_dataset_train.npz")
    test_path = os.path.join(processed_dir, f"{ds}_dataset_test.npz")

    print(f"=== 数据集: {ds.upper()} ===")

    if os.path.exists(train_path):
        train_data = np.load(train_path, allow_pickle=True)
        print(f"  训练集样本数: {len(train_data['images'])}")
    else:
        print(f"  [!] 找不到训练集文件: {train_path}")

    if os.path.exists(test_path):
        test_data = np.load(test_path, allow_pickle=True)
        print(f"  测试集样本数: {len(test_data['images'])}")
    else:
        print(f"  [!] 找不到测试集文件: {test_path}")
    print("-" * 30)