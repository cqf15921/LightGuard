import os
import glob
import argparse
import numpy as np
import pandas as pd
from scapy.all import rdpcap, IP
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# ==========================================
# 1. USTC-TFC2016 原始 PCAP 预处理类
# ==========================================
class LightGuardPreprocessor:
    def __init__(self, input_dir, output_idx_path, img_size=28, truncate_len=784):
        self.input_dir = input_dir
        self.output_idx_path = output_idx_path
        self.img_size = img_size
        self.truncate_len = truncate_len
        os.makedirs(os.path.dirname(self.output_idx_path), exist_ok=True)

    def traffic_cleaning(self, packet):
        if IP in packet:
            return bytes(packet[IP])
        return None

    def traffic_truncation(self, raw_bytes):
        if len(raw_bytes) >= self.truncate_len:
            return raw_bytes[:self.truncate_len]
        else:
            return raw_bytes + b'\x00' * (self.truncate_len - len(raw_bytes))

    def pcap_to_images(self):
        images = []
        labels = []

        for category in ['Benign', 'Malware']:
            cat_path = os.path.join(self.input_dir, category)
            if not os.path.exists(cat_path):
                print(f"[!] 警告: 找不到目录 {cat_path}")
                continue

            pcap_files = [f for f in os.listdir(cat_path) if f.endswith('.pcap')]

            for pcap_file in pcap_files:
                print(f"正在处理: {category}/{pcap_file}")
                file_path = os.path.join(cat_path, pcap_file)
                label_name = pcap_file.split('.')[0]

                try:
                    packets = rdpcap(file_path)
                    sessions = packets.sessions()

                    for session_name, session_pkts in sessions.items():
                        session_bytes = b''
                        for pkt in session_pkts:
                            cleaned_data = self.traffic_cleaning(pkt)
                            if cleaned_data:
                                session_bytes += cleaned_data

                        if len(session_bytes) == 0:
                            continue

                        truncated_data = self.traffic_truncation(session_bytes)
                        img_array = np.frombuffer(truncated_data, dtype=np.uint8).reshape(self.img_size, self.img_size)

                        images.append(img_array)
                        labels.append(label_name)

                except Exception as e:
                    print(f"解析 {pcap_file} 出错: {e}")

        return np.array(images), np.array(labels)

    def save_as_idx(self, images, labels):
        if len(images) == 0:
            print("[!] 错误：没有提取到任何图像数据，无法保存。")
            return

        print("正在按 8:2 的比例划分训练集和测试集...")
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.2, random_state=42, stratify=labels
        )

        train_path = self.output_idx_path.replace('.npz', '_train.npz')
        test_path = self.output_idx_path.replace('.npz', '_test.npz')

        np.savez_compressed(train_path, images=X_train, labels=y_train)
        np.savez_compressed(test_path, images=X_test, labels=y_test)

        print(f"训练集已保存至: {train_path} (样本数: {len(X_train)})")
        print(f"测试集已保存至: {test_path} (样本数: {len(X_test)})")


# ==========================================
# 2. CSV 特征转化为 28x28 图像的通用函数
# ==========================================
def tabular_to_image(X_raw, img_size=28, truncate_len=784):
    """将一维CSV统计特征归一化，并填充/截断为28x28图像结构，以适配LightGuard CNN"""
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_scaled = (X_scaled * 255).astype(np.uint8)

    num_samples = X_scaled.shape[0]
    num_features = X_scaled.shape[1]

    if num_features > truncate_len:
        X_padded = X_scaled[:, :truncate_len]
    else:
        padding = np.zeros((num_samples, truncate_len - num_features), dtype=np.uint8)
        X_padded = np.hstack((X_scaled, padding))

    images = X_padded.reshape(-1, img_size, img_size)
    return images


# ==========================================
# 3. CIC_IoT_2023 CSV 处理逻辑
# ==========================================
def process_cic_iot_csv(raw_dir, output_dir):
    print(f"\n[*] 开始处理 CIC_IoT_2023 数据集...")
    csv_files = glob.glob(os.path.join(raw_dir, '**/*.csv'), recursive=True)
    if not csv_files:
        print(f"[!] 错误: 在 {raw_dir} 下找不到 CSV 文件")
        return

    df_list = []
    for file in csv_files:
        print(f"    读取: {os.path.basename(file)}")
        try:
            df = pd.read_csv(file)
            df_list.append(df)
        except Exception as e:
            print(f"    [!] 读取 {file} 失败: {e}")

    if not df_list:
        return

    full_df = pd.concat(df_list, ignore_index=True)
    full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    full_df.dropna(inplace=True)

    label_col = 'label' if 'label' in full_df.columns else full_df.columns[-1]

    # 【修复核心】：过滤掉样本数少于2的极端稀有类别
    class_counts = full_df[label_col].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    dropped_classes = class_counts[class_counts < 2].index
    if len(dropped_classes) > 0:
        print(f"    [!] 为保证分层抽样，自动过滤样本数<2的稀有类别: {list(dropped_classes)}")
    full_df = full_df[full_df[label_col].isin(valid_classes)]

    labels = full_df[label_col].astype(str).values
    X_raw = full_df.drop(columns=[label_col]).select_dtypes(include=[np.number]).values

    images = tabular_to_image(X_raw)

    print(f"正在按 8:2 划分 CIC_IoT_2023 数据集 (共 {len(labels)} 条有效样本)...")
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)

    train_path = os.path.join(output_dir, 'cic_iot_2023_dataset_train.npz')
    test_path = os.path.join(output_dir, 'cic_iot_2023_dataset_test.npz')
    np.savez_compressed(train_path, images=X_train, labels=y_train)
    np.savez_compressed(test_path, images=X_test, labels=y_test)
    print(f"[+] CIC_IoT_2023 处理完成！")


# ==========================================
# 4. ToN-IoT CSV 处理逻辑
# ==========================================
def process_ton_iot_csv(raw_dir, output_dir):
    print(f"\n[*] 开始处理 ToN-IoT 数据集...")
    csv_path = os.path.join(raw_dir, 'train_test_network.csv')

    if not os.path.exists(csv_path):
        print(f"[!] 错误: 找不到文件 {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    label_col = 'type' if 'type' in df.columns else df.columns[-1]

    # 【修复核心】：同样过滤 ToN-IoT 中可能存在的极端稀有类别
    class_counts = df[label_col].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    dropped_classes = class_counts[class_counts < 2].index
    if len(dropped_classes) > 0:
        print(f"    [!] 为保证分层抽样，自动过滤样本数<2的稀有类别: {list(dropped_classes)}")
    df = df[df[label_col].isin(valid_classes)]

    labels = df[label_col].astype(str).values
    X_raw = df.drop(columns=[label_col]).select_dtypes(include=[np.number]).values

    images = tabular_to_image(X_raw)

    print(f"正在按 8:2 划分 ToN-IoT 数据集 (共 {len(labels)} 条有效样本)...")
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)

    train_path = os.path.join(output_dir, 'ton_iot_dataset_train.npz')
    test_path = os.path.join(output_dir, 'ton_iot_dataset_test.npz')
    np.savez_compressed(train_path, images=X_train, labels=y_train)
    np.savez_compressed(test_path, images=X_test, labels=y_test)
    print(f"[+] ToN-IoT 处理完成！")


# ==========================================
# 主入口模块
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightGuard Data Preprocessing")
    parser.add_argument('--dataset', type=str, default='USTC_TFC2016',
                        choices=['USTC_TFC2016', 'CIC_IoT_2023', 'ToN-IoT', 'all'],
                        help='选择要处理的数据集名称')

    args = parser.parse_args()

    BASE_RAW_DIR = "data/raw"
    BASE_PROCESSED_DIR = "data/processed"
    os.makedirs(BASE_PROCESSED_DIR, exist_ok=True)

    if args.dataset in ['USTC_TFC2016', 'all']:
        print("=== 开始执行 USTC-TFC2016 流量预处理 ===")
        raw_data_dir = os.path.join(BASE_RAW_DIR, "USTC_TFC2016")
        output_path = os.path.join(BASE_PROCESSED_DIR, "ustc_tfc2016_dataset.npz")
        preprocessor = LightGuardPreprocessor(raw_data_dir, output_path)
        imgs, lbls = preprocessor.pcap_to_images()
        if len(imgs) > 0:
            preprocessor.save_as_idx(imgs, lbls)

    if args.dataset in ['CIC_IoT_2023', 'all']:
        process_cic_iot_csv(os.path.join(BASE_RAW_DIR, 'CIC_IoT_2023'), BASE_PROCESSED_DIR)

    if args.dataset in ['ToN-IoT', 'all']:
        process_ton_iot_csv(os.path.join(BASE_RAW_DIR, 'ToN-IoT'), BASE_PROCESSED_DIR)

    print("\n=== 所有请求的数据集均已预处理完成 ===")