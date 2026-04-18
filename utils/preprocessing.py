import os
import glob
import argparse
import numpy as np
from collections import defaultdict
from scapy.all import IP, IPv6, TCP, UDP
from scapy.utils import PcapReader
from sklearn.model_selection import train_test_split


class NetVisionPreprocessor:
    def __init__(self, input_dir, output_idx_path, dataset_name, max_packets=0, img_size=28, truncate_len=784):
        """
        :param max_packets: 单个文件读取的数据包上限，设为 0 则表示全量读取
        """
        self.input_dir = input_dir
        self.output_idx_path = output_idx_path
        self.dataset_name = dataset_name
        self.max_packets = max_packets
        self.img_size = img_size
        self.truncate_len = truncate_len

        # 确保输出主目录存在
        os.makedirs(os.path.dirname(self.output_idx_path), exist_ok=True)

        # 【新增】为当前数据集创建一个专门存放临时缓存文件的目录
        self.temp_dir = os.path.join(os.path.dirname(self.output_idx_path), f"temp_{self.dataset_name}")
        os.makedirs(self.temp_dir, exist_ok=True)

    def traffic_cleaning(self, packet):
        # 同时支持 IPv4 和 IPv6 层的提取
        if packet.haslayer(IP):
            return bytes(packet[IP])
        elif packet.haslayer(IPv6):
            return bytes(packet[IPv6])
        return None

    def traffic_truncation(self, raw_bytes):
        if len(raw_bytes) >= self.truncate_len:
            return raw_bytes[:self.truncate_len]
        else:
            return raw_bytes + b'\x00' * (self.truncate_len - len(raw_bytes))

    def process_all_pcaps(self):
        """核心改动：只负责遍历并逐个文件提取，提取后立刻存为临时缓存文件"""
        search_pattern_pcap = os.path.join(self.input_dir, '**', '*.pcap')
        search_pattern_pcapng = os.path.join(self.input_dir, '**', '*.pcapng')

        pcap_files = glob.glob(search_pattern_pcap, recursive=True) + \
                     glob.glob(search_pattern_pcapng, recursive=True)

        if not pcap_files:
            print(f"[!] 警告: 在 {self.input_dir} 下找不到任何 pcap/pcapng 文件。")
            return False

        for file_path in pcap_files:
            base_name = os.path.basename(file_path)
            label_name = os.path.splitext(base_name)[0]

            # 【断点续传逻辑】检查该文件是否已经处理并保存过
            temp_file_path = os.path.join(self.temp_dir, f"{label_name}_{base_name}.npz")
            if os.path.exists(temp_file_path):
                print(f"\n[跳过] 缓存已存在，跳过处理: {base_name}")
                continue

            print(f"\n正在处理: {base_name} (分配标签: {label_name})")

            sessions = defaultdict(bytes)
            images = []
            labels = []

            try:
                with PcapReader(file_path) as pcap_reader:
                    i = 0
                    while True:
                        try:
                            pkt = pcap_reader.read_packet()
                        except EOFError:
                            break
                        except Exception:
                            continue

                        if self.max_packets > 0 and i >= self.max_packets:
                            print(f"    [!] 达到预设采样上限 ({self.max_packets}包)，停止读取本文件。")
                            break

                        if i > 0 and i % 50000 == 0:
                            print(f"    ... 已读取 {i} 个数据包 ...")

                        cleaned_data = self.traffic_cleaning(pkt)
                        if not cleaned_data:
                            i += 1
                            continue

                        has_ip4 = pkt.haslayer(IP)
                        has_ip6 = pkt.haslayer(IPv6)

                        if has_ip4 or has_ip6:
                            net_layer = pkt[IP] if has_ip4 else pkt[IPv6]
                            src = net_layer.src
                            dst = net_layer.dst
                            proto = net_layer.proto if has_ip4 else net_layer.nh
                            sport, dport = 0, 0

                            if pkt.haslayer(TCP):
                                sport = pkt[TCP].sport
                                dport = pkt[TCP].dport
                            elif pkt.haslayer(UDP):
                                sport = pkt[UDP].sport
                                dport = pkt[UDP].dport

                            end1 = f"{src}:{sport}"
                            end2 = f"{dst}:{dport}"
                            session_key = f"{proto}-" + "-".join(sorted([end1, end2]))

                            if len(sessions[session_key]) < self.truncate_len:
                                sessions[session_key] += cleaned_data

                        i += 1

                print(f"    [+] {base_name} 读取完毕！共提取 {len(sessions)} 个会话，正在保存临时缓存...")
                for sess_key, session_bytes in sessions.items():
                    if len(session_bytes) == 0:
                        continue
                    truncated_data = self.traffic_truncation(session_bytes)
                    img_array = np.frombuffer(truncated_data, dtype=np.uint8).reshape(self.img_size, self.img_size)

                    images.append(img_array)
                    labels.append(label_name)

                # 【落盘逻辑】如果这个文件提取到了流量，将其单独保存
                if len(images) > 0:
                    np.savez_compressed(temp_file_path, images=np.array(images), labels=np.array(labels))
                    print(f"    [成功] 数据已保存至 -> {temp_file_path}")
                else:
                    print(f"    [-] 警告: {base_name} 未提取到有效会话流量。")

            except Exception as e:
                print(f"[!] 解析 {base_name} 出错: {e}")

        return True

    def merge_and_save(self):
        """核心改动：从 temp 目录读取所有小缓存文件，合并并划分为最终的 train/test"""
        print(f"\n[*] 正在合并 {self.dataset_name} 的所有临时缓存文件...")
        temp_files = glob.glob(os.path.join(self.temp_dir, '*.npz'))

        if not temp_files:
            print(f"[!] 错误：在 {self.temp_dir} 找不到任何缓存文件，无法合并。")
            return

        all_images = []
        all_labels = []

        # 遍历读取临时目录下的所有 npz 文件
        for t_file in temp_files:
            try:
                data = np.load(t_file, allow_pickle=True)
                all_images.append(data['images'])
                all_labels.append(data['labels'])
            except Exception as e:
                print(f"    [!] 加载缓存文件 {t_file} 失败: {e}")

        if not all_images:
            print("[!] 未收集到任何图像数据。")
            return

        # 拼接所有文件里的特征矩阵
        print("[*] 正在拼接所有特征矩阵...")
        images = np.concatenate(all_images, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        print(f"[*] 合并完毕，共收集到 {len(images)} 条数据。正在按 8:2 划分训练集和测试集...")

        # 过滤掉样本数少于 2 的类别，防止 split 报错
        unique_labels, counts = np.unique(labels, return_counts=True)
        valid_labels = unique_labels[counts >= 2]

        valid_indices = np.isin(labels, valid_labels)
        images = images[valid_indices]
        labels = labels[valid_indices]

        if len(images) < 2:
            print("[!] 过滤后有效样本不足，无法划分数据集。")
            return

        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.2, random_state=42, stratify=labels
        )

        train_path = self.output_idx_path.replace('.npz', '_train.npz')
        test_path = self.output_idx_path.replace('.npz', '_test.npz')

        np.savez_compressed(train_path, images=X_train, labels=y_train)
        np.savez_compressed(test_path, images=X_test, labels=y_test)

        print(f"[+] 训练集已保存至: {train_path} (样本数: {len(X_train)})")
        print(f"[+] 测试集已保存至: {test_path} (样本数: {len(X_test)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NetVision PCAP Preprocessing (OOM Safe & Resumable)")
    parser.add_argument('--dataset', type=str, default='USTC_TFC2016',
                        choices=['USTC_TFC2016', 'CIC_IoT_2023', 'ToN-IoT', 'all'],
                        help='选择要处理的数据集名称')
    parser.add_argument('--max_packets', type=int, default=300000,
                        help='单个 PCAP 文件读取的数据包上限 (设为 0 则全量读取)')

    args = parser.parse_args()

    BASE_RAW_DIR = "data/raw"
    BASE_PROCESSED_DIR = "data/processed"
    os.makedirs(BASE_PROCESSED_DIR, exist_ok=True)

    datasets_to_process = ['USTC_TFC2016', 'CIC_IoT_2023', 'ToN-IoT'] if args.dataset == 'all' else [args.dataset]

    for ds_name in datasets_to_process:
        print(f"\n{'=' * 40}")
        print(f"=== 开始执行 {ds_name} 流量预处理 ===")
        print(f"{'=' * 40}")
        raw_data_dir = os.path.join(BASE_RAW_DIR, ds_name)

        if not os.path.exists(raw_data_dir):
            print(f"[!] 错误: 找不到数据集目录 {raw_data_dir}")
            continue

        safe_ds_name = ds_name.lower().replace('-', '_')
        output_path = os.path.join(BASE_PROCESSED_DIR, f"{safe_ds_name}_dataset.npz")

        preprocessor = NetVisionPreprocessor(raw_data_dir, output_path, dataset_name=ds_name,
                                             max_packets=args.max_packets)

        # 1. 逐个提取并缓存
        has_data = preprocessor.process_all_pcaps()

        # 2. 提取完毕后，统一合并缓存文件并划分训练集
        if has_data:
            preprocessor.merge_and_save()

    print("\n=== 所有请求的数据集均已预处理完成 ===")