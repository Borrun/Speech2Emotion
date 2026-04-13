import os

def rename_wav_files(folder_path, prefix="unknown", digits=3):
    """
    重命名指定文件夹中的 .wav 文件

    :param folder_path: 目标文件夹路径
    :param prefix: 文件名前缀
    :param digits: 编号位数（默认3位 -> 001）
    """
    # 获取所有 wav 文件
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(".wav")]
    
    # 排序（避免乱序）
    files.sort()

    for idx, filename in enumerate(files, start=1):
        old_path = os.path.join(folder_path, filename)
        
        # 生成新文件名
        new_name = f"{prefix}_{str(idx).zfill(digits)}.wav"
        new_path = os.path.join(folder_path, new_name)

        # 重命名
        os.rename(old_path, new_path)
        print(f"{filename} -> {new_name}")

    print("✅ 重命名完成！")


if __name__ == "__main__":
    folder = "/Users/Zhuanz1/Speech2Emotion/annotater/wavs/疑问基调"
    prefix = "confused"

    rename_wav_files(folder, prefix)