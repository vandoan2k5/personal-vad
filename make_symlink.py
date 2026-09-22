import os
from pathlib import Path

# Cập nhật đường dẫn thực tế của bạn
SOURCE_FOLDERS = [
    Path(r"/kaggle/input/datasets/doannv1809/hust-tse-1-folder"),  # Hoặc "/path/to/folder1" trên Linux/macOS
    Path(r"/kaggle/input/datasets/doannv1809/hust-tse-2-folder"),
    Path(r"/kaggle/input/datasets/doannv1809/hust-tse-3-folder"),
    Path(r"/kaggle/input/datasets/doannv1809/vctk-tse-1-folder")
]
TARGET_FOLDER = Path(r"/kaggle/personal-vad/data/audio_with_id")

TARGET_FOLDER.mkdir(parents=True, exist_ok=True)
count = 0

for src in SOURCE_FOLDERS:
    for id_dir in src.iterdir():
        if id_dir.is_dir():
            # Tạo thư mục ID thật ở đích
            dest_id_dir = TARGET_FOLDER / id_dir.name
            dest_id_dir.mkdir(exist_ok=True)

            # Tạo symlink cho từng file âm thanh bên trong
            for file in id_dir.iterdir():
                if file.is_file():
                    dest_file = dest_id_dir / file.name
                    if not dest_file.exists():
                        dest_file.symlink_to(file.resolve())
                        count += 1

print(f"Hoàn tất! Đã tạo thành công {count} symlink.")