# import subprocess
# from huggingface_hub import HfApi

# # Khởi tạo API (cần chạy huggingface-cli login trước đó)
# api = HfApi()

# folder_symlink = "/kaggle/personal-vad/data" # Thư mục chứa các symlink gom về

# # Lệnh tar với cờ:
# # -c: create
# # -z: gzip
# # -h: DEREFERENCE - cực kỳ quan trọng, biến symlink thành file/folder thật
# # -f -: xuất ra stdout (pipe)
# cmd = ["tar", "-czhf", "-", "-C", folder_symlink, "."]

# print("Bắt đầu nén dữ liệu thật từ symlink và stream lên Hugging Face...")

# process = subprocess.Popen(cmd, stdout=subprocess.PIPE)

# api.upload_file(
#     path_or_fileobj=process.stdout,
#     path_in_repo="backup_full_data.tar.gz",
#     repo_id="luvox-ai/Personal-VAD-Dataset",  # Thay username và repo của bạn
#     repo_type="dataset"
# )

# process.stdout.close()
# process.wait()

# print("Hoàn tất upload!")

from huggingface_hub import HfApi

api = HfApi()

folder_symlink = "/kaggle/personal-vad/data" # Đổi đường dẫn tới thư mục symlink của bạn

print("Bắt đầu upload toàn bộ dữ liệu qua symlink...")
api.upload_folder(
    folder_path=folder_symlink,
    repo_id="luvox-ai/Personal-VAD-Dataset",
    repo_type="dataset",
)

print("Hoàn tất upload!")