import os
import shutil

# ==== Cấu hình đường dẫn ====
src1 = "client_v4_gcn_predictions"
src2 = "client_v1_gcn_predictions"
dst  = "merged_predictions"

# Danh sách file cần copy riêng
names_to_copy = [
    "GH013464",
    "GH013467",
    "Look_around_Back_Other_3",
    "Look_around_Back_Walking_2",
    "Look_around_Front_Other_3",
    "Look_around_Front_Walking_3",
    "Look_around_Side_Other_3",
    "Look_around_Side_Walking_4",
    "Sit_still_Chair_Back_Walking_3",
    "Sit_still_Chair_Side_Other_3",
    "Sit_still_No_chair_Back_Other_3",
    "Sit_still_No_chair_Front_Walking_3",
    "Sit_still_No_chair_Side_Other_3",
    "Stand_still_Back_Walking_3",
    "Stand_still_Side_Other_3",
    "Stand_still_Side_Walking_3",
    "Walking_Back_Walking_4",
    "Walking_Back_Walking_5",
]

# ==== Tạo thư mục đích ====
os.makedirs(dst, exist_ok=True)

# ==== 1. Copy các file theo danh sách từ src1 ====
for name in names_to_copy:
    src_path = os.path.join(src1, f"{name}_predictions.csv")
    if os.path.exists(src_path):
        shutil.copy2(src_path, os.path.join(dst, f"{name}_predictions.csv"))
        print(f"✅ Copied from src1: {src_path}")
    else:
        print(f"⚠️ Không tìm thấy trong src1: {src_path}")

# ==== 2. Copy các file theo danh sách từ src2 ====
for name in names_to_copy:
    src_path = os.path.join(src2, f"{name}_predictions.csv")
    if os.path.exists(src_path):
        shutil.copy2(src_path, os.path.join(dst, f"{name}_predictions.csv"))
        print(f"✅ Copied from src2: {src_path}")
    else:
        print(f"⚠️ Không tìm thấy trong src2: {src_path}")

# ==== 3. Copy nguyên toàn bộ folder src2 vào dst (giữ nguyên cấu trúc) ====
for root, dirs, files in os.walk(src2):
    # Tính đường dẫn tương đối so với src2
    rel_path = os.path.relpath(root, src2)
    # Tạo thư mục tương ứng trong dst
    target_dir = os.path.join(dst, rel_path)
    os.makedirs(target_dir, exist_ok=True)

    for file in files:
        src_file = os.path.join(root, file)
        dst_file = os.path.join(target_dir, file)
        if not os.path.exists(dst_file):  # tránh đè nếu bạn muốn
            shutil.copy2(src_file, dst_file)
            print(f"📂 Copied (full src2): {src_file} -> {dst_file}")

print("🎉 Hoàn tất copy tất cả!")
