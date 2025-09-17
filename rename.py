import os


if __name__ == '__main__':
    base_folder = 'data/client_v4_video/5FPS_all_csv'
    list_file = sorted(os.listdir(base_folder))
    list_path = []
    for root, folders, files in os.walk(base_folder):
        for file in files:
            list_path.append(os.path.join(root, file))
    for old_path in list_path:
        new_path = old_path.replace('.MP4.csv','.csv')
        os.rename(old_path, new_path)
