import os

label=input()
# Path to DOTA annotations outside current folder
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), '..','..', 'DOTA'))
LABEL_DIR = os.path.join(BASE_DIR, label)  # or 'label_test'

def count_files_with_gsd(label_dir):
    count = 0
    total = 0
    for file in os.listdir(label_dir):
        if not file.endswith('.txt'):
            continue
        total += 1
        path = os.path.join(label_dir, file)
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines[:10]:  # Only check the first few lines
                if 'gsd' in line.lower():
                    count += 1
                    break
    return count, total

if __name__ == "__main__":
    count, total = count_files_with_gsd(LABEL_DIR)
    print(f"Files with 'gsd' field: {count}/{total} ({(count/total)*100:.2f}%)")
