import shutil
import kagglehub
import os
from utils import CLASS

# Environment
download_dataset = False

if download_dataset:
  os.environ["KAGGLEHUB_CACHE"] = os.path.join(os.getcwd(), ".cache")
  path = kagglehub.dataset_download("sshikamaru/fruit-recognition")
  print("Path to dataset files:", path)
else:
  print("Skip downloading dataset")

if os.path.exists("dataset"):
  shutil.rmtree("dataset")
os.makedirs("dataset")

DATASET_FOLDER_PATH = ".cache/datasets/sshikamaru/fruit-recognition/versions/2/train/train"
folders = os.listdir(DATASET_FOLDER_PATH)

for label in CLASS:
  os.makedirs(os.path.join("dataset", label))
  for folder in folders:
    if not folder.lower().startswith(label): continue
    files = os.listdir(os.path.join(DATASET_FOLDER_PATH, folder))
    for file in files:
      os.rename(
        os.path.join(DATASET_FOLDER_PATH, folder, file),
        os.path.join("dataset", label, file)
      )
      print(f"[{label}] {file} 옮기기 완료")

open("dataset/.gitkeep", "a").close()

print("데이터셋 준비를 모두 마쳤습니다.")
