import os
import cv2
from tqdm import tqdm

# 資料夾路徑
input_folder = "infer_1"
output_folder = "infer_crop"

# 確保輸出資料夾存在
os.makedirs(output_folder, exist_ok=True)

# 取得資料夾中所有檔案
image_file_names = os.listdir(input_folder)

# 進度條初始化
progress_bar = tqdm(total=len(image_file_names), desc='Processing Images', unit='image')

# 逐一處理每張圖片
for image_name in image_file_names:
    image_path = os.path.join(input_folder, image_name)
    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)[1]

    # 畫 bounding box
    x, y, w, h = cv2.boundingRect(gray)
    cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cX, cY = x + w // 2, y + h // 2

    crop_length = 220
    h, w, _= image.shape
    left_x = max(0, cX - int(crop_length/2)) 
    right_x = min(w, cX + int(crop_length/2))
    top_y = max(0, cY - int(crop_length/2))
    bot_y = min(h, cY + int(crop_length/2))

    result_image = image[top_y:bot_y, left_x:right_x]
    result_image = cv2.resize(result_image, (300, 300), interpolation=cv2.INTER_AREA)

    # 輸出圖片的檔案名稱
    output_path = os.path.join(output_folder, image_name)
    cv2.imwrite(output_path, result_image)

    # 更新進度條
    progress_bar.update(1)

# 關閉進度條
progress_bar.close()
print("Image processing completed.")
