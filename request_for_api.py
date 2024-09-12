import cv2
import numpy as np
import requests

def send_image_to_api(image_path, api_url):
    with open(image_path, 'rb') as img_file:
        files = {'image': img_file}
        response = requests.post(api_url, files=files)
    
    if response.status_code == 200:
        return response.content  # Trả về dữ liệu ảnh từ API
    else:
        print(f"Error: {response.status_code}")
        return None

def display_image(image_data):
    # Chuyển đổi dữ liệu ảnh thành mảng NumPy
    np_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    
    if image is not None:
        # Hiển thị ảnh bằng OpenCV
        cv2.imshow('Result Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: Không thể giải mã ảnh.")

if __name__ == "__main__":
    # Địa chỉ API và đường dẫn ảnh
    api_url = 'http://192.168.1.6:5000/predict'
    image_path = 'f4f7868e-60d3-4bce-914a-a796a221f141.jpg'
    
    # Gửi ảnh đến API và nhận về ảnh
    image_data = send_image_to_api(image_path, api_url)
    
    if image_data:
        # Hiển thị ảnh nhận về từ API
        display_image(image_data)
