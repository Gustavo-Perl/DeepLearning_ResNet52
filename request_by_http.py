import requests
import cv2

url = 'http://127.0.0.1:5000/api'
nome_imagem = input('Digite o nome da imagem (APENAS JPG): ')
img = cv2.imread(nome_imagem)
_, img_encoded = cv2.imencode('.jpg', img)
r = requests.post(url, data = img_encoded.tobytes(), headers = {'content-type': 'image/jpg'})
print(r.json())