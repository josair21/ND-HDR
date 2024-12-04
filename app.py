from flask import Flask, request, jsonify
from keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import cv2
import os
import base64
import io
from ultralytics import YOLO
import tensorflow as tf

app = Flask(__name__)

# Load a pretrained YOLO model_localize
path_model_localize = r"models\detect\best.pt"
model_localize = YOLO(path_model_localize)

# Model Classifier
path_classifier = r"models\classifier\model_PavicNet_BRACOL_V7.hdf5"
model_all_0 = load_model(path_classifier)


def diagnose_image(image_path):
    # image_path = r"C:\Users\Lucas\Documents\Projeto_Estagio\Codigos\image.jpeg"
        
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img2 = None
    # Perform object detection on an image
    results = model_localize(image_path)

    boxes = results[0].boxes.xyxy

    list_count = [0, 0, 0, 0, 0]
    classes = ['Rust', 'Miner', 'Phoma', 'Cercospora', 'Não Identificado']
    num = len(boxes)
    if num > 0:
        for i in range(num):
            box = boxes[i]

            x, y, w, h = map(int, box[:4])  # Converte coordenadas para inteiros

            cropped = image[y:h, x:w]  # Recorta a imagem

            path_save_image = f'./models/save_images/{i}.jpg'
            # Salva a imagem cortada no diretorio

            cv2.imwrite(path_save_image, cropped)
            # Ler a imagem cortada

            img_cortada = tf.keras.utils.load_img(path_save_image , target_size=(224, 224), color_mode="rgb")
            img_cortada = tf.keras.utils.img_to_array(img_cortada)
            img_cortada = img_cortada/255

            # Cria os retangulos verdes na imagem
            img2 = cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)  # (0, 255, 0) representa a cor verde

            # Realiza a classificação do modelo
            pred = model_all_0.predict(tf.expand_dims(tf.convert_to_tensor(img_cortada), axis=0), verbose=None)
            
            # Determina a classe da doença
            pred2 = pred[0]
            pred2 = np.array(pred2 > 0.8)  # limiar
            i = i+1

            colorRGB = (0, 0, 0)
            # Nomeia a doença
            if pred2[0]:
                colorRGB = (255, 0, 0)
                cv2.putText(img2, f"Rust", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2.5, colorRGB, 5)
                list_count[0] += 1
            elif pred2[1]:
                colorRGB = (255, 255, 255)
                cv2.putText(img2, f"Miner", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2.5, colorRGB, 5)
                list_count[1] += 1
            elif pred2[2]:
                colorRGB = (255, 0, 255)
                cv2.putText(img2, f"Phoma", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2.5, colorRGB, 5)
                list_count[2] += 1
            elif pred2[3]:  # and confidence > 0.95:
                colorRGB = (255, 255, 0)
                cv2.putText(img2, f"Cercospora", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2.5, colorRGB, 5)
                list_count[3] += 1
            else:
                colorRGB = (0, 0, 0)
                list_count[4] += 1
            
            img2 = cv2.rectangle(image, (x, y), (w, h), colorRGB, 2)  # (0, 255, 0) representa a cor verde


        os.remove(path_save_image)

        return img2, classes, list(map(str,list_count))
    
    os.remove(path_save_image)
    return image, classes, list_count


@app.route('/process', methods=['POST'])
def process_image():
    # Verificar se o arquivo foi enviado
    if 'image' not in request.files:
        return jsonify({'error': 'Nenhuma imagem foi enviada'}), 400

    file = request.files['image']
    
    file_path = os.path.join('.\models\save_images', file.filename)
    file.save(file_path)
    try:
        image, deseases_list, count_list = diagnose_image(file_path)
        image.astype(np.uint8)
        image = Image.fromarray(image)

    except Exception as e:
        return jsonify({'error': 'Erro ao processar a imagem: ' + str(e)}), 400

    # Salvar a imagem em um buffer
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="JPEG")  # Salvando como JPEG para uniformidade
    img_buffer.seek(0)

    # Converter a imagem para base64 para enviar como resposta
    encoded_image = base64.b64encode(img_buffer.read()).decode('utf-8')

    # Retornar os dados no formato solicitado
    response = {
        'image': encoded_image,  # Imagem como base64
        'strings': deseases_list,
        'ints': count_list
    }

    return jsonify(response)




if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
    