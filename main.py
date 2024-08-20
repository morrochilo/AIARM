import os
import cv2
import mss
import numpy as np
import pygetwindow as gw
import keyboard
import pytesseract

# Establecer el prefijo TESSDATA_PREFIX
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

# Identificar la ventana de interés
window = gw.getWindowsWithTitle("Autodesk Inventor Professional 2024")[0]
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Obtener las coordenadas de la ventana
left, top, width, height = window.left, window.top, window.width, window.height

# Crear un objeto de captura de pantalla
sct = mss.mss()

# Nombre de la ventana
window_name = "Monitor"

# Configurar la ventana para que sea a pantalla completa
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    # Capturar la pantalla de la ventana específica
    monitor = {"top": top, "left": left, "width": width, "height": height}
    img = np.array(sct.grab(monitor))

    # Convertir de BGR a RGB (si es necesario)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Procesamiento de la imagen (por ejemplo, detección de bordes)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Obtener datos detallados del OCR
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

    # Recorrer los datos y dibujar rectángulos alrededor de cada palabra
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        if int(data['conf'][i]) > 60:  # Confianza mínima para dibujar la caja
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #edges = cv2.Canny(gray, 50, 150)

    # Señalizar irregularidades (ejemplo simple: dibujar bordes)
    #img[edges > 0] = [0, 255, 0]

    # Mostrar la imagen procesada usando cv2.imshow
    cv2.imshow("Monitor", img)

    # Salir del bucle si se presiona la tecla 'q'
    if keyboard.is_pressed('q'):
        print("Se presionó 'q', saliendo del bucle...")
        break

    # Añadir una breve espera para evitar un uso excesivo de la CPU
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

# Liberar los recursos
cv2.destroyAllWindows()