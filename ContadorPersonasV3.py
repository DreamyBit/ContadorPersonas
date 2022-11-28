
def main(args):

    import cv2
    import numpy as np
    import time

    net = cv2.dnn.readNet("./yolov3.weights", "./yolov3.cfg")

    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]


    # ENTRADA DE VIDEO con buffer de 1 frame, toma la camara principal del dispositivo por defecto
    if (args.camera is not None and type(args.camera) is int and args.camera >= 0 ):
        nCamera = args.camera
    else:
        nCamera = 0
    cap = cv2.VideoCapture(nCamera)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Variables
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    font = cv2.FONT_HERSHEY_PLAIN
    class_ids = []
    confidences = []
    boxes = []

    while(True):
        # Lectura de imagen
        ret, frame = cap.read()

        # Reset variables
        boxes.clear()
        confidences.clear()
        class_ids.clear()

        if(ret):

            # reescalado de imagen
            frame = cv2.resize(frame, (640, 480))
            height, width, channel = frame.shape

            # deteccion de objetos usando blob; (608,608) puede ser usado en lugar de (320,320)
            blob = cv2.dnn.blobFromImage(
                frame, 1/255, (320, 320), (0, 0, 0), True, crop=False)

            # deteccion de objetos usando la red neuronal
            net.setInput(blob)
            outs = net.forward(output_layers)

            # procesamiento de objetos
            for out in outs:
                for detection in out:

                    # definiciones
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    # si la red neuronal posee una confianza mayor a 50% de que el objeto es una persona se procesa
                    if confidence > 0.5:
                        # Objecto encontrado
                        center_x = int(detection[0]*width)
                        center_y = int(detection[1]*height)
                        w = int(detection[2]*width)
                        h = int(detection[3]*height)
                        # Cuadrar coordenadas
                        x = int(center_x-w/2)
                        y = int(center_y-h/2)
                        # Guardar informacion
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

                        # Non-max Suppression, evita que una persona figure como muchas
                        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)

            count = 0

            # Dibujar rectangulos en la imagen
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = COLORS[i]
                    if int(class_ids[i] == 0):
                        count += 1
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, label+" " +
                                    str(round(confidences[i], 3)), (x, y-5), font, 1, color, 1)

            # Dibujar contador
            cv2.putText(frame, str(count), (100, 200),
                        cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 10)
            cv2.imshow("Detected_Images", frame)

        # Esc para salir
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Contador de personas')
    parser.add_argument('-c','--camera', type=int, required=False, help='Numero de la camara a tomar como entrada')
    args = parser.parse_args()
    
    main(args)