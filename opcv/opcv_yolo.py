#################################################
#       MJ's Object detect based on YOLOv3      #
#                   20240321                    #
#################################################

import cv2
import numpy as np

VideoSignal = cv2.VideoCapture(0)   # 보통 카메라번호는 0 아니면 1입니다.
YOLO_model = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")  # 빠른데 성능이 낮아요
# YOLO_model = cv2.dnn.readNet("yolov3-spp.weights", "yolov3-spp.cfg")      # 느린데 성능이 좋아요

object_name_list = []   # 검출된 물체의 이름을 담을 리스트입니다.
with open("coco.names", "r") as f:  # 이름파일을 리스트에 담아줍니다.
    object_name_list = [line.strip() for line in f.readlines()]
layer_names = YOLO_model.getLayerNames()    # 무수히 많은 딥러닝 레이어층이 있어요
output_layers = [layer_names[i - 1] for i in YOLO_model.getUnconnectedOutLayers()]  # 얻은 영상을 하나씩 통과시킬겁니다.

while True:
    # 카메라로 부터 정보를 받아옵니다.
    ret, frame = VideoSignal.read()
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # YOLO 모델에 넣어서 판별
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    YOLO_model.setInput(blob)
    outs = YOLO_model.forward(output_layers)

    class_ids = []  # 사물이름
    confidences = []    # 신뢰성
    detected_boxes_list = []    # 인식된 상자

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:    # 50% 미만은 인식 못했다고 할겁니다.
                # 인식된 사물 크기
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                dw = int(detection[2] * w)
                dh = int(detection[3] * h)
                # 그에 따른 그려줄 상자
                x = int(center_x - dw / 2)
                y = int(center_y - dh / 2)
                detected_boxes_list.append([x, y, dw, dh])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 비최대억제 알고리즘 적용, 겹쳐지는 상자는 제거할수 있도록
    indexes = cv2.dnn.NMSBoxes(detected_boxes_list, confidences, 0.5, 0.4)

    # 이제 진짜 그려줍시다.
    for i in range(len(detected_boxes_list)):
        if i in indexes:
            x, y, w, h = detected_boxes_list[i] # 상자
            label = str(object_name_list[class_ids[i]]+" "+str(round(confidences[i],2)))    # 물체 이름과 점수

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)    # 상자 그려
            cv2.putText(frame, label, (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1) # 이름 그려
            print(f"object: {label}")   # 어떤 물체가 인식되었는지 이름과 예측점수 출력

    # 이미지 창
    cv2.imshow("MJ's Object detect based on YOLOv3", frame)

    if cv2.waitKey(1) == 27:    # esc를 입력하면 방금 사용한 YOLO 모델에 어떤 레이어들이 들어가있는지 출력하고 종료
        print(YOLO_model.getLayerNames())
        break