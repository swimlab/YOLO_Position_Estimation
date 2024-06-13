import pyrealsense2 as rs
import cv2, math
from decimal import Decimal
from decimal import *
import numpy as np
import os
#import pycuda.autoinit
import time
from PIL import Image

from ultralytics import YOLO

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results

def predict_and_detect(chosen_model, img, depth_frame, intr, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=2):
    results = predict(chosen_model, img, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            x_min = box.xyxy[0][0]
            y_min = box.xyxy[0][1]
            x_max = box.xyxy[0][2]
            y_max = box.xyxy[0][3]
            start_pooint = (int(x_min),int(y_min))
            end_point = (int(x_max),int(y_max))

            x = int(x_min +( x_max-x_min)/2)
            y = int(y_min + (y_max-y_min)/2)

            x = round(x)
            y = round(y)
            dist = depth_frame.get_distance(int(x), int(y))*1000 #convert to mm

            #calculate real world coordinates
            Xtemp = dist*(x -intr.ppx)/intr.fx
            Ytemp = dist*(y -intr.ppy)/intr.fy
            Ztemp = dist

            theta = 0
            Xtarget = Xtemp - 35 #35 is RGB camera module offset from the center of the realsense
            Ytarget = -(Ztemp*math.sin(theta) + Ytemp*math.cos(theta))
            Ztarget = Ztemp*math.cos(theta) + Ytemp*math.sin(theta)

            coordinat = (Decimal(str(Ztarget)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))

            if result.names[int(box.cls[0])] == 'sports ball':
                print("Class Name %s, Dist %d"%(result.names[int(box.cls[0])], dist))
                #print("Distance to Camera at (class : {0},  distance : {1:0.2f} mm)".format(result.names[int(box.cls[0])], dist), end="\r")
                img = cv2.putText(img.astype(np.uint8),"Dis: "+str(round(dist,2))+'mm',(x,y+5),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)
                
                img = cv2.rectangle(img.astype(np.uint8), (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                            (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
                img = cv2.circle(img.astype(np.uint8),(x,y),5,[0,0,255],5)
                img = cv2.putText(img.astype(np.uint8), f"{result.names[int(box.cls[0])]}",
                            (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), text_thickness)
    return img, results

def main() :
    theta = 0
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)
        
    # get camera intrinsics
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    
    
    #=========== Yolov4 TensorRt ağırlıkları yüklenmektedir =======================
    #model = YOLO("yolov9c.pt")
    model = YOLO("yolov8s.pt")
    #model = YOLO("yolov8m.pt")
 

    COLORS = [[0, 0, 255]]
    prev_frame_time=0
    new_frame_time=0
    while True:
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        #depth_image = np.asanyarray(depth_frame.get_data())
            
        image = Image.fromarray(color_image)
        img = np.asarray(image)
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        result_img, _ = predict_and_detect(model, img, depth_frame, intr, classes=[], conf=0.5)
        #classes,confidences,boxes = YOLOv4_video(img)
        

        cv2.imshow("Image", result_img)
        #cv2.imshow("Depth", depth_image_ocv)
            
        cv2.waitKey(1)


    cv2.destroyAllWindows()

    print("\nFINISH")

if __name__ == "__main__":
    main()