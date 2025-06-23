import cv2
import numpy as np
import time
from scipy.spatial import distance as dist
from AcquireAndDisplay import init_camera, image_stream, frame_queue
import threading

#
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# Step 1: 初始化相机与启动采集线程  #initialize camera and start acquisition thread
cam, system, cam_list = init_camera()
producer_thread = threading.Thread(target=image_stream, args=(cam, frame_queue))
producer_thread.daemon = True
producer_thread.start()


#参考物体设置Reference Object Setting
KNOWN_WIDTH = 0.8  # mm


# 初始化像素与实际尺寸的比例
pixels_per_metric = None
fps_start = time.time()
fps_counter = 0
fps = 0.0

#图像处理主循环   #Main Image Processing Loop
while True:
    
    frame = frame_queue.get()  # 阻塞等待图像帧 # Blocking wait for image frame
    image = cv2.resize(frame, (800, 600))   #image size
    
    if len(image.shape) == 2:
    # 是灰度图 # It's a grayscale image
        gray = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
    # 是彩色图，转换灰度# It's a color image, convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 图像预处理 # Image Preprocessing
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 100)

    cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        continue

    for i, c in enumerate(cnts):
        if cv2.contourArea(c) < 500:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)

        if len(approx) == 5:
            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = order_points(box)

            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            if pixels_per_metric is None:
                pixels_per_metric = dB / KNOWN_WIDTH

            dimA = dA / pixels_per_metric
            dimB = dB / pixels_per_metric

            cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 0), 2)
            cv2.putText(image, f"{dimA:.1f}mm", (int(tltrX), int(tltrY - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(image, f"{dimB:.1f}mm", (int(trbrX), int(trbrY + 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        elif len(approx) > 10:
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            if radius > 10 and pixels_per_metric is not None:
                diameter = (2 * radius) / pixels_per_metric
                cv2.circle(image, (int(x), int(y)), int(radius), (0, 0, 255), 2)
                cv2.putText(image, f"{diameter:.3f}mm", (int(x - 20), int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
# HOLE
        if hierarchy[0][i][3] != -1:
            cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(image, "H", (cx - 10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.05, (0, 255, 255), 1)
    
# 计算并显示FPS #calculate and display FPS
    fps_counter += 1
    if fps_counter >= 10:
        fps = 10 / (time.time() - fps_start)
        fps_start = time.time()
        fps_counter = 0

    cv2.putText(image, f"FPS: {fps:.1f}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 显示处理后的图像
    cv2.imshow("Live Detection", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break





# clear up resources
cv2.destroyAllWindows()
cam.DeInit()
cam_list.Clear()
system.ReleaseInstance()
