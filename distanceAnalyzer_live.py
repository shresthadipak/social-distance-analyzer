import cv2
from objectDetectionModule import objectDetector

detector  = objectDetector()

cap = cv2.VideoCapture('videos/raw_video.mp4')
width = int(cap.get(3))
height = int(cap.get(4))
fps = int(cap.get(5))/2
out = cv2.VideoWriter('output/output2.avi', cv2.VideoWriter_fourcc(*'MPEG'), fps, (width, height))

while True:
    ret, frame = cap.read()

    if not ret:
        break

    img = detector.object_detect(frame)   

    out.write(img)

    # cv2.imshow("Distance Analyzer", img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()    