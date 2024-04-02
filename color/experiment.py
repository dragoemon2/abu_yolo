import cv2

PATH = '/home/ryunosuke/ai_ws/abu_yolo/datasets/abu_otameshi/train/images/000015_jpg.rf.ee94ea64babbbb222b35bb3729bbb12e.jpg'

rgb = cv2.imread(PATH)
hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)

#sをmaxにする
#hsv[:,:,1] = 179

#vを適当にする
hsv[:,:,2] = 150

rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imwrite('color/converted.jpg', rgb)
