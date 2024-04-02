import pathlib
import numpy as np
import cv2

def get_data():
    data_path = pathlib.Path(__file__).parent.parent / 'datasets/abu_otameshi/train'

    for label_path in (data_path / 'labels').iterdir():
        with open(label_path, 'r') as f:
            text = f.read()

        if text == '':
            continue

        color_data = [[float(x) for x in line.split(' ')] for line in text.split('\n')]

        colors = []

        for i in range(3):
            colors.append([c[1:] for c in color_data if c[0] == i])
            
        image_path = data_path / 'images' / (label_path.stem + '.jpg')

        yield cv2.imread(image_path.as_posix()), colors


def get_hsv(img: np.ndarray, colors):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for color_id, xyxys in enumerate(colors):
        for x, y, w, h in xyxys:
            x = int(x * img.shape[1])
            y = int(y * img.shape[0])
            w = int(w * img.shape[1])
            h = int(h * img.shape[0])
            x1 = int(x - w/2)
            y1 = int(y - h/2)
            x2 = int(x + w/2)
            y2 = int(y + h/2)

            c = 0.5
            width = x2 - x1
            height = y2 - y1

            x1_limited = x1 + int(width * (1 - c)/2)
            y1_limited = y1 + int(height * (1 - c)/2)
            x2_limited = x2 - int(width * (1 - c)/2)
            y2_limited = y2 - int(height * (1 - c)/2)

            hsv_limited = hsv[y1_limited:y2_limited, x1_limited:x2_limited]

            if hsv_limited.size == 0:
                continue

            h = np.median(hsv_limited[:,:,0])
            s = np.median(hsv_limited[:,:,1])
            v = np.median(hsv_limited[:,:,2])

            cv2.imwrite(f'color/images/converted_{color_id}.jpg', cv2.cvtColor(hsv_limited, cv2.COLOR_HSV2BGR))

            yield color_id, h, s, v

def get_hsv_all():
    for img, colors in get_data():
        yield from get_hsv(img, colors)

import matplotlib.pyplot as plt

def scatter_plot():
    for color_id, h, s, v in get_hsv_all():
        c=['blue', 'purple', 'red']
        print(h, s, c[color_id])
        plt.scatter(h, s, color=c[color_id])

    plt.xlabel('Hue')
    plt.ylabel('Saturation')

    plt.show()
        
    

        

if __name__ == '__main__':
    scatter_plot()
        

    
            