
import requests as r
import json
from pprint import pprint

import base64
from io import BytesIO
from PIL import Image

def send_request(file_list = ['./images/zidane.jpg'], 
                    model_name = 'yolov5s',
                    img_size = 640,
                    download_image = False):

    files = [('file_list', open(file,"rb")) for file in file_list]

    other_form_data = {'model_name': model_name,
                    'img_size': img_size,
                    'download_image': download_image}

    res = r.post("http://localhost:8000/detect/", 
                    data = other_form_data, 
                    files = files)

    if download_image:
        json_data = res.json()

        for img_data in json_data:
            for bbox_data in img_data:
                if 'image_base64' in bbox_data.keys():
                    img = Image.open(BytesIO(base64.b64decode(str(bbox_data['image_base64']))))
                    img.show()
                else:
                    pprint(bbox_data)

    else:
        pprint(json.loads(res.text))


if __name__ == '__main__':
    send_request(file_list=['./images/bus.jpg'], download_image = True)
