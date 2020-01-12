import os
import io
import base64
import json
import requests
import argparse
from PIL import Image

def request(path, host, port):
    instances = { 'instances': [] }

    name, ext = os.path.splitext(path)
    if ext not in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
        im = Image.open(path).convert('RGB')
        path = '/tmp/{}.jpg'.format(os.path.basename(name))
        im.save(path)

    with io.open(path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    instances['instances'].append({
        'image_bytes': {'b64': encoded_image},
        'key': os.path.basename(path),
    })

    url = 'http://{}:{}/predict'.format(host, port)

    res = requests.post(url, data=json.dumps(instances), headers={'Content-Type': 'application/json'}).json()

    for entry in res.values():
        print(entry['label'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='get predicted result from automl edge container on local')
    parser.add_argument('image_file_path', help='image file path')
    parser.add_argument('--host', default='localhost', help='host name')
    parser.add_argument('--port', default=5000, help='port number')
    args = parser.parse_args()

    request(args.image_file_path, args.host, args.port)
