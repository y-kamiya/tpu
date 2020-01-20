import sys
import os
import json
import base64
import numpy as np
import tensorflow.compat.v1 as tf

from absl import app
from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_string('host', default='127.0.0.1', help='')
flags.DEFINE_integer('port', default=5000, help='')

from flask import Flask, request, jsonify, make_response
flask = Flask(__name__)

sys.path.append('./models/official/efficientnet')
sys.path.append('./models/common')

import eval_ckpt_main as eval_ckpt
import utils

sess = tf.Session()
probs = None
iterator = None

script_dir = os.path.dirname(__file__)
model_dir = os.path.join(script_dir, 'model')
labels_map_path = os.path.join(model_dir, 'labels_map.txt')
classes = json.loads(tf.gfile.Open(labels_map_path).read())
eval_driver = eval_ckpt.get_eval_driver('efficientnet-b0')
image_files = tf.placeholder(dtype=tf.string, shape=[None])

@flask.route("/predict", methods=['POST'])
def postPredict():
    params = request.get_json()
    entry = params['instances'][0]

    path = '/tmp/{}.jpg'.format(os.getpid())
    with open(path, 'wb') as f:
        f.write(base64.decodebytes(entry['image_bytes']['b64'].encode('utf-8')))

    sess.run(iterator.initializer, feed_dict={image_files: [path]})
    out_probs = sess.run(probs, feed_dict={image_files: [path]})
    idx = np.argsort(out_probs)[::-1]
    prediction_idx = idx[:5]
    prediction_prob = [out_probs[pid] for pid in idx[:5]]

    response = {}
    for j, idx in enumerate(prediction_idx):
        response[j] = {
            'label': classes[str(idx)],
            'probability': prediction_prob[j].astype(np.float64),
        }

    return make_response(jsonify(response))

def main(unused_argv):
    global probs, sess, iterator

    images, iterator = eval_driver.build_dataset_eval(image_files)
    probs = eval_driver.build_model(images, is_training=False)
    if isinstance(probs, tuple):
      probs = probs[0]
    eval_driver.restore_model(sess, model_dir, True, False)

    flask.run(host=FLAGS.host, port=FLAGS.port)


if __name__ == '__main__':
    app.run(main)
