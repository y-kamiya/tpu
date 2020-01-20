from absl import app
from absl import flags
import tensorflow as tf
import os
import sys

tf.enable_eager_execution()

FLAGS = flags.FLAGS
flags.DEFINE_string('file', default=None, help='')

image_feature_description = {
  'image/height': tf.io.FixedLenFeature([], tf.int64),
  'image/width': tf.io.FixedLenFeature([], tf.int64),
  'image/colorspace': tf.io.FixedLenFeature([], tf.string),
  'image/channels': tf.io.FixedLenFeature([], tf.int64),
  'image/class/label': tf.io.FixedLenFeature([], tf.int64),
  'image/class/synset': tf.io.FixedLenFeature([], tf.string),
  'image/format': tf.io.FixedLenFeature([], tf.string),
  'image/filename': tf.io.FixedLenFeature([], tf.string),
  'image/encoded': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
  # 入力の tf.Example のプロトコルバッファを上記のディクショナリを使って解釈
  return tf.io.parse_single_example(example_proto, image_feature_description)

def read_content(file):
    raw_image_dataset = tf.data.TFRecordDataset(file)
    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

    for image_features in parsed_image_dataset:
      w = image_features['image/width'] 
      h = image_features['image/height'] 
      im = tf.io.decode_jpeg(image_features['image/encoded'])
      shape = im.shape
      print('{}, {}, {}, {}, {}, {}'.format(image_features['image/filename'], w, shape[1], h, shape[0], w * h))

def main(unused_argv):
    if os.path.isdir(FLAGS.file):
        list = os.listdir(FLAGS.file)
        for file in sorted(list):
            read_content(os.path.join(FLAGS.file, file))
    else:    
        read_content(FLAGS.file)

if __name__ == '__main__':
    flags.mark_flags_as_required(['file'])
    app.run(main)
