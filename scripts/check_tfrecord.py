import os
import tensorflow as tf
# from PIL import Image
# import io
#
# dir = 'dataset284_imagenet_small/tf_records/train/'
# tfrecord_files = os.listdir(dir)
# for tfrecord in tfrecord_files:
#     path = os.path.join(dir, tfrecord)
#     for example in tf.python_io.tf_record_iterator(path):
#         data = tf.train.Example.FromString(example)
#         encoded_jpg = data.features.feature['image/encoded'].bytes_list.value[0]
#         img = Image.open(io.BytesIO(encoded_jpg))
#         if img.format != 'JPEG':
#             print(tfrecord)
#         
#

dir = 'dataset284_imagenet_small_1024/tf_records/train/'
files = os.listdir(dir)
total_images = 0
for f_i, file in enumerate(files): 
    print(f_i) 
    total_images += sum([1 for _ in tf.python_io.tf_record_iterator(dir + file)])
