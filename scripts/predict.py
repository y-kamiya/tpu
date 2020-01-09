import sys
import os
import json
import tensorflow.compat.v1 as tf
from absl import app
from absl import flags

sys.path.append('./models/official/efficientnet')
sys.path.append('./models/common')

import eval_ckpt_main as eval_ckpt

FLAGS = flags.FLAGS
flags.DEFINE_string('image_dir', default=None, help='')
flags.DEFINE_string('ckpts_dir', default=None, help='')
flags.DEFINE_string('training_labels', default=None, help='')

LABELS_MAP_FILE = 'labels_map.txt'

def predict(labels_map_path):
    model_name = 'efficientnet-b0'

    eval_driver = eval_ckpt.get_eval_driver(model_name)
    eval_glob = '{}/*.jpg'.format(FLAGS.image_dir)
    eval_label = get_eval_label_file(FLAGS.image_dir)

    classes = json.loads(tf.gfile.Open(labels_map_path).read())
    classes_rev = { v:k for k,v in classes.items() }
    synset_classes = [int(classes_rev[label]) for label in tf.gfile.GFile(eval_label).read().split('\n') if label != '']

    image_files = sorted([os.path.join(FLAGS.image_dir, f) for f in os.listdir(FLAGS.image_dir) if is_jpg(f)])

    pred_idx, pred_prob = eval_driver.run_inference(
        FLAGS.ckpts_dir, image_files, synset_classes, True, False)

    show_accuracy(pred_idx, synset_classes)

    show_probs(pred_idx, pred_prob, image_files, synset_classes, classes, False)
    show_probs(pred_idx, pred_prob, image_files, synset_classes, classes, True)

def show_accuracy(pred_idx, classes):
    num_images = len(classes)
    top1_cnt, top3_cnt, top5_cnt = 0.0, 0.0, 0.0
    for i, label in enumerate(classes):
      top1_cnt += label in pred_idx[i][:1]
      top3_cnt += label in pred_idx[i][:3]
      top5_cnt += label in pred_idx[i][:5]
      if i % 10 == 0:
        print('Step {}: top1_acc = {:4.2f}% top3_acc = {:4.2f}% top5_acc = {:4.2f}%'.format(
            i, 100 * top1_cnt / (i + 1), 100 * top3_cnt / (i + 1), 100 * top5_cnt / (i + 1)))
        sys.stdout.flush()
    top1, top3, top5 = 100 * top1_cnt / num_images, 100 * top3_cnt / (i + 1), 100 * top5_cnt / num_images
    print('Final: top1_acc = {:4.2f}%  top3_acc = {:4.2f}%  top5_acc = {:4.2f}%'.format(top1, top3, top5))

def is_jpg(file):
    _, ext = os.path.splitext(file)
    return ext in ['.jpg', '.jpeg']

def show_probs(pred_idx, pred_prob, image_files, synset_classes, classes, is_matched):
    print('### {} images ###'.format('correct' if is_matched else 'wrong')) 
    for i in range(len(synset_classes)):
        true_class = synset_classes[i]

        if is_matched and true_class not in pred_idx[i]:
            continue
        if not is_matched and true_class in pred_idx[i]:
            continue

        print('predicted class for image {}: '.format(image_files[i]))
        for j, idx in enumerate(pred_idx[i]):
            print('  -> top_{} ({:4.2f}%): {}  '.format(j, pred_prob[i][j] * 100,
                                                    classes[str(idx)]))


def get_eval_label_file(dir):
    file = os.path.join(dir, 'synset_labels.txt')
    if os.path.exists(file):
        return file

    file = os.path.join(dir, 'classlist')
    if os.path.exists(file):
        return file

    assert False, 'synset_labels.txt or classlist is necessary in image_dir'

def create_labels_map(synset_labels_path):
    assert os.path.exists(synset_labels_path), 'wrong path to synset_labels.txt'

    with open(synset_labels_path, 'r') as f:
        labels = sorted(list(set(f.read().split('\n'))))
        labels = filter(lambda a: a != '', labels)
        labels_map = {k:v for k, v in enumerate(labels)}

    output_file = os.path.join(FLAGS.ckpts_dir, LABELS_MAP_FILE)
    with open(output_file, 'w') as f:
        json.dump(labels_map, f)

def modify_checkpoint_file():
    checkpoint_path = os.path.join(FLAGS.ckpts_dir, 'checkpoint')
    with open(checkpoint_path, "r") as f:
        contents = f.read()
        replaced = contents.replace('../', '')

    with open(checkpoint_path, "w") as f:
        f.write(replaced)

def main(unused_argv):
    labels_map_path = os.path.join(FLAGS.ckpts_dir, LABELS_MAP_FILE)
    if not os.path.exists(labels_map_path):
        create_labels_map(FLAGS.training_labels)
        modify_checkpoint_file()

    predict(labels_map_path)

if __name__ == '__main__':
    flags.mark_flags_as_required(['image_dir', 'ckpts_dir'])
    app.run(main)
