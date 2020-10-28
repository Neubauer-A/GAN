import tensorflow as tf
from functools import partial
from numpy import asarray
from os import listdir
from PIL import Image

AUTOTUNE = tf.data.experimental.AUTOTUNE

def size_scale_model(img_size):
    return tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.Resizing(img_size, img_size, interpolation='bilinear'),
            tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./127.5, offset=-1)])

def load_image(filename):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)
    image.close()
    return pixels

def image_example(image_string, label=False):
    feature = ({
        'image': _bytes_feature(image_string),
        'label': _bytes_feature(label)
    }
    if label
    else {'image': _bytes_feature(image_string),})
    return tf.train.Example(features=tf.train.Features(feature=feature))

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
      value = value.numpy() 
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class ImageProcessor:
    def __init__(self, img_size, num_classes=1):
        self.img_size = img_size
        self.ensure_shape = [img_size, img_size, 3]
        self.num_classes = num_classes
    
    def make_tfrec(self, directory, record_file, label=None):
        resize_rescale = size_scale_model(self.img_size)
        if label:
            label = tf.keras.utils.to_categorical(label-1, num_classes=self.num_classes)
            label = tf.io.serialize_tensor(label)
        with tf.io.TFRecordWriter(record_file) as writer:
            for filename in listdir(directory):
                image_array = load_image(directory+'/'+filename)
                image_array = resize_rescale(image_array)
                image_string = tf.io.serialize_tensor(image_array)
                if label:
                    tf_example = image_example(image_string, label)
                    writer.write(tf_example.SerializeToString())
                    continue
                else:
                    tf_example = image_example(image_string)
                    writer.write(tf_example.SerializeToString())

    def load_dataset(self, records, labeled=False, batch_size=32, shuffle=1000):
        ds = tf.data.TFRecordDataset(records)
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        ds = ds.with_options(ignore_order)
        ds = ds.map(partial(self.parse_image, labeled=labeled), num_parallel_calls=AUTOTUNE)
        return ds.shuffle(shuffle).batch(batch_size).prefetch(buffer_size=AUTOTUNE)

    def parse_image(self, example_proto, labeled):
        image_feature_description = ({
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.string),
        }
        if labeled
        else {'image': tf.io.FixedLenFeature([], tf.string),})
        example = tf.io.parse_single_example(example_proto, image_feature_description)
        feature = tf.ensure_shape(tf.io.parse_tensor(example['image'], 
                                  out_type='float32'), self.ensure_shape)
        if labeled:
            label = tf.ensure_shape(tf.io.parse_tensor(example['label'], 
                                  out_type='float32'), [self.num_classes])
            return feature, label
        return feature