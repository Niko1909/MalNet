

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

(mal_train, mal_valid, mal_test), mal_info = tfds.load(
    'malaria',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`.""" 
    image = tf.image.resize(image, [128, 128]) #I changed this, be careful
    return tf.cast(image, tf.float32) / 255., label

# def augment_img(image, label, seed=2):
#     image = tf.image.resize_with_crop_or_pad(image, 106, 106)
#     new_seed = tf.random.experimental.stateless_split(seed, num=1)[0,:]
#     image = tf.image.stateless_random_crop(image, size=[100,100,3], seed=seed)
#     image = tf.image.stateless_random_jpeg_quality(image, 85, 100,seed=new_seed)

mal_train = mal_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
# mal_train = mal_train.map(augment_img, num_parallel_calls=tf.data.AUTOTUNE)
mal_train = mal_train.cache()
mal_train = mal_train.shuffle(mal_info.splits['train'].num_examples)
mal_train = mal_train.batch(128)
mal_train = mal_train.prefetch(tf.data.AUTOTUNE)

mal_valid = mal_valid.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
mal_valid = mal_valid.batch(128)
mal_valid = mal_valid.cache()
mal_valid = mal_valid.prefetch(tf.data.AUTOTUNE)

mal_test = mal_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
mal_test = mal_test.batch(128)
mal_test = mal_test.cache()
mal_test = mal_test.prefetch(tf.data.AUTOTUNE)

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(128, 2, 2, padding='same', activation='selu'),
  tf.keras.layers.MaxPool2D(2, 2),
  tf.keras.layers.Conv2D(64, 2, 2, padding='same', activation='selu'),
  tf.keras.layers.MaxPool2D(2, 2),
  tf.keras.layers.Conv2D(32, 2, 2, padding='same', activation='selu'),
  tf.keras.layers.MaxPool2D(2, 2),
  tf.keras.layers.Conv2D(16, 2, 2, padding='same', activation='selu'),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(16)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(
    mal_train,
    epochs=20,
    validation_data=mal_valid,
)

model.save('MalariaNet_V1')
