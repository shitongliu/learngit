import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow_examples.models.pix2pix import pix2pix
from pathlib import Path
from scipy.io import loadmat
from IPython.display import clear_output

# global parameter
OUTPUT_CHANNELS = 2.
BATCH_SIZE = 32
EPOCHS = 40
VAL_SUBSPLITS = 5


def read_images_from_disk(directory, flags=None):
    """
    read images(luminance) from specified directory.

    :param directory:   root
    :param flags:       flags for image read.
    :return:            file name and all images in above root
                        ,type: list
    """
    all_images = []
    all_paths = (Path.cwd()) / Path(directory)
    all_paths = list(all_paths.glob('*'))
    all_paths = [str(path) for path in all_paths]
    all_names = [path.split('\\')[-1] for path in all_paths]

    for path in all_paths:
        img = cv2.imread(path, flags=flags)
        all_images.append(img)

    return all_images, all_names


def load_mask_from_disk(directory):
    all_mask = []
    all_paths = (Path.cwd()) / Path(directory)
    all_paths = list(all_paths.glob('*'))
    all_paths = [str(path) for path in all_paths]
    all_names = [path.split('\\')[-1] for path in all_paths]

    for path in all_paths:
        gt = loadmat(path)
        mask = gt['groundTruth'][0, 0]['Segmentation'][0, 0]
        all_mask.append(mask)

    return all_mask, all_names


def normalize(input_images, input_masks, for_training=True):
    output_images = []
    output_masks = []
    for image, mask in zip(input_images, input_masks):
        image = tf.image.resize(image, (128, 128))
        mask = tf.image.resize(np.repeat(np.expand_dims(mask, -1), 3, -1), (128, 128))

        if for_training:
            if tf.random.uniform(()) > 0.3:
                image_aug = tf.cast(tf.image.flip_left_right(image), tf.float32) / 255.0
                mask_aug = tf.image.flip_left_right(mask)[:, :, 0]
                mask_aug = np.round(mask_aug * (OUTPUT_CHANNELS-1) / np.max(mask_aug))
                mask_aug = tf.cast(mask_aug, tf.float32)
                mask_aug = tf.expand_dims(mask_aug, -1)
                output_images.append(tf.expand_dims(image_aug, 0))
                output_masks.append(tf.expand_dims(mask_aug, 0))

        image = tf.cast(image, tf.float32) / 255.0
        mask = mask[:, :, 0]
        mask = np.round(mask * (OUTPUT_CHANNELS-1) / np.max(mask))
        mask = tf.cast(mask, tf.float32)
        mask = tf.expand_dims(mask, -1)

        output_images.append(tf.expand_dims(image, 0))
        output_masks.append(tf.expand_dims(mask, 0))

    output_images = tf.concat(output_images, 0)
    output_masks = tf.concat(output_masks, 0)

    return output_images, output_masks


def unet_model(output_channels):
    base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)
    # 使用这些层的激活设置
    layer_names = [
        'block_1_expand_relu',  # 64x64
        'block_3_expand_relu',  # 32x32
        'block_6_expand_relu',  # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',  # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # 创建特征提取模型（编码器）
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
    down_stack.trainable = False

    # 解码器
    up_stack = [
        pix2pix.upsample(512, 3, apply_dropout=True),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3, apply_dropout=True),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3, apply_dropout=True),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3, apply_dropout=True),  # 32x32 -> 64x64
    ]

    # 这是模型的最后一层
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2,
        padding='same', activation='softmax'
    )  # 64x64 -> 128x128

    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    x = inputs

    # 在模型中降频取样
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # 升频取样然后建立跳跃连接
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def show_prediction(sample_image, sample_mask):
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])


print('start.')

training_images, training_names = read_images_from_disk(r"data\images\train", flags=cv2.IMREAD_COLOR)
validate_images, validate_names = read_images_from_disk(r"data\images\val", flags=cv2.IMREAD_COLOR)
test_images, test_names = read_images_from_disk(r"data\images\test", flags=cv2.IMREAD_COLOR)

training_masks, _ = load_mask_from_disk(r"data\groundTruth\train")
validate_masks, _ = load_mask_from_disk(r"data\groundTruth\val")
test_masks, _ = load_mask_from_disk(r"data\groundTruth\test")

training_images, training_masks = normalize(training_images, training_masks)
validate_images, validate_masks = normalize(validate_images, validate_masks)
test_images, test_masks = normalize(test_images, test_masks, for_training=False)

# no = 100
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(training_images[no, :, :, :])
# plt.xticks([])
# plt.yticks([])
# plt.subplot(1, 2, 2)
# plt.imshow(training_masks[no, :, :])
# plt.xticks([])
# plt.yticks([])
# plt.show()

model = unet_model(OUTPUT_CHANNELS)
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_prediction(validate_images[10], validate_masks[10])


training_len = len(training_names)
test_len = len(test_names)
STEPS_PER_EPOCH = training_len // BATCH_SIZE
VALIDATION_STEPS = test_len // BATCH_SIZE // VAL_SUBSPLITS
model_history = model.fit(
    x=training_images,
    y=training_masks,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    # callbacks=[DisplayCallback()],
    validation_steps=VALIDATION_STEPS,
    validation_data=(validate_images, validate_masks)
)

show_prediction(validate_images[5], validate_masks[5])

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()

print('done.')
