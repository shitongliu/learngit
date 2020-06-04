# Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang.
# "Image Super-Resolution Using Deep Convolutional Networks"(SRCNN)
#
# http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html
from __future__ import absolute_import, division, print_function, unicode_literals
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime


def read_images_from_dir(directory, flags=None):
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
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0]

        all_images.append(np.expand_dims(img, axis=-1))

    return all_images, all_names


def decompose(img, f_sub, stride):
    """
    decompose original image(GrayScale) into sub-images.

    :param img:     original image
    :param f_sub:   size of sub-images, like (h, w, ...)
    :param stride:  stride
    :return:        sub-image batch. shape=(cnt, f_sub[0], f_sub[1], 1)
    """
    sub_images = []
    h, w, _ = img.shape
    xrange = list(range(f_sub[0], h, stride))
    xrange.append(h - 1 - f_sub[0])
    yrange = list(range(f_sub[1], w, stride))
    yrange.append(w - 1 - f_sub[1])
    for i in xrange:
        for j in yrange:
            # sub_images = [sub_images, img[i:i+f_sub[0], j:j+f_sub[1]]]
            temp = np.expand_dims(img[i-f_sub[0]:i, j-f_sub[1]:j], axis=0)
            sub_images.append(temp)
    sub_images = np.concatenate(sub_images, axis=0)
    return sub_images


def compute_psnr(y_true, y_pred, max_value=1.0):
    """
    compute psnr.

    :param y_true:      ground true
    :param y_pred:      predict value
    :param max_value:   max value in above two images
    :return:            psnr
    """
    diff = np.double(y_true) - np.double(y_pred)
    rmse = np.sqrt(np.mean(diff ** 2))
    psnr = 20 * np.log10(max_value / rmse)
    return psnr


print('start.')

# global parameter
up_scale = 3            # upscaling factor
f_sub = (33, 33, 1)     # patch size
stride = 14             # patch stride
n1, n2 = 64, 32         # filter numbers of layer 1,2
f1, f2, f3 = 9, 5, 5    # filter size of layer 1,2,3
lr = 1e-3               # learning rate
mt = 0.9                # momentum of delta (delta[i+1] = mt*delta[i] + ...)
batch_size = 256        # size of training batch
epochs = 2             # training epochs
remark = ''

# load the luminance(HR images) and bicubic(LR interpolated image) images
load_from_disk = True
if load_from_disk:
    all_hr_images, all_img_names = read_images_from_dir("Train\\HR_Y", flags=cv2.IMREAD_GRAYSCALE)
    all_bic_images, _ = read_images_from_dir("Train\\Bicubic(x{})".format(up_scale), flags=cv2.IMREAD_GRAYSCALE)
else:
    all_hr_images, all_img_names = read_images_from_dir("Train\\HR_RGB", flags=cv2.IMREAD_COLOR)
    all_bic_images = []
    all_imgs_and_names = dict(zip(all_img_names, all_hr_images))
    Path("Train\\HR_Y").mkdir(parents=True, exist_ok=True)
    Path("Train\\Bicubic(x{})".format(up_scale)).mkdir(parents=True, exist_ok=True)
    for name, img_hr in all_imgs_and_names.items():
        img_blur = cv2.GaussianBlur(src=img_hr, ksize=(5, 5), sigmaX=1)
        img_lr = cv2.resize(
            src=img_blur, dsize=(0, 0), fx=1 / up_scale, fy=1 / up_scale,
            interpolation=cv2.INTER_CUBIC
        )
        img_bic = cv2.resize(
            src=img_lr, dsize=img_hr.shape[:2][::-1],
            interpolation=cv2.INTER_CUBIC
        )
        all_bic_images.append(np.expand_dims(img_bic, axis=-1))

        cv2.imwrite(filename=str((Path.cwd()) / Path("Train\\HR_Y")) + '\\' + name,
                    img=img_hr)
        cv2.imwrite(filename=str((Path.cwd()) / Path("Train\\Bicubic(x{})".format(up_scale))) + '\\' + name,
                    img=img_bic)

# decompose into sub-images
all_hr_patches = []
all_bic_patches = []
for img in all_hr_images:
    all_hr_patches.append(decompose(img, f_sub, stride))
for img in all_bic_images:
    all_bic_patches.append(decompose(img, f_sub, stride))
all_hr_patches = np.concatenate(all_hr_patches, axis=0)
all_bic_patches = np.concatenate(all_bic_patches, axis=0)

# normalize and shuffle
all_hr_patches = all_hr_patches / 255.      # y
all_bic_patches = all_bic_patches / 255.    # x
permu = np.random.permutation(all_hr_patches.shape[0])  # shuffle the first dim
all_hr_patches = all_hr_patches[permu, :, :, :]
all_bic_patches = all_bic_patches[permu, :, :, :]


# build and compile the model
model = tf.keras.Sequential([
    # Patch extraction and representation
    layers.Conv2D(
        n1, [f1, f1], padding='same', activation='relu',
        input_shape=(None, None, 1),
        # Gaussian distribution and weights regularize leads to
        # bizarre training curve and worse performance, why?
        # kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.001),
        # bias_initializer=tf.initializers.zeros,
        # kernel_regularizer=regularizers.l2(0.001)
    ),
    # Non-linear mapping
    layers.Conv2D(
        n2, [f2, f2], padding='same', activation='relu',
        # kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.001),
        # bias_initializer=tf.initializers.zeros,
        kernel_regularizer=regularizers.l2(0.001)
    ),
    # Reconstruction
    layers.Conv2D(
        1, [f3, f3], padding='same',
        # kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.001),
        # bias_initializer=tf.initializers.zeros,
        kernel_regularizer=regularizers.l2(0.001)
    )
])
model.compile(
    # SGD leads to poor performance too, why?
    # optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=mt),
    optimizer='adam',
    loss='mse',
    metrics=['mse']
)
model.summary()

# training
date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = "logs/" + date_str
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
training_num = np.int(all_bic_patches.shape[0] * 0.9)
history = model.fit(
    x=all_bic_patches[:training_num, :, :, :],
    y=all_hr_patches[:training_num, :, :, :],
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    callbacks=[tensorboard_callback],
    validation_data=(
        all_bic_patches[training_num:, :, :, :],
        all_hr_patches[training_num:, :, :, :]
    )
)

# save the weights
model.save_weights(
    './models/SRCNN_{}_F{}-{}-{}_N{}-{}-1_TIME'.format(remark, f1, f2, f3, n1, n2) + date_str
)

# visualize the training progress
mse = history.history['mse']
psnr = 20*np.log10(1.0 / np.sqrt(mse))
val_mse = history.history['val_mse']
val_psnr = 20*np.log10(1.0 / np.sqrt(val_mse))

plt.figure()
plt.plot(mse, label='mse')
plt.plot(val_mse, label='val_mse')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()

plt.figure()
plt.plot(psnr, label='psnr')
plt.plot(val_psnr, label='val_psnr')
plt.xlabel('Epoch')
plt.ylabel('PSNR')
plt.legend()
plt.show()

print('lowest training mse:{:6.4f}'.format(np.max(mse)))
print('lowest val mse:{:6.4f}'.format(np.max(val_mse)))
print('highest training psnr:{:4.2f}'.format(np.max(psnr)))
print('highest val psnr:{:4.2f}'.format(np.max(val_psnr)))

# use Set5 for test
all_test_paths = (Path.cwd()) / Path("Test\\Set5")
all_test_paths = list(all_test_paths.glob('*'))
all_test_paths = [str(path) for path in all_test_paths]
cnt = 0
total_psnr = 0
for path in all_test_paths:
    test_name = str(path.split('\\')[-1])

    test_rgb = cv2.imread(path, flags=cv2.IMREAD_COLOR)
    test_hr = (cv2.cvtColor(test_rgb, cv2.COLOR_BGR2YCrCb) / 255.0)[:, :, 0]
    test_hr = np.expand_dims(test_hr, (0, -1))

    test_blur = cv2.GaussianBlur(src=test_hr[0, :, :, 0], ksize=(5, 5), sigmaX=1)
    test_lr = cv2.resize(
        src=test_blur[0, :, :, 0], dsize=(0, 0), fx=1 / up_scale, fy=1 / up_scale,
        interpolation=cv2.INTER_CUBIC
    )
    test_bic = cv2.resize(
        src=test_lr, dsize=test_hr.shape[1:3][::-1],
        interpolation=cv2.INTER_CUBIC
    )
    test_bic = np.expand_dims(test_bic, (0, -1))

    # infer and get the psnr
    test_sr = model(test_bic)
    bic_psnr = compute_psnr(test_hr[0, :, :, 0], test_bic[0, :, :, 0])
    sr_psnr = compute_psnr(test_hr[0, :, :, 0], test_sr[0, :, :, 0])

    print(test_name + ', bic_psnr:{:4.2f}, sr_psnr:{:4.2f}'.format(bic_psnr, sr_psnr))
    cnt += 1
    total_psnr += sr_psnr

print('average psnr: {:4.2f}'.format(total_psnr / cnt))

print('done.')
