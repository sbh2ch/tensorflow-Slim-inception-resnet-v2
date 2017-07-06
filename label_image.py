import os
import tensorflow as tf
from datasets import imagenet
from nets import inception_resnet_v2
from preprocessing import inception_preprocessing

# set your .ckpt file
checkpoints_dir = 'C:/Users/kiost/IdeaProjects/tensor-models\slim'

slim = tf.contrib.slim

batch_size = 3
image_size = 299

with tf.Graph().as_default():
    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        # set your image
        imgPath = 'panda.jpg'
        testImage_string = tf.gfile.FastGFile(imgPath, 'rb').read()
        testImage = tf.image.decode_jpeg(testImage_string, channels=3)
        processed_image = inception_preprocessing.preprocess_image(testImage, image_size, image_size, is_training=False)
        processed_images = tf.expand_dims(processed_image, 0)

        logits, _ = inception_resnet_v2.inception_resnet_v2(processed_images, num_classes=1001, is_training=False)
        probabilities = tf.nn.softmax(logits)

        init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(checkpoints_dir, 'inception_resnet_v2_2016_08_30.ckpt'),
            slim.get_model_variables('InceptionResnetV2'))

        with tf.Session() as sess:
            init_fn(sess)

            np_image, probabilities = sess.run([processed_images, probabilities])
            probabilities = probabilities[0, 0:]
            sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]

            names = imagenet.create_readable_names_for_imagenet_labels()
            for i in range(15):
                index = sorted_inds[i]
                print((probabilities[index], names[index]))
