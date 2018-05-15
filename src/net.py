import numpy as np
import chainer
from chainer import functions as F
from chainer import links as L

class VGG19(chainer.Chain):

    mean = np.asarray([104, 117, 124], dtype=np.float32)

    @classmethod
    def preprocess(cls, image, input_type='RGB'):
        if input_type == 'RGB':
            image = image[:,:,::-1]
        image = np.rollaxis(image - cls.mean, 2)
        return image.reshape((1,) + image.shape)

    @classmethod
    def postprocess(cls, image, output_type='RGB'):
        image = image.reshape(image.shape[1:])
        image = np.transpose(image, (1, 2, 0)) + cls.mean
        if output_type == 'RGB':
            return image[:,:,::-1]
        else:
            return image

    def __init__(self):
        super(VGG19, self).__init__(

            conv1 = L.Convolution2D(3, 32, 3, stride=2, pad=1, nobias=True),
            conv1_bn = L.BatchNormalization(32, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv1_sc = L.Scale(1, (32,), bias_term=True),
            # relu activation
            conv2_1_ex = L.Convolution2D(32, 32, 1, stride=1, pad=1, nobias=True),
            conv2_1_ex_bn = L.BatchNormalization(32, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv2_1_ex_sc = L.Scale(1, (32,), bias_term=True),
            # relu activation
            conv2_1_dw = L.DepthwiseConvolution2D(32, 1, 3, stride=1, pad=1, nobias=True),
            conv2_1_dw_bn = L.BatchNormalization(32, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv2_1_dw_sc = L.Scale(1, (32,), bias_term=True),
            # relu activation
            conv2_1_ln = L.Convolution2D(32, 16, 1, stride=1, pad=1, nobias=True),
            conv2_1_ln_bn = L.BatchNormalization(16, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv2_1_ln_sc = L.Scale(1, (16,), bias_term=True),
            # relu activation
            conv2_2_ex = L.Convolution2D(16, 96, 1, stride=1, pad=1, nobias=True),
            conv2_2_ex_bn = L.BatchNormalization(96, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv2_2_ex_sc = L.Scale(1, (96,), bias_term=True),
            # relu activation
            conv2_2_dw = L.DepthwiseConvolution2D(96, 1, 3, stride=2, pad=1, nobias=True),
            conv2_2_dw_bn = L.BatchNormalization(96, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv2_2_dw_sc = L.Scale(1, (96,), bias_term=True),
            # relu activation
            conv2_2_ln = L.Convolution2D(96, 24, 1, stride=1, pad=1, nobias=True),
            conv2_2_ln_bn = L.BatchNormalization(24, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv2_2_ln_sc = L.Scale(1, (24,), bias_term=True),
            # relu activation
            conv3_1_ex = L.Convolution2D(24, 144, 1, stride=1, pad=1, nobias=True),
            conv3_1_ex_bn = L.BatchNormalization(144, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv3_1_ex_sc = L.Scale(1, (144,), bias_term=True),
            # relu activation
            conv3_1_dw = L.DepthwiseConvolution2D(144, 1, 3, stride=2, pad=1, nobias=True),
            conv3_1_dw_bn = L.BatchNormalization(144, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv3_1_dw_sc = L.Scale(1, (144,), bias_term=True),
            # relu activation
            conv3_1_ln = L.Convolution2D(144, 24, 1, stride=1, pad=1, nobias=True),
            conv3_1_ln_bn = L.BatchNormalization(24, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv3_1_ln_sc = L.Scale(1, (24,), bias_term=True),
            # relu activation
            conv3_2_ex = L.Convolution2D(24, 144, 1, stride=1, pad=1, nobias=True),
            conv3_2_ex_bn = L.BatchNormalization(144, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv3_2_ex_sc = L.Scale(1, (144,), bias_term=True),
            # relu activation
            conv3_2_dw = L.DepthwiseConvolution2D(144, 1, 3, stride=2, pad=1, nobias=True),
            conv3_2_dw_bn = L.BatchNormalization(144, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv3_2_dw_sc = L.Scale(1, (144,), bias_term=True),
            # relu activation
            conv3_2_ln = L.Convolution2D(144, 32, 1, stride=1, pad=1, nobias=True),
            conv3_2_ln_bn = L.BatchNormalization(32, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv3_2_ln_sc = L.Scale(1, (32,), bias_term=True),
            # relu activation
            conv4_1_ex = L.Convolution2D(32, 192, 1, stride=1, pad=1, nobias=True),
            conv4_1_ex_bn = L.BatchNormalization(192, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv4_1_ex_sc = L.Scale(1, (192,), bias_term=True),
            # relu activation
            conv4_1_dw = L.DepthwiseConvolution2D(192, 1, 3, stride=2, pad=1, nobias=True),
            conv4_1_dw_bn = L.BatchNormalization(192, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv4_1_dw_sc = L.Scale(1, (192,), bias_term=True),
            # relu activation
            conv4_1_ln = L.Convolution2D(192, 32, 1, stride=1, pad=1, nobias=True),
            conv4_1_ln_bn = L.BatchNormalization(32, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv4_1_ln_sc = L.Scale(1, (32,), bias_term=True),
            # relu activation
            conv4_2_ex = L.Convolution2D(32, 192, 1, stride=1, pad=1, nobias=True),
            conv4_2_ex_bn = L.BatchNormalization(192, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv4_2_ex_sc = L.Scale(1, (192,), bias_term=True),
            # relu activation
            conv4_2_dw = L.DepthwiseConvolution2D(192, 1, 3, stride=2, pad=1, nobias=True),
            conv4_2_dw_bn = L.BatchNormalization(192, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv4_2_dw_sc = L.Scale(1, (192,), bias_term=True),
            # relu activation
            conv4_2_ln = L.Convolution2D(192, 32, 1, stride=1, pad=1, nobias=True),
            conv4_2_ln_bn = L.BatchNormalization(32, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv4_2_ln_sc = L.Scale(1, (32,), bias_term=True),
            # relu activation
            conv4_3_ex = L.Convolution2D(32, 192, 1, stride=1, pad=1, nobias=True),
            conv4_3_ex_bn = L.BatchNormalization(192, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv4_3_ex_sc = L.Scale(1, (192,), bias_term=True),
            # relu activation
            conv4_3_dw = L.DepthwiseConvolution2D(192, 1, 3, stride=2, pad=1, nobias=True),
            conv4_3_dw_bn = L.BatchNormalization(192, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv4_3_dw_sc = L.Scale(1, (192,), bias_term=True),
            # relu activation
            conv4_3_ln = L.Convolution2D(192, 64, 1, stride=1, pad=1, nobias=True),
            conv4_3_ln_bn = L.BatchNormalization(64, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv4_3_ln_sc = L.Scale(1, (64,), bias_term=True),
            # relu activation
            conv4_4_ex = L.Convolution2D(64, 384, 1, stride=1, pad=1, nobias=True),
            conv4_4_ex_bn = L.BatchNormalization(384, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv4_4_ex_sc = L.Scale(1, (384,), bias_term=True),
            # relu activation
            conv4_4_dw = L.DepthwiseConvolution2D(384, 1, 3, stride=2, pad=1, nobias=True),
            conv4_4_dw_bn = L.BatchNormalization(384, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv4_4_dw_sc = L.Scale(1, (384,), bias_term=True),
            # relu activation
            conv4_4_ln = L.Convolution2D(384, 64, 1, stride=1, pad=1, nobias=True),
            conv4_4_ln_bn = L.BatchNormalization(64, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv4_4_ln_sc = L.Scale(1, (64,), bias_term=True),
            # relu activation
            conv4_5_ex = L.Convolution2D(64, 384, 1, stride=1, pad=1, nobias=True),
            conv4_5_ex_bn = L.BatchNormalization(384, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv4_5_ex_sc = L.Scale(1, (384,), bias_term=True),
            # relu activation
            conv4_5_dw = L.DepthwiseConvolution2D(384, 1, 3, stride=2, pad=1, nobias=True),
            conv4_5_dw_bn = L.BatchNormalization(384, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv4_5_dw_sc = L.Scale(1, (384,), bias_term=True),
            # relu activation
            conv4_5_ln = L.Convolution2D(384, 64, 1, stride=1, pad=1, nobias=True),
            conv4_5_ln_bn = L.BatchNormalization(64, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv4_5_ln_sc = L.Scale(1, (64,), bias_term=True),
            # relu activation
            conv4_6_ex = L.Convolution2D(64, 384, 1, stride=1, pad=1, nobias=True),
            conv4_6_ex_bn = L.BatchNormalization(384, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv4_6_ex_sc = L.Scale(1, (384,), bias_term=True),
            # relu activation
            conv4_6_dw = L.DepthwiseConvolution2D(384, 1, 3, stride=2, pad=1, nobias=True),
            conv4_6_dw_bn = L.BatchNormalization(384, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv4_6_dw_sc = L.Scale(1, (384,), bias_term=True),
            # relu activation
            conv4_6_ln = L.Convolution2D(384, 64, 1, stride=1, pad=1, nobias=True),
            conv4_6_ln_bn = L.BatchNormalization(64, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv4_6_ln_sc = L.Scale(1, (64,), bias_term=True),
            # relu activation
            conv4_7_ex = L.Convolution2D(64, 384, 1, stride=1, pad=1, nobias=True),
            conv4_7_ex_bn = L.BatchNormalization(384, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv4_7_ex_sc = L.Scale(1, (384,), bias_term=True),
            # relu activation
            conv4_7_dw = L.DepthwiseConvolution2D(384, 1, 3, stride=2, pad=1, nobias=True),
            conv4_7_dw_bn = L.BatchNormalization(384, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv4_7_dw_sc = L.Scale(1, (384,), bias_term=True),
            # relu activation
            conv4_7_ln = L.Convolution2D(384, 96, 1, stride=1, pad=1, nobias=True),
            conv4_7_ln_bn = L.BatchNormalization(96, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv4_7_ln_sc = L.Scale(1, (96,), bias_term=True),
            # relu activation
            conv5_1_ex = L.Convolution2D(96, 576, 1, stride=1, pad=1, nobias=True),
            conv5_1_ex_bn = L.BatchNormalization(576, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv5_1_ex_sc = L.Scale(1, (576,), bias_term=True),
            # relu activation
            conv5_1_dw = L.DepthwiseConvolution2D(576, 1, 3, stride=2, pad=1, nobias=True),
            conv5_1_dw_bn = L.BatchNormalization(576, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv5_1_dw_sc = L.Scale(1, (576,), bias_term=True),
            # relu activation
            conv5_1_ln = L.Convolution2D(576, 96, 1, stride=1, pad=1, nobias=True),
            conv5_1_ln_bn = L.BatchNormalization(96, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv5_1_ln_sc = L.Scale(1, (96,), bias_term=True),
            # relu activation
            conv5_2_ex = L.Convolution2D(96, 576, 1, stride=1, pad=1, nobias=True),
            conv5_2_ex_bn = L.BatchNormalization(576, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv5_2_ex_sc = L.Scale(1, (576,), bias_term=True),
            # relu activation
            conv5_2_dw = L.DepthwiseConvolution2D(576, 1, 3, stride=2, pad=1, nobias=True),
            conv5_2_dw_bn = L.BatchNormalization(576, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv5_2_dw_sc = L.Scale(1, (576,), bias_term=True),
            # relu activation
            conv5_2_ln = L.Convolution2D(576, 96, 1, stride=1, pad=1, nobias=True),
            conv5_2_ln_bn = L.BatchNormalization(96, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv5_2_ln_sc = L.Scale(1, (96,), bias_term=True),
            # relu activation
            conv5_3_ex = L.Convolution2D(96, 576, 1, stride=1, pad=1, nobias=True),
            conv5_3_ex_bn = L.BatchNormalization(576, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv5_3_ex_sc = L.Scale(1, (576,), bias_term=True),
            # relu activation
            conv5_3_dw = L.DepthwiseConvolution2D(576, 1, 3, stride=2, pad=1, nobias=True),
            conv5_3_dw_bn = L.BatchNormalization(576, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv5_3_dw_sc = L.Scale(1, (576,), bias_term=True),
            # relu activation
            conv5_3_ln = L.Convolution2D(576, 160, 1, stride=1, pad=1, nobias=True),
            conv5_3_ln_bn = L.BatchNormalization(160, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv5_3_ln_sc = L.Scale(1, (160,), bias_term=True),
            # relu activation
            conv6_1_ex = L.Convolution2D(160, 960, 1, stride=1, pad=1, nobias=True),
            conv6_1_ex_bn = L.BatchNormalization(960, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv6_1_ex_sc = L.Scale(1, (960,), bias_term=True),
            # relu activation
            conv6_1_dw = L.DepthwiseConvolution2D(960, 1, 3, stride=2, pad=1, nobias=True),
            conv6_1_dw_bn = L.BatchNormalization(960, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv6_1_dw_sc = L.Scale(1, (960,), bias_term=True),
            # relu activation
            conv6_1_ln = L.Convolution2D(960, 160, 1, stride=1, pad=1, nobias=True),
            conv6_1_ln_bn = L.BatchNormalization(160, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv6_1_ln_sc = L.Scale(1, (160,), bias_term=True),
            # relu activation
            conv6_2_ex = L.Convolution2D(160, 960, 1, stride=1, pad=1, nobias=True),
            conv6_2_ex_bn = L.BatchNormalization(960, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv6_2_ex_sc = L.Scale(1, (960,), bias_term=True),
            # relu activation
            conv6_2_dw = L.DepthwiseConvolution2D(960, 1, 3, stride=2, pad=1, nobias=True),
            conv6_2_dw_bn = L.BatchNormalization(960, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv6_2_dw_sc = L.Scale(1, (960,), bias_term=True),
            # relu activation
            conv6_2_ln = L.Convolution2D(960, 160, 1, stride=1, pad=1, nobias=True),
            conv6_2_ln_bn = L.BatchNormalization(160, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv6_2_ln_sc = L.Scale(1, (160,), bias_term=True),
            # relu activation
            conv6_3_ex = L.Convolution2D(160, 960, 1, stride=1, pad=1, nobias=True),
            conv6_3_ex_bn = L.BatchNormalization(960, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv6_3_ex_sc = L.Scale(1, (960,), bias_term=True),
            # relu activation
            conv6_3_dw = L.DepthwiseConvolution2D(960, 1, 3, stride=2, pad=1, nobias=True),
            conv6_3_dw_bn = L.BatchNormalization(960, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv6_3_dw_sc = L.Scale(1, (960,), bias_term=True),
            # relu activation
            conv6_3_ln = L.Convolution2D(960, 320, 1, stride=1, pad=1, nobias=True),
            conv6_3_ln_bn = L.BatchNormalization(320, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv6_3_ln_sc = L.Scale(1, (320,), bias_term=True),
            # relu activation
            conv6_4 = L.Convolution2D(320, 1280, 1, stride=1, pad=1, nobias=True),
            conv6_4_bn = L.BatchNormalization(1280, decay=0.9, eps=1e-5, dtype=np.float32, use_gamma=False, use_beta=False),
            conv6_4_sc = L.Scale(1, (1280,), bias_term=True),
            # relu activation
            fc7 = L.Convolution2D(320, 1000, 1, stride=1, pad=1, nobias=True),
        )

    def __call__(self, x): # remember about "blocks elementwise multiplication"
        # TODO: add every layer name + "pools" + "blocks"
        layer_names = [ '1', '1_bn', '1_sc',
                        '2_1_ex', '2_1_ex_bn', '2_1_ex_sc',
                        '2_1_dw', '2_1_dw_bn', '2_1_dw_sc',
                        '2_1_ln', '2_1_ln_bn', '2_1_ln_sc',
                        '2_2_ex', '2_2_ex_bn', '2_2_ex_sc',
                        '2_2_dw', '2_2_dw_bn', '2_2_dw_sc',
                        '2_2_ln', '2_2_ln_bn', '2_2_ln_sc',
                        '3_1_ex', '3_1_ex_bn', '3_1_ex_sc',
                        '3_1_dw', '3_1_dw_bn', '3_1_dw_sc',
                        '3_1_ln', '3_1_ln_bn', '3_1_ln_sc',
                        '3_2_ex', '3_2_ex_bn', '3_2_ex_sc',
                        '3_2_dw', '3_2_dw_bn', '3_2_dw_sc',
                        '3_2_ln', '3_2_ln_bn', '3_2_ln_sc',
                        '4_1_ex', '4_1_ex_bn', '4_1_ex_sc',
                        '4_1_dw', '4_1_dw_bn', '4_1_dw_sc',
                        '4_1_ln', '4_1_ln_bn', '4_1_ln_sc',
                        '4_2_ex', '4_2_ex_bn', '4_2_ex_sc',
                        '4_2_dw', '4_2_dw_bn', '4_2_dw_sc',
                        '4_2_ln', '4_2_ln_bn', '4_2_ln_sc',
                        '4_3_ex', '4_3_ex_bn', '4_3_ex_sc',
                        '4_3_dw', '4_3_dw_bn', '4_3_dw_sc',
                        '4_3_ln', '4_3_ln_bn', '4_3_ln_sc',
                        '4_4_ex', '4_4_ex_bn', '4_4_ex_sc',
                        '4_4_dw', '4_4_dw_bn', '4_4_dw_sc',
                        '4_4_ln', '4_4_ln_bn', '4_4_ln_sc',
                        '4_5_ex', '4_5_ex_bn', '4_5_ex_sc',
                        '4_5_dw', '4_5_dw_bn', '4_5_dw_sc',
                        '4_5_ln', '4_5_ln_bn', '4_5_ln_sc',
                        '4_6_ex', '4_6_ex_bn', '4_6_ex_sc',
                        '4_6_dw', '4_6_dw_bn', '4_6_dw_sc',
                        '4_6_ln', '4_6_ln_bn', '4_6_ln_sc',
                        '4_7_ex', '4_7_ex_bn', '4_7_ex_sc',
                        '4_7_dw', '4_7_dw_bn', '4_7_dw_sc',
                        '4_7_ln', '4_7_ln_bn', '4_7_ln_sc',
                        '5_1_ex', '5_1_ex_bn', '5_1_ex_sc',
                        '5_1_dw', '5_1_dw_bn', '5_1_dw_sc',
                        '5_1_ln', '5_1_ln_bn', '5_1_ln_sc',
                        '5_2_ex', '5_2_ex_bn', '5_2_ex_sc',
                        '5_2_dw', '5_2_dw_bn', '5_2_dw_sc',
                        '5_2_ln', '5_2_ln_bn', '5_2_ln_sc',
                        '5_3_ex', '5_3_ex_bn', '5_3_ex_sc',
                        '5_3_dw', '5_3_dw_bn', '5_3_dw_sc',
                        '5_3_ln', '5_3_ln_bn', '5_3_ln_sc',
                        '6_1_ex', '6_1_ex_bn', '6_1_ex_sc',
                        '6_1_dw', '6_1_dw_bn', '6_1_dw_sc',
                        '6_1_ln', '6_1_ln_bn', '6_1_ln_sc',
                        '6_2_ex', '6_2_ex_bn', '6_2_ex_sc',
                        '6_2_dw', '6_2_dw_bn', '6_2_dw_sc',
                        '6_2_ln', '6_2_ln_bn', '6_2_ln_sc',
                        '6_3_ex', '6_3_ex_bn', '6_3_ex_sc',
                        '6_3_dw', '6_3_dw_bn', '6_3_dw_sc',
                        '6_3_ln', '6_3_ln_bn', '6_3_ln_sc',
                        '6_4', '6_4_bn', '6_4_sc']
        layers = {}
        h = x
        for layer_name in layer_names:
            #conv1
            #bn
            #relu
            if layer_name.endswith('bn')
            h = F.relu(self['conv' + layer_name](h))
            layers[layer_name] = h
        return layers
        # TODO: check if it works
