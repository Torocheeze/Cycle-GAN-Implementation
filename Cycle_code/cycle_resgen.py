from tensorflow.python.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Activation, BatchNormalization, Add, Lambda, Multiply, Subtract, add
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.activations import sigmoid, relu
from tensorflow.python.keras.layers.convolutional import UpSampling2D, Conv2D
from tensorflow.python.keras.models import Model
from tensorflow.keras import backend as K

def residual_block(y, nb_channels, _strides=(1, 1), _project_shortcut=False):
    shortcut = y

    # down-sampling is performed with a stride of 2
    y = Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)

    y = Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
    y = BatchNormalization()(y)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    y = add([shortcut, y])
    y = LeakyReLU()(y)

    return y

def dk_block(y, filters):
    y = Conv2D(filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(y)
    y = BatchNormalization()(y)
    y = relu(y)

    return y

def uk_block(y, filters, short_cut):
    y = UpSampling2D(size=2)(y)
    y = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')(y)
    y = BatchNormalization()(y)
    y = Concatenate()([y, short_cut])

    return y

def dis_block(y, filters):
    y = Conv2D(filters, kernel_size=4, strides=2, padding='same')(y)
    y = BatchNormalization()(y)
    y = LeakyReLU(alpha=0.2)(y)

    return y

###############################################################################
###############################################################################

def res_generator(image_size = (128,128,3), resblock_num = 3):
    start = Input(shape=image_size)

    # Down layer1
    d1 = Conv2D(64, (7, 7), padding='same', activation='relu')(start)

    # Down layer2 128
    d2 = dk_block(d1, 128)

    # Down layer3 128
    d3 = dk_block(d2, 128)

    # Resnet Block 128
    reblocks = residual_block(d3, 128)
    for _ in range(resblock_num-1):
        reblocks = residual_block(reblocks, 128)

    # Up layer1 128
    u1 = uk_block(reblocks, 128, d2)

    # Up layer2 64
    u2 = uk_block(u1, 64, d1)

    # Up layer3
    u3 = Conv2D(3, (7, 7), padding='same')(u2)
    output = Activation('tanh')(u3)

    model = Model(start, output)
    return model


def discriminator(input_shape = (128, 128, 3), patch = True):
    start = Input(input_shape)

    # dis_layer1
    d1 = Conv2D(64, kernel_size=4, strides=2, padding='same')(start)
    d1 = LeakyReLU(alpha=0.2)(d1)

    # dis_layer2
    d2 = dis_block(d1, 128)

    # dis_layer3
    d3 = dis_block(d2, 256)

    # dis_layer4
    d4 = dis_block(d3, 512)

    # patch or not
    if patch:
        output = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
    else:
        output = Flatten()(d4)
        output = Dense(1)(output)
        output = Activation('Linear')(output)

    return Model(start, output)
