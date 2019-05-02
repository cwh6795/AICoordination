import shutil
import os
import re
import cv2


# will use them for creating custom directory iterator
import numpy as np
from six.moves import range

from keras.models import Model
from keras.layers import Dense
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import DirectoryIterator, ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from keras import backend as K

class DirectoryIteratorWithBoundingBoxes(DirectoryIterator):
    # ("./data/img/train", train_datagen, bounding_boxes=dict_train, target_size=(200, 200))
    # {'/Blouse/She-Fr_Blouse/image0001.jpg': {'x1': x1/300, 'y1': y1/250, 'x2': x2/300, 'y2': y2/250, 'shape': (250, 300, 3)}}
    def __init__(self, directory, image_data_generator, target_size=(256, 256),
                 color_mode: str = 'rgb', classes=None, class_mode: str = 'categorical', batch_size: int = 32,
                 shuffle: bool = True, seed=None, data_format=None, save_to_dir=None,
                 save_prefix: str = '', save_format: str = 'jpeg', follow_links: bool = False):
        # https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/keras/_impl/keras/preprocessing/image.py
        # directory - 이미지 가져올 PATH
        # image_data_generator - 우리가 아는 그것, 왜 DirIter에 사용되는지는 아직 모름
        # target_size - 이미지 불러올 때 이미지 resize할 크기 bbox가 이미 비율 방식으로 정의되어 있기 때문에 사이즈 변경해도 상관없음
        # color_mode - 이미지 RGB, Gray Scale 방식으로 가져올 지 선택
        # classes - 하위 디렉토리 정의, 정의하지 않으면, 자동으로 하위 디렉토리 모두를 가져옴
        # class_mode - 우리 디렉토리 안에 있는게 categorial 하므로 categorical로 정의, binary(두개의 카테고리인듯) None 등으로 설정할 수 있음
        # batch_size - 하나의 Batch size
        # shuffle - epoch 사이에 suffle 할지, seed - data shuffle에 쓰일 seed
        # data_format - keras에서는 기본적으로 channel-last 사용 (num of samp,height, width, channel)가 channel last, (num of samp, channel, width, height)가 channel first
        # save_to_dir - yielded 된 결과물을 저장할 디렉토리(Optional)
        # save_prefix - save_to_dir 옵션 수행될 때 이미지 앞에 붙을 접두어 
        # save_format - save_to_dir 옵션 수행될 때 이미지 저장할 포맷 
        # follow_links - 모름
        super().__init__(directory, image_data_generator, target_size, color_mode, classes, class_mode, batch_size,
                         shuffle, seed, data_format, save_to_dir, save_prefix, save_format, follow_links)

    def next(self):
        """
        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel

        # self.image_shape -> (200, 200, 3)
        # batch_x.shape -> (32, 200, 200, 3)
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())

        #grayscale = false 
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            # index_array에는 읽어들인 이미지들의 index가 저장되어 있다. index_array=[num1,num2,~~]
            fname = self.filenames[j]
            # DirectoryIterator는 이미지들을 읽어들인 다음 filenames에 저장한다. filnames[num1]
            img = image.load_img(os.path.join(self.directory, fname),
                                 grayscale=grayscale,
                                 target_size=self.target_size)
            x = image.img_to_array(img, data_format=self.data_format)
            #x = self.image_data_generator.random_transform(x)
            #x = self.image_data_generator.standardize(x)
            batch_x[i] = x

        # optionally save augmented images to disk for debugging purposes
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), 3), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x

        return batch_x, batch_y

# param1 : imagenet의 weight 값을 초기 값으로 사용하겠다는 것
# param2 : 네트워크 상단의 fully-connected layer 포함 여부 -> 기존 resnet은 1000개의 output 가짐
# 우리는 48개의 출력만 가지면 되므로 우리가 따로 outputlayer를 만들 것 그러므로 포함 시키지 않을 것
# param3 : include_top 이 FALSE 일 때만 사용하는 파라미터, pooling 이후의 결과물이 2D Tensor 가 된다.
# average pooling을 통해서 (none, 7, 7, 2048) -> (none, 2048)의 tensor로 바뀌게 된다.
model_resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model_resnet2 = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# 출력해보니 총 176개의 layer 들이 있고 그 중 마지막 12개만 trainable 하게 만든다.
for layer in model_resnet2.layers[:-12]:
    # 6 - 12 - 18 have been tried. 12 is the best.
    layer.trainable = False

# Tensor("global_average_pooling2d_1/Mean:0", shape=(?, 2048), dtype=float32)
x = model_resnet.output
x = Dense(512, activation='elu', kernel_regularizer=l2(0.001))(x)
x = Dense(128, activation='elu', kernel_regularizer=l2(0.001))(x)
y = Dense(3, activation='softmax', name='img')(x)

x2 = model_resnet2.output
x2 = Dense(512, activation='elu', kernel_regularizer=l2(0.001))(x2)
x2 = Dense(128, activation='elu', kernel_regularizer=l2(0.001))(x2)
y2 = Dense(3, activation='softmax', name='img')(x2)

final_model = Model(inputs=model_resnet.input, outputs=y)

print(final_model.summary())

opt = SGD(lr=0.0001, momentum=0.9, nesterov=True)

final_model.compile(optimizer=opt,
                    loss={'img': 'categorical_crossentropy'},
                    metrics={'img': ['accuracy', 'top_k_categorical_accuracy']# default: top-5
                             })

# https://keras.io/preprocessing/image/
# 이미지 데이터 생성기. -> 이미 존재하는 데이터를 조금 조작함으로써 데이터를 늘릴 수 있음
train_datagen = ImageDataGenerator(rotation_range=30.,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator()

train_iterator = DirectoryIteratorWithBoundingBoxes("./data/top_bot_full/train", train_datagen, target_size=(200, 200))

test_iterator = DirectoryIteratorWithBoundingBoxes("./data/top_bot_full/val", test_datagen, target_size=(200, 200))

# learning rate reducer -> 
lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               patience=12,
                               factor=0.5,
                               verbose=1)
tensorboard = TensorBoard(log_dir='./logs')
early_stopper = EarlyStopping(monitor='val_loss',
                              patience=30,
                              verbose=1)
checkpoint = ModelCheckpoint('./models/top_bot_full.h5')

def custom_generator(iterator):
    while True:
        batch_x, batch_y = iterator.next()
        yield (batch_x, batch_y)

final_model.fit_generator(custom_generator(train_iterator),
                          steps_per_epoch=500,
                          epochs=30, validation_data=custom_generator(test_iterator),
                          validation_steps=100,
                          verbose=1,
                          shuffle=True,
                          callbacks=[lr_reducer, checkpoint, early_stopper, tensorboard],
                          workers=12,
                          use_multiprocessing=True)

final_model2 = Model(inputs=model_resnet2.input, outputs=y2)
final_model2.load_weights('./models/top_bot_full.h5')
final_model2.compile(optimizer=opt,
                    loss={'img': 'categorical_crossentropy'},
                    metrics={'img': ['accuracy', 'top_k_categorical_accuracy']})
                          
final_model2.fit_generator(custom_generator(train_iterator),
                          steps_per_epoch=2000,
                          epochs=200, validation_data=custom_generator(test_iterator),
                          validation_steps=200,
                          verbose=1,
                          shuffle=True,
                          callbacks=[lr_reducer, checkpoint, early_stopper, tensorboard],
                          workers=12,
                          use_multiprocessing=True)
                          

test_datagen = ImageDataGenerator()

test_iterator = DirectoryIteratorWithBoundingBoxes("./data/top_bot_full/test", test_datagen, target_size=(200, 200))

scores = final_model.evaluate_generator(custom_generator(test_iterator), steps=2000)

print('loss: ' + str(scores[0]))
print('Image accuracy: ' + str(scores[1]))
print('top_5_categorical_accuracy: ' + str(scores[2]))


model_json = final_model.to_json()
with open('./models/top_bot_full.json', 'w') as jfile:
	jfile.write(model_json)
