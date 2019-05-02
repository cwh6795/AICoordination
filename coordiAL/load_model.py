from keras.models import model_from_json
from PIL import Image
import shutil
import os
import sys
import re
import cv2
import random
import time
import numpy as np
from six.moves import range

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'ytbML'))
from matrix_index import *
from keras.models import Model
from keras.layers import Dense
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import DirectoryIterator, ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from keras import backend as K

src_path = './testing'
dest_path = './coordiend'
img_res_path = './DIR'

def getAttr(list):
	sex = []
	style = []
	for name in list:
		if len(name.split('_')) != 2:
			print('name error! : format -> sex_style.jpg ')
			exit()
		else:
			sex.append(name.split('_')[0])
			style.append(name.split('_')[1].split('.')[0])

	return sex, style

def getAttr_ver2(list):
	sex = []
	age = []
	style = []
	for name in list:
		if len(name.split('_')) != 3:
			print('name error! : format -> sex_age_style.jpg ')
			exit()
		else:
			sex.append(name.split('_')[0])
			age.append(name.split('_')[1])
			style.append(name.split('_')[2].split('.')[0])


	return sex, age, style


def loadMatrix():
	men_type_file_ = np.loadtxt('./csv/data/men_type.csv', delimiter=',', dtype=np.int32)
	men_texture_file_ = np.loadtxt('./csv/data/texture_regu.csv', delimiter=',', dtype=np.int32)
	women_type_file_ = np.loadtxt('./csv/data/women_type.csv', delimiter=',', dtype=np.int32)
	women_texture_file_ = np.loadtxt('./csv/data/texture_regu.csv', delimiter=',', dtype=np.int32)

	return [men_type_file_, men_texture_file_, women_type_file_, women_texture_file_]


def men_type_mapping(typeind):
	if typeind == 1 or typeind == 2:
		return typeind + 1
	elif typeind == 3:
		return typeind + 2

def top_type_mapping(typeind):
	if typeind == 0 or typeind == 1:
		return typeind + 1
	elif typeind == 2 or typeind == 3:
		return typeind + 3
	elif typeind >= 4:
		return typeind + 7


def get_max_list(matrix, top_k):
	max_list = []
	index_list = []
	max = 0
	index = [0, 0]
	for x in range(top_k):
		for j in range(len(matrix)):
			for k in range(len(matrix[0])):
				if max < matrix[j][k] and ([j, k] not in index_list):
					max = matrix[j][k]
					index = [j, k]
		max_list.append(max)
		index_list.append(index)
		max = 0
		index = [0, 0]

	return max_list, index_list


def get_rand_list(matrix, top_k):
	rand_list = []
	index_list = []
	i = 0
	while i < top_k:
		rand_row = random.randrange(0, len(matrix))
		rand_col = random.randrange(0, len(matrix[0]))

		if matrix[rand_row][rand_col] != 0:
			max_list.append(999)
			index_list.append([rand_row, rand_col])
		else:
			continue
		i += 1

	return max_list, index_list


texture_json = open('./models/texture/model_texture19.json', 'r')
style_json = open('./models/style/model_style7.json', 'r')
top_bot_full_json = open('./models/top_bot_full/top_bot_full.json', 'r')
top_json = open('./models/top/top.json', 'r')
bot_json = open('./models/bot/bot.json', 'r')

model_texture = texture_json.read()
model_style = style_json.read()
model_top_bot_full = top_bot_full_json.read()
model_top = top_json.read()
model_bot = bot_json.read()

texture_json.close()
style_json.close()
top_bot_full_json.close()
top_json.close()
bot_json.close()

model_texture = model_from_json(model_texture)
model_style = model_from_json(model_style)
model_top_bot_full = model_from_json(model_top_bot_full)
model_top = model_from_json(model_top)
model_bot = model_from_json(model_bot)

model_texture.load_weights('./models/texture/model_texture19.h5')
model_style.load_weights('./models/style/model_style7.h5')
model_top_bot_full.load_weights('./models/top_bot_full/top_bot_full.h5')
model_top.load_weights('./models/top/top.h5')
model_bot.load_weights('./models/bot/bot.h5')

opt = SGD(lr=0.0001, momentum=0.9, nesterov=True)
model_texture.compile(optimizer=opt, loss={'img': 'categorical_crossentropy'}, metrics={'img': ['accuracy', 'top_k_categorical_accuracy']})
model_style.compile(optimizer=opt, loss={'img': 'categorical_crossentropy'}, metrics={'img': ['accuracy', 'top_k_categorical_accuracy']})
model_top_bot_full.compile(optimizer=opt, loss={'img': 'categorical_crossentropy'}, metrics={'img': ['accuracy', 'top_k_categorical_accuracy']})
model_top.compile(optimizer=opt, loss={'img': 'categorical_crossentropy'}, metrics={'img': ['accuracy', 'top_k_categorical_accuracy']})
model_bot.compile(optimizer=opt, loss={'img': 'categorical_crossentropy'}, metrics={'img': ['accuracy', 'top_k_categorical_accuracy']})

men_type_file, men_texture_file, women_type_file, women_texture_file = loadMatrix()

while True:
	img_list = os.listdir(src_path)
	if len(img_list) == 0:
		time.sleep(1)
		continue

	target_list = os.listdir(src_path)
	file_list = [target_list[0]]
	# sex, style = getAttr(file_list)
	sex, age, style = getAttr_ver2(file_list)

	x = np.zeros((len(file_list), 200, 200, 3), dtype=K.floatx())
	for i, img_list in enumerate(file_list):
		k = image.load_img(os.path.join(src_path, img_list), grayscale=False, target_size=(200, 200))
		k = image.img_to_array(k)
		x[i] = k

	texture_ = np.zeros((len(file_list), 19), dtype=K.floatx())
	style_ = np.zeros((len(file_list), 7), dtype=K.floatx())
	top_bot_full_ = np.zeros((len(file_list), 3), dtype=K.floatx())
	top_ = np.zeros((len(file_list), 6), dtype=K.floatx())
	bot_ = np.zeros((len(file_list), 8), dtype=K.floatx())

	texture_ = model_texture.predict(x, batch_size=None, verbose=0, steps=None)
	style_ =  model_style.predict(x, batch_size=None, verbose=0, steps=None)
	top_bot_full_ = model_top_bot_full.predict(x, batch_size=None, verbose=0, steps=None)
	top_ = model_top.predict(x, batch_size=None, verbose=0, steps=None)
	bot_ = model_bot.predict(x, batch_size=None, verbose=0, steps=None)

	for tx, st, to_bo_fu, to, bo in zip(texture_, style_, top_bot_full_, top_, bot_):
		print('----coordinate start-----')
		two = np.argmax(tx)
		thr = np.argmax(st)
		print(to_bo_fu)
		print('to : %s' % TYPE_TOP[np.argmax(to)])
		print('bo : %s' % TYPE_BOT[np.argmax(bo)])
		to_bo_fu_be = to_bo_fu
		to_bo_fu = np.delete(to_bo_fu, 1)
		clo = np.argmax(to_bo_fu)
		if clo == 0:
			if sex[0] == 'men':
				bo = np.delete(bo, 4)
				bo = np.delete(bo, 1)
				fou_be = np.argmax(bo)
				fou = men_type_mapping(fou_be)
			elif sex[0] == 'women':
				fou = np.argmax(bo)
			else:
				print('sex value error')
				exit()
		elif clo == 1:
			fou = np.argmax(to)

		print('\t', two, ' ', TEXTURE[two])
		print('\t', thr, ' ', STYLE[thr])
		#print('\t', clo, ' ', TOP_BOT_FULL[clo])
		print('\t', clo, ' ', TOP_BOT[clo])
		if clo == 0:
			print('\t', fou, ' ', TYPE_BOT[fou])
		else:
			print('\t', fou, ' ', TYPE_TOP[fou])

		if sex[0] == 'men':
			if clo == 0:
				axis_type = men_type_file[fou_be, :]
				axis_texture = men_texture_file[two, :]
				axis_type = np.reshape(axis_type, (1, len(TYPE_TOP)))
				axis_texture = np.reshape(axis_texture, (1, len(TEXTURE)))
			elif clo == 1:
				axis_type = men_type_file[:, fou]
				axis_texture = men_texture_file[:, two]
				axis_type = np.reshape(axis_type, (4, 1))
				axis_texture = np.reshape(axis_texture, (len(TEXTURE), 1))
		elif sex[0] == 'women':
			if clo == 0:
				axis_type = women_type_file[fou, :]
				axis_texture = women_texture_file[two, :]
				axis_type = np.reshape(axis_type, (1, len(TYPE_TOP)))
				axis_texture = np.reshape(axis_texture, (1, len(TEXUTRE)))
			else:
				axis_type = women_type_file[:, fou]
				axis_texture = women_texture_file[:, two]
				axis_type = np.reshape(axis_type, (len(TYPE_BOT), 1))
				axis_texture = np.reshape(axis_texture, (len(TEXTURE), 1))
		else:
			print('sex value error')
			exit()

		if clo == 0:
			last = axis_texture + np.transpose(axis_type)
		elif clo == 1:
			last = np.transpose(axis_texture) + axis_type

		#print(last)
		'''
		while True:
			top_k = input('input top k : ')
			if top_k.isdigit():
				top_k = int(top_k)
				break
			else:
				print('invalid value')
		'''
		top_k = 20

		max_list, index_list = get_max_list(last, top_k)
		rand_list, rand_index = get_rand_list(last, top_k)
		rec_list_val = []
		rec_list_rand = []

		if clo == 0:
			for index in index_list:
				rec_list_val.append((TYPE_TOP[index[0]], TEXTURE[index[1]]))
		elif clo == 1:
			if sex == 'men':
				for index in index_list:
					rec_list_val.append((TYPE_BOT_MEN[index[0]], TEXTURE[index[1]]))
			elif sex == 'women':
				for index in index_list:
					rec_list_val.append((TYPE_BOT_WOMEN[index[0]], TEXTURE[index[1]]))
			else:
				print('sex value error!')
				exit()

		if clo == 0:
			for index in rand_index:
				rec_list_rand.append((TYPE_TOP[index[0]], TEXTURE[index[1]]))
		elif clo == 1:
			if sex == 'men':
				for index in index_list:
					rec_list_rand.append((TYPE_BOT_MEN[index[0]], TEXTURE[index[1]]))
			elif sex == 'women':
				for index in index_list:
					rec_list_rand.append((TYPE_BOT_WOMEN[index[0]], TEXTURE[index[1]]))
			else:
				print('sex value error!')
				exit()

		print('-----/coordinate for top_k value/-----')
		for line in rec_list_val:
			print(line)
		print('-----/coordinate for top_k value/-----')
		print('-----/coordinate for random value/-----')
		for line in rec_list_rand:
			print(line)
		print('-----/coordinate for random value/-----')

		rec_img_array = []
		fin_img_list = []
		for rec in rec_list_val:
			path = os.path.join(os.path.join(os.path.join(img_res_path, sex[0]), rec[0]), rec[1])
			tmp_img_list = os.listdir(path)
			for i, k in enumerate(tmp_img_list):
				tmp_img_list[i] = os.path.join(path, k)
			rec_img_array.append(tmp_img_list)

		print('rec_img_array : ', rec_img_array)
		for img_array in rec_img_array:
			if len(img_array) != 0:
				A = random.randrange(0, len(img_array))
				fin_img_list.append(img_array[A])

		print('fin_img_list : ', fin_img_list)
		for fin_img in fin_img_list:
			shutil.copy2(fin_img, './final/' + fin_img.split('/')[3] + '_' + fin_img.split('/')[4] + '.jpg')

	shutil.move(os.path.join(src_path, file_list[0]), os.path.join(dest_path, file_list[0]))
	time.sleep(1)