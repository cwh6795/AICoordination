from keras.models import model_from_json
from PIL import Image
import socket
import time
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
flag = 0
dir_name = ""

class Socket():
	def __init__(self):
		self.PORT = 8008
		self.ServerSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.ServerSocket.bind(("", self.PORT))

	def sock_start(self):
		global dir_name, flag
		print("waiting client")
		self.ServerSocket.listen()
		client, addr = self.ServerSocket.accept()
		self.tmp_client = client
		print(addr, "accessed")
		dir = client.recv(1024)
		print("dir ->", dir.decode())
		dir_name = dir.decode()
		flag = 1

	def sock_end(self):
		global flag
		global send_str
		exx = self.tmp_client.sendall((send_str + "\n").encode())
		print("send_str -> ", send_str)
		print("python sending done", exx)
		flag = 0

sock = Socket()

		

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


def loadMatrix2():
	men_type_file_ = np.loadtxt('./csv/csv_data/men_type.csv', delimiter=',', dtype=np.int32)
	men_texture_file_ = np.loadtxt('./csv/csv_data/men_texture.csv', delimiter=',', dtype=np.int32)
	women_type_file_ = np.loadtxt('./csv/csv_data/women_type.csv', delimiter=',', dtype=np.int32)
	women_texture_file_ = np.loadtxt('./csv/csv_data/women_texture.csv', delimiter=',', dtype=np.int32)

	return [men_type_file_, men_texture_file_, women_type_file_, women_texture_file_]

def loadMatrix():
	men_type_file_ = np.loadtxt('./csv/men_type.csv', delimiter=',', dtype=np.float32)
	men_texture_file_ = np.loadtxt('./csv/men_texture.csv', delimiter=',', dtype=np.float32)
	women_type_file_ = np.loadtxt('./csv/women_type.csv', delimiter=',', dtype=np.float32)
	women_texture_file_ = np.loadtxt('./csv/women_texture.csv', delimiter=',', dtype=np.float32)

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
	max_list2 = []
	index_list2 = []
	max = 0
	index = [0, 0]
	for x in range(top_k):
		for j in range(len(matrix)):
			for k in range(len(matrix[0])):
				if max < matrix[j][k] and ([j, k] not in index_list2):
					max = matrix[j][k]
					index = [j, k]
		max_list2.append(max)
		index_list2.append(index)
		max = 0
		index = [0, 0]

	return max_list2, index_list2


def get_rand_list(matrix, top_k):
	rand_list = []
	index_list = []
	i = 0
	while i < top_k:
		rand_row = random.randrange(0, len(matrix))
		rand_col = random.randrange(0, len(matrix[0]))

		if matrix[rand_row][rand_col] != 0:
			rand_list.append(999)
			index_list.append([rand_row, rand_col])
		else:
			continue
		i += 1

	return rand_list, index_list


def predictStyle(img_name, model_style):
	XX = np.zeros((1, 200, 200, 3), dtype=K.floatx())
	KK = image.load_img(img_name, grayscale=False, target_size=(200, 200))
	KK = image.img_to_array(KK)
	XX[0] = KK
	style_ = model_style.predict(XX, batch_size=None, verbose=0, steps=None)

	style_index = np.argmax(style_[0])

	return STYLE[style_index]

def convert_sum10(tmp_list):
	sum_val = sum(tmp_list)
	new_list = list(tmp_list)
	for i in range(len(new_list)):
		new_list[i] = round((new_list[i] * 10) / sum_val)

	return new_list

def print_array(array):
	for i in range(len(array)):
		for j in range(len(array[i])):
			print('%4d' % int(array[i][j]), end='')
		print()

# main start ===================================================

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
#	img_list = os.listdir(src_path)
#	if len(img_list) == 0:
#		time.sleep(1)
#		continue
	sock.sock_start()
	if flag == 0:
		time.sleep(1)
		continue
	
	src_path = dir_name
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
		if to_bo_fu[0] > to_bo_fu[2]:
			send_str = 'top'
			print("BOT")
		else:
			send_str = 'bot'
			print("TOP")
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
				axis_texture = np.reshape(axis_texture, (1, len(TEXTURE)))
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

		#print_array(last)
		'''
		while True:
			top_k = input('input top k : ')
			if top_k.isdigit():
				top_k = int(top_k)
				break
			else:
				print('invalid value')
		'''
		top_k = 5
		max_list, index_list = get_max_list(last, top_k)
		rand_list, rand_index = get_rand_list(last, top_k)
		rec_list_val = []
		rec_list_rand = []

		if clo == 0:
			for index in index_list:
				rec_list_val.append((TYPE_TOP[index[0]], TEXTURE[index[1]]))
		elif clo == 1:
			if sex[0] == 'men':
				for index in index_list:
					rec_list_val.append((TYPE_BOT_MEN[index[0]], TEXTURE[index[1]]))
			elif sex[0] == 'women':
				for index in index_list:
					rec_list_val.append((TYPE_BOT_WOMEN[index[0]], TEXTURE[index[1]]))
			else:
				print('sex value error!')
				exit()

		if clo == 0:
			for index in rand_index:
				rec_list_rand.append((TYPE_TOP[index[0]], TEXTURE[index[1]]))
		elif clo == 1:
			if sex[0] == 'men':
				for index in index_list:
					rec_list_rand.append((TYPE_BOT_MEN[index[0]], TEXTURE[index[1]]))
			elif sex[0] == 'women':
				for index in index_list:
					rec_list_rand.append((TYPE_BOT_WOMEN[index[0]], TEXTURE[index[1]]))
			else:
				print('sex value error!')
				exit()

		print('-----/coordinate for top_k value/-----')
		for line in rec_list_val:
			print(line)
		print('-----/coordinate for top_k value/-----')

		rec_img_array = []
		fin_img_list = []
		# get img's list in img_res_path('DIR') from recommended attribute
		for rec in rec_list_val:
			# ./DIR/men/jean/abstract
			path = os.path.join(os.path.join(os.path.join(img_res_path, sex[0]), rec[0]), rec[1])
			tmp_img_list = os.listdir(path)
			for i, k in enumerate(tmp_img_list):
				# ./DIR/men/jan/abstract/img0001.jpg
				tmp_string = os.path.join(path, k)
				if style[0] == 'none':
					tmp_img_list[i] = (tmp_string, 'none')
				elif style[0] in STYLE:
					tmp_style = predictStyle(tmp_string, model_style)
					tmp_img_list[i] = (tmp_string, tmp_style)
			# don't mind user's input(style)
			rec_img_array.append(tmp_img_list)

		rec_ratio = convert_sum10(max_list)

		# rec_img_array list's element is list
		# rec_img_array 의 요소인 각 리스트는 (추천할 이미지 경로+이름, 스타일로 지정되어 있다)
		# ex) ('./DIR/women/Skirt/tonal/img_0000008.jpg', 'none')
		rec_img_array = np.array(rec_img_array)
		for i, img_array in enumerate(rec_img_array):
			if len(img_array) != 0:
				# 스타일로 필터링 하지 않을 때
				if style[0] == 'none':
					if len(img_array) < rec_ratio[i]:
							print('이미지의 수가 너무 적어요 보충해야합니다.')
							rec_ratio[i] = len(img_array)

					cho_index = np.random.choice(range(0, len(img_array)), int(rec_ratio[i]), replace=False)
					if len(cho_index) != 0:
						for q in cho_index:
							fin_img_list.append(img_array[q][0])
				# 스타일이 지정되어 그것으로 필터링 해야 할 때
				elif style[0] in STYLE:
					tmp_list = []
					for X in range(len(img_array)):
						if img_array[X][1] == style[0]:
							tmp_list.append(img_array[X])

					if len(tmp_list) < rec_ratio[i]:
							print('이미지의 수가 너무 적어요 보충해야합니다.(style)')
							rec_ratio[i] = len(tmp_list)

					if len(tmp_list) != 0:
						cho_index = np.random.choice(range(0, len(tmp_list)), int(rec_ratio[i]), replace=False)
						for q in cho_index:
							fin_img_list.append(tmp_list[q][0])



		print('fin_img_list : ', fin_img_list)
		for i, fin_img in enumerate(fin_img_list):
			shutil.copy2(fin_img, './recommend/' + fin_img.split('/')[3] + '_' + fin_img.split('/')[4] + '_' + style[0] + str(i) + '.jpg')
	# move user's cloth picture
	shutil.move(os.path.join(src_path, file_list[0]), os.path.join(dest_path, file_list[0]))
	sock.sock_end()
	time.sleep(1)

