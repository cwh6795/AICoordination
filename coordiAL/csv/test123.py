import shutil
import os
import re
import cv2
import gc

# will use them for creating custom directory iterator
import numpy as np
from six.moves import range

# regular expression for splitting by whitespace
splitter = re.compile("\s+")
img_path = './data/img_cutted'
base_path = './data/texture'
attr_file_name = 'attr_texture2.txt'

texture_row = {'abstract':0, 'animal':1, 'bird':2, 'butterfly':3, 'camo':4, 'camouflage':5, 'colorblock':6, 'daisy':7, 'destroyed':8, 'diamond':9, 'dot':10, 'grid':11, 
                'lace':12, 'marled':13, 'palm':14, 'ringer':15, 'stripe':16, 'tonal':17, 'zigzag':18}
texture_column = {'abstract':0, 'animal':1, 'bird':2, 'butterfly':3, 'camo':4, 'camouflage':5, 'colorblock':6, 'daisy':7, 'destroyed':8, 'diamond':9, 'dot':10, 'grid':11, 
                'lace':12, 'marled':13, 'palm':14, 'ringer':15, 'stripe':16, 'tonal':17, 'zigzag':18}

style_row = {'athletic':0, 'elegant':1, 'life':2, 'retro':3, 'sweet':4, 'trench':5, 'youth':6}
style_column = {'athletic':0, 'elegant':1, 'life':2, 'retro':3, 'sweet':4, 'trench':5, 'youth':6}

matrix_style = np.zeros((7, 7))
matrix_texture = np.zeros((19, 19))

def process_folders(k, x):
	list_a = []
	index_list = []
	list_all = []
	with open('/media/ml-wony/vol2/ytbML/data/anno/haha.txt', 'r') as list_attr_file, \
		open('/media/ml-wony/vol2/ytbML/data/anno/list_attr_img.txt', 'r') as list_attr_file3:
		list_attr_file2 = [line.rstrip('\n') for line in list_attr_file][2:]
		list_attr_file2 = [splitter.split(line) for line in list_attr_file2]
		for i, v in enumerate(list_attr_file2):
			if v[1] in k:
				list_a.append((v[0], v[1]))
				index_list.append(i)
		print(index_list)
		if x == 1:
			list_attr_img = [line.rstrip('\n') for line in list_attr_file3][2:][:70000]
		elif x == 2:
			list_attr_img = [line.rstrip('\n') for line in list_attr_file3][2:][70000:140000]
		elif x == 3:
			list_attr_img = [line.rstrip('\n') for line in list_attr_file3][2:][140000:210000]
		elif x == 4:
			list_attr_img = [line.rstrip('\n') for line in list_attr_file3][2:][210000:]
		list_attr_img = [splitter.split(line) for line in list_attr_img]

	for i_list in list_attr_img:
		attr_index_list = []
		for i, in_list in enumerate(index_list):
			if i_list[in_list + 1] is '1':
				attr_index_list.append(list_a[i][0])
		if len(attr_index_list) == 2:
			list_all.append((i_list[0][4:], attr_index_list))
			if k == ['8']:
				matrix_style[style_row[attr_index_list[0]]][style_column[attr_index_list[1]]] += 1
			elif k == ['9']:
				matrix_texture[texture_row[attr_index_list[0]]][texture_column[attr_index_list[1]]] += 1
		# elif len(attr_index_list) == 1:
		# 	list_all.append((i_list[0][4:], attr_index_list))
		# 	if k == ['8']:
		# 		matrix_style[style_row[attr_index_list[0]]][style_column[attr_index_list[0]]] += 1
		# 	elif k == ['9']:
		# 		matrix_texture[texture_row[attr_index_list[0]]][texture_column[attr_index_list[0]]] += 1
			

	for string in list_all:
		print(string)

def regu(matrix, div):
	for i in range(len(matrix)):
		for j in range(len(matrix[0])):
			matrix[i][j] = round(float(matrix[i][j]) / float(div), 3)

	return matrix

for i in range(1, 5):
	process_folders(['8'], i)

for i in range(1, 5):
	process_folders(['9'], i)

print(matrix_texture)
print(matrix_style)

matrix_style = matrix_style + np.transpose(matrix_style)
matrix_texture = matrix_texture + np.transpose(matrix_texture)

def file_writer(filename, matrix):
    with open(filename, 'w') as filetxt:
        for row in matrix:
            for i, dot in enumerate(row):
                if i != len(row) - 1:
                    filetxt.write(str(dot) + ',')
                else:
                    filetxt.write(str(dot))
            filetxt.write('\n')

file_writer('./style.csv', matrix_style)
file_writer('./texture.csv', matrix_texture)


matrix_style_regu = regu(matrix_style, float(matrix_style.max()/10.0))
matrix_texture_regu = regu(matrix_texture, float(matrix_texture.max()/10.0))

file_writer('./style_regu.csv', matrix_style_regu)
file_writer('./texture_regu.csv', matrix_texture_regu)
