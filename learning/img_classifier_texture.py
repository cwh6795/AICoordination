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

list_a = []
index_list = []
list_all = []

def process_folders():
    with open('./data/anno/list_attr_cloth.txt', 'r') as list_attr_file:
        list_attr_file2 = [line.rstrip('\n') for line in list_attr_file][2:]

    list_attr_file2 = [splitter.split(line) for line in list_attr_file2]
    for i, v in enumerate(list_attr_file2):
        if v[1] in ['1']:
            list_a.append((v[0], v[1]))
            index_list.append(i)
            
    with open('./' + attr_file_name, 'w') as attr_style_cloth:
        [attr_style_cloth.write('%-40s%-5s\n' %(v[0], v[1])) for v in list_a]

    with open('./data/anno/list_eval_partition.txt', 'r') as eval_partition_file, \
            open('./data/anno/list_attr_img.txt', 'r') as list_attr_file:
        list_eval_partition = [line.rstrip('\n') for line in eval_partition_file][2:]
        list_eval_partition = [splitter.split(line) for line in list_eval_partition] 
        list_attr_img = [line.rstrip('\n') for line in list_attr_file][2:]
        list_attr_img = [splitter.split(line) for line in list_attr_img]
        #i_list -> [img/.../, -1, -1 ]
        # v ->[img/../, train]
    process = 0
    for i_list, v in zip(list_attr_img, list_eval_partition):
        print(process/len(list_eval_partition))
        attr_index_list = []
        #in_list -> [img/.../, 4]
        for i, in_list in enumerate(index_list):
            if i_list[in_list + 1] is '1':
                attr_index_list.append(list_a[i][0])
        if len(attr_index_list):
            list_all.append((i_list[0][4:], attr_index_list, v[1]))
        process += 1

    print('half')
    gc.collect()

    # list_all -> [/img.../, [catlist], train]
    process = 0
    for i in list_all:
        print (i)
    for element in list_all:
        print(process/len(list_all))
        # test, train, val 폴더가 존재하는지 확인하고 없으면 만듬
        if not os.path.exists(os.path.join(base_path, element[2])):
            os.mkdir(os.path.join(base_path, element[2]))
        # test, train, val 아래에 Blouse, Button-down 같은 옷 종류 폴더 존재하는지 확인하고 없으면 만든다.
        for v in element[1]:
            if not os.path.exists(os.path.join(os.path.join(base_path, element[2]), v)):
                os.mkdir(os.path.join(os.path.join(base_path, element[2]), v))
            # te,tr,va/옷종류/ 아래에 옷에 대한 정확한 이름이 존재하는지 (Sheer_Pleated-Front_Blouse 같은) 확인하고 없으면 만든다.
            # 결론적으로 만들어지는 경로 -> ./[train,test,valid]/[cloth_type]/[cloth_name]
            if not os.path.exists(os.path.join(os.path.join(os.path.join(os.path.join(base_path, element[2]), v)),
                                  element[0].split('/')[0])):
                os.mkdir(os.path.join(os.path.join(os.path.join(os.path.join(base_path, element[2]), v)),
                         element[0].split('/')[0]))
            # shutil(셸 유틸리티) move(A, B) 시 A를 B 디렉터리로 옮긴다.
            # move("./data/img/Sheer_Pleated-Front_Blouse/img_00000001.jpg", "./data/img/train/Blouse/Sheer_Pleated-Front_Blouse/img_00000001.jpg")
            # 즉 사진을 만든 폴더에 옮기는 작업을 하는 것.
            shutil.copy2(os.path.join(img_path, element[0]),
                        os.path.join(os.path.join(os.path.join(base_path, element[2]), v), element[0]))
        process += 1


process_folders()

