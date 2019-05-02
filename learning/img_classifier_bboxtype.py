import shutil
import os
import re
import cv2

# will use them for creating custom directory iterator
import numpy as np
from six.moves import range

# regular expression for splitting by whitespace
splitter = re.compile("\s+")
img_path = './data/img'
base_path = './data/type'

def process_folders():
    # Read the relevant annotation file and preprocess it
    # Assumed that the annotation files are under '<project folder>/data/anno' path
    with open('./data/anno/list_eval_partition.txt', 'r') as eval_partition_file:
        # 파일의 3번째 줄(index : 2)부터 한줄 씩 가져와서 파일 맨 끝의 \n 문자열을 삭제하고 list_eval_partition 객체에 넣는다.
        list_eval_partition = [line.rstrip('\n') for line in eval_partition_file][2:]
        # 가져온 배열에서 공백문자를 기준으로 잘라 배열에 넣는다 -> [[image01, 1], [image02, 2], ...]
        list_eval_partition = [splitter.split(line) for line in list_eval_partition] 
        # 현재 각 줄은 ['img/Sheer_Pleated-Front_Blouse/img_00000001.jpg', 'train'] 이렇게 되어있음
        # 새로운 튜플의 리스트를 만들고 각 튜플의 원소를 3개씩 가지게 하는데
        # 첫번쨰 튜플 원소는 첫번째 문자열의 4번째부터, 즉 img/뒤부터를 원소로 가짐
        # 두번째 튜플 원소는 첫번째 문자열을 / 로 나누고 (img, Sheer...Blouse, img_00001.jpg) 거기서 두번째 거를 또 '_'로 나누고 (Sheer, Pleated-Front, Blouse) 거기서 마지막 것을 원소로 한다, 즉 해당 옷의 종류를 가져온다.
        list_all = [(v[0][4:], v[0].split('/')[1].split('_')[-1], v[1]) for v in list_eval_partition]

    # Put each image into the relevant folder in train/test/validation folder
    # .join 함수 -> 파일명을 상위 디렉토리와 합쳐주는 함수 join('C:\\', path) -> C:\path
    # element -> (img/sheer_...Blouse/img_0001.jpg, Blouse, train)
    for element in list_all:
        # test, train, val 폴더가 존재하는지 확인하고 없으면 만듬
        if not os.path.exists(os.path.join(base_path, element[2])):
            os.mkdir(os.path.join(base_path, element[2]))
        # test, train, val 아래에 Blouse, Button-down 같은 옷 종류 폴더 존재하는지 확인하고 없으면 만든다.
        if not os.path.exists(os.path.join(os.path.join(base_path, element[2]), element[1])):
            os.mkdir(os.path.join(os.path.join(base_path, element[2]), element[1]))
        # te,tr,va/옷종류/ 아래에 옷에 대한 정확한 이름이 존재하는지 (Sheer_Pleated-Front_Blouse 같은) 확인하고 없으면 만든다.
        # 결론적으로 만들어지는 경로 -> ./[train,test,valid]/[cloth_type]/[cloth_name]
        if not os.path.exists(os.path.join(os.path.join(os.path.join(os.path.join(base_path, element[2]), element[1])),
                              element[0].split('/')[0])):
            os.mkdir(os.path.join(os.path.join(os.path.join(os.path.join(base_path, element[2]), element[1])),
                     element[0].split('/')[0]))
        # shutil(셸 유틸리티) move(A, B) 시 A를 B 디렉터리로 옮긴다.
        # move("./data/img/Sheer_Pleated-Front_Blouse/img_00000001.jpg", "./data/img/train/Blouse/Sheer_Pleated-Front_Blouse/img_00000001.jpg")
        # 즉 사진을 만든 폴더에 옮기는 작업을 하는 것.
        shutil.copy2(os.path.join(img_path, element[0]),
                    os.path.join(os.path.join(os.path.join(base_path, element[2]), element[1]), element[0]))
process_folders()
