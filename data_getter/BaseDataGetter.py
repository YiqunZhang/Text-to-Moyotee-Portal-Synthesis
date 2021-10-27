import numpy as np
import cv2 as cv
import random
from data_getter.face_object import FaceObject
from data_getter.embedding import Embedding

def gene_file_name(name_list,lable_list):
    res = ''
    for i in range(len(name_list)):
        res += name_list[i] + '' + lable_list[i] + '-'
    return res[:-1]


class DataGetter():
    def __init__(self):
        self.embedding = Embedding()
        self.main_data = (
            ['HA', None, 1, (0, 0), (120, 120)],
            ['EG', None, 4, (23, 21), (78, 78)],
            ['EY', None, 3, (26, 24), (72, 72)],
            ['MO', None, 2, (72, 45), (30, 30)],
            ['CL', None, 0, (48, 24), (72, 72)]
        )

        self.full_size = (120, 120)
        self.dic = self.get_dic_from_file()

        # 读数据
        for i in self.main_data:
            i[1] = FaceObject(i[0])

    # layer 从 0 开始, 值小表示在下面
    # position 是一个2元组, 分别为 (起点x, 起点y)
    # size 是一个2元组, 分别为 (宽度, 高度)
    def composite(self, img_list, layer_list, position_list, size_list, full_size):
        pic_res = np.ones((full_size[0], full_size[1], 4), dtype=np.int32) * 254

        for i in range(len(layer_list)):
            index = layer_list.index(i)

            img_temp = cv.resize(img_list[index], size_list[index])
            for num_x in range(img_temp.shape[0]):
                for num_y in range(img_temp.shape[1]):
                    pixel = img_temp[num_x, num_y, :]
                    if pixel[3] > 0:
                        pic_res[position_list[index][0] + num_x, position_list[index][1] + num_y, :] = pixel
        return pic_res

    def get_des_text(self, tag_list):
        hair = tag_list[0]
        glasses = tag_list[1]
        eyes = tag_list[2]
        mouth = tag_list[3]
        cloth = 'a ' + tag_list[4]

        res_text = ''

        start_list = ['I have ', 'I am a boy with ', 'I am a cool boy with ', 'I am a smart boy with ',
                      'I am a beautiful boy with ']
        random_index = random.randint(0, len(start_list) - 1)

        res_text += start_list[random_index]

        random_times = random.randint(2, 3)
        random_times = 3
        list_comp = [hair, mouth, eyes]
        if random_times == 1:
            random_times = random.randint(0, 2)
            res_text += list_comp[random_times]
        elif random_times == 2:
            random_times_list = random.sample(range(0, 3), 2)
            res_text += list_comp[random_times_list[0]] + ' and ' + list_comp[random_times_list[1]]
        elif random_times == 3:
            random_times_list = random.sample(range(0, 3), 3)
            res_text += list_comp[random_times_list[0]] + ', ' + list_comp[random_times_list[1]] + ' and ' + list_comp[
                random_times_list[2]]

        list_comp = [glasses, cloth]
        if glasses == 'no glasses':
            random_num = random.randint(0, 2)
            random_num = 2
            if random_num == 0:
                res_text += '.'
            elif random_num == 1:
                random_num = random.randint(0, 1)
                if random_num == 0:
                    res_text += ', and I do not wear' + ' glasses' + '.'
                else:
                    res_text += ', and I wear ' + list_comp[random_num] + '.'
            elif random_num == 2:
                random_num = random.randint(0, 1)
                if random_num == 0:
                    res_text += ', I do not wear' + ' glasses' + ' and wear ' + list_comp[1]
                else:
                    res_text += ', I wear ' + list_comp[1] + ' and do not wear glasses.'
        else:
            random_num = random.randint(0, 2)
            random_num = 2
            if random_num == 0:
                res_text += '.'
            elif random_num == 1:
                random_num = random.randint(0, 1)
                res_text += ', and I wear ' + list_comp[random_num] + '.'
            elif random_num == 2:
                random_times_list = random.sample(range(0, 2), 2)
                res_text += ', and I wear ' + list_comp[random_times_list[0]] + ' and ' + list_comp[
                    random_times_list[1]] + '.'

        return res_text

    def get_dic_from_file(self):
        dic = {}
        with open('data/dic.txt', 'r') as f:
            dic_text = f.read()
        for row in dic_text.split('\n'):
            dic[row.split('#')[0]] = row.split('#')[1]
        return dic

    def get_tag_list(self, label_list):
        tag_list = []
        for i in label_list:
            tag = self.dic[i.split('_')[0]]
            tag_list.append(tag)
        return tag_list

    def gene_pic_with_lable_des(self):
        img_list = []
        layer_list = []
        position_list = []
        size_list = []
        lable_list = []
        name_list = []

        for j in range(len(self.main_data)):
            name_list.append(self.main_data[j][0])
            layer_list.append(self.main_data[j][2])
            position_list.append(self.main_data[j][3])
            size_list.append(self.main_data[j][4])

            pic_label = self.main_data[j][1].get_random_pic()
            img_list.append(pic_label[0])
            lable_list.append(pic_label[1])

        res_img = self.composite(img_list, layer_list, position_list, size_list, self.full_size)
        res_img = res_img.astype(np.uint8)
        res_img = cv.resize(res_img,(64,64))
        res_tag_list = []
        res_des_list = []
        res_embedding_list = []
        for i in range(5):
            res_tag = self.get_tag_list(lable_list)
            res_tag_list.append(res_tag)

            res_des = self.get_des_text(res_tag)
            res_des_list.append(res_des)

            res_embedding = self.embedding.cover(res_des)
            res_embedding_list.append(res_embedding)

        return res_img, res_tag_list, res_des_list, res_embedding_list
