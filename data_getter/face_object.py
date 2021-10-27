import random
import imageio
import os

class FaceObject:

    def __init__(self, name):
        self.name = name
        self.pic = []
        self.lable = []
        self.read_pic()

        # self.layer = layer
        # self.position = position
        # self.size = size

    def read_pic(self):
        self.pic = []
        self.lable = []
        # 获取文件列表
        file_list = os.listdir('data/'+self.name)

        # 遍历文件
        for file_name in file_list:
            # 检查文件拓展名
            try:
                real_name = file_name.split('.png')[0]
                extension_name = file_name.split('.png')[1]
            except IndexError:
                continue

            # 读图
            pic_temp = imageio.imread('data/' + self.name + '/' + file_name)

            # 添加结果
            self.pic.append(pic_temp)
            self.lable.append(real_name)

    def get_len(self):
        return len(self.pic)

    def get_random_pic(self):
        index = random.randrange(0, self.get_len())
        return self.pic[index], self.lable[index]

