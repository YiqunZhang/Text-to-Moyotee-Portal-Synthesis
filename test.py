
from data_getter.BaseDataGetter import DataGetter

# 初始化
data_gettter = DataGetter()

for i in range(10000):
    # 取随机数据
    if i % 100 == 0:
        print(i)
    random_data = data_gettter.gene_pic_with_lable_des()
