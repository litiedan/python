import shutil
import os
#第一部分，准备工作，拼接出要存放的文件夹的路径
file = '/home/lzq/file/python/07批量修改文件名/01/01.txt'
file_dir = '/home/lzq/file/python/07批量修改文件名/01/'+'2'
shutil.copy(file,file_dir)


# //深度信息刷新的时间