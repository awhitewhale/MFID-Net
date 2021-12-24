import os
import shutil

'''
把path文件夹中所有png文件重命名为000001.png, 000002.png, ...
'''

def alter_fileName(target_path):
    n = 0
    filelist = os.listdir(target_path)
    for i in filelist:
        oldname = target_path+os.sep+filelist[n]
        newname = target_path+os.sep+str(n+1).zfill(6)+i[-4:]
        os.rename(oldname, newname)
        n += 1

def file_copy(path,target_path):
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith('.png'):
                list = (os.path.join(root, name))
                shutil.copy(list, target_path)

path = "/home/data/RESIDE/ITS/val/clear"
target_path="/home/data/RESIDE/ITS/val/renamed_clear"
file_copy(path, target_path)
alter_fileName(target_path)