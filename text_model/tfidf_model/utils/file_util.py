# coding:utf-8
import os



def read_file_content(filename):
    f = open(filename, 'r', encoding='utf-8') # r 代表read
    fileread_str = f.read()
    f.close()
    return fileread_str


def foreach_filepath(filepath):
    file_arr = []
    parents = os.listdir(filepath)
    for parent in parents:
        child = os.path.join(filepath, parent)
        #print(child)
        if os.path.isdir(child):
            file_arr += foreach_all_file(child)
            # print(child)
        else:
            file_arr.append(child)

    return file_arr



if __name__ == '__main__':
    res = read_file_content("file_util.py")
    print(res)
    print(foreach_all_file("D:\\git_code\\server\\server\\AI_Core_Proj\\trunk\\Tools\\"))