# -*- coding: utf-8 -*-

from acora import AcoraBuilder


#########################################################################
## main 主函数 ##

if __name__ == '__main__':

    builder = AcoraBuilder()

    #self.__builder.add(line.rstrip("\n").decode("utf-8"))
    builder.add("hello")
    builder.add("world")
#    builder.add("海康威视")
#    builder.add("格林美")

#    builder.update(["japan"])

    tree = builder.build()

    hit_list = []

    content_str = "china hello 娃哈哈 海康威视 你好  world he格林美ll hello japan but not happy korea"

    for hit_word, pos in tree.finditer(content_str):

        hit_list.append([hit_word, pos])

    print(hit_list)


    hit_list2 = tree.findall(content_str)
    print(hit_list2)

    print(type(not(0 == len(hit_list2))))
