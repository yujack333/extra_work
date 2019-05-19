
# -*- coding: utf-8 -*-
# @Time    : 2019/5/19 1:07
# @Author  : YuYi

from tkinter import *

root = Tk()

text = Text(root, width=30, height=2)  # 30的意思是30个平均字符的宽度，height设置为两行
text.pack()

text.insert(INSERT, 'I Love\n')


mainloop()