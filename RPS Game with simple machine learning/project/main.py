# -*- coding: utf-8 -*-
# @Time    : 2019/5/18 12:40
# @Author  : YuYi
import random
from tkinter import *


class RpsGame:
    def __init__(self, seq_len):
        self.round = 1
        self.N = seq_len
        assert (self.N % 2) == 1
        self.current_seq = ''
        self.seq_frequency = {}
        self.win_statistics = [0, 0, 0]
        self.num2str = ['Rock', 'Paper', 'Scissors', '']
        self.winer = ['Human', 'Computer', 'Ties', '']
        self.strategy = [1, 2, 0]
        self.human_choose = -1
        self.computer_choose = -1
        self.predict_human_choose = -1
        self.result = -1

    def step(self, human_choose):
        self.human_choose = human_choose
        self.predict_human_choose, self.computer_choose = self.make_decision()
        self.result = self.judge(human_choose, self.computer_choose)
        self.win_statistics[self.result] += 1
        self.update_seq_frequency(human_choose)
        self.update_current_seq(human_choose, self.computer_choose)
        self.round += 1

    def update_seq_frequency(self, human_choose):
        if self.round > int((self.N-1)/2):
            if self.current_seq not in self.seq_frequency:
                self.seq_frequency[self.current_seq] = [0, 0, 0]
                self.seq_frequency[self.current_seq][human_choose] = 1
            else:
                self.seq_frequency[self.current_seq][human_choose] += 1

    def update_current_seq(self, human_choose, computer_choose):
        if self.round > int((self.N-1)/2):
            self.current_seq = self.current_seq[2:]
            self.current_seq += str(human_choose)
            self.current_seq += str(computer_choose)
        else:
            self.current_seq += str(human_choose)
            self.current_seq += str(computer_choose)

    def judge(self, human_choose, computer_choose):
        win_condition = [[0, 2], [1, 0], [2, 1]]
        if human_choose == computer_choose:
            result = 2
        elif [human_choose, computer_choose] in win_condition:
            result = 0
        else:
            result = 1
        return result

    def make_decision(self):
        assert len(self.current_seq) <= self.N-1

        if self.current_seq not in self.seq_frequency:
            predict_human_choose = random.randint(0, 2)
        else:
            predict_human_choose = self.argmax(self.seq_frequency[self.current_seq])

        computer_choose = self.strategy[predict_human_choose]
        return predict_human_choose, computer_choose

    def argmax(self, vector):
        max_num = max(vector)
        for i, num in enumerate(vector):
            if num == max_num:
                return i


def interface():
    root.title('Rock Paper Scissors:20 rounds/game')
    Label(root,
          text='Round:%d' % (game.round-1),
          width=30,
          height=1,
          font=("Arial", 24)).grid(row=0)
    Label(root,
          text='Human',
          width=30,
          height=1,
          font=("Arial", 26)).grid(row=1)
    Label(root,
          text='Choose:',
          width=30,
          height=1,
          font=("Arial", 22)).grid(row=2, column=0)
    Button(root,
           text="Rock",
           height=3,
           width=13,
           command=lambda: button(0)).grid(row=2, column=1)
    Button(root,
           text="Paper",
           height=3,
           width=13,
           command=lambda: button(1)).grid(row=2, column=2)
    Button(root,
           text="Scissors",
           height=3,
           width=13,
           command=lambda: button(2)).grid(row=2, column=3)
    Label(root,
          text='Human chooses:%s' % game.num2str[game.human_choose],
          width=30,
          height=1,
          font=("Arial", 22)).grid(row=3)
    Label(root,
          text='Computer',
          width=30,
          height=1,
          font=("Arial", 26)).grid(row=4)
    Label(root,
          text='Predicted human chooses:%s' % game.num2str[game.predict_human_choose],
          width=30,
          height=1,
          font=("Arial", 22)).grid(row=5)
    Label(root,
          text='Therefore computer chooses:%s' % game.num2str[game.computer_choose],
          width=30,
          height=1,
          font=("Arial", 22)).grid(row=6)
    Label(root,
          text='The Winer: %s' % game.winer[game.result],
          width=30,
          height=1,
          font=("Arial", 26)).grid(row=7)
    Label(root,
          text='Statistics',
          width=30,
          height=1,
          font=("Arial", 26)).grid(row=8)
    Label(root,
          text='Human wins: %d' % game.win_statistics[0],
          width=30,
          height=1,
          font=("Arial", 22)).grid(row=9)
    Label(root,
          text='Computer wins: %d' % game.win_statistics[1],
          width=30,
          height=1,
          font=("Arial", 22)).grid(row=10)
    Label(root,
          text='Ties: %d' % game.win_statistics[2],
          width=30,
          height=1,
          font=("Arial", 22)).grid(row=11)


def button(human_choose):
    # global game
    game.step(human_choose)
    interface()


def tk_main():
    interface()
    root.mainloop()


# global
root = Tk()
game = RpsGame(seq_len=5)
if __name__ == '__main__':
    tk_main()

