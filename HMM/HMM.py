# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict

class HMM:
    def __init__(self):
        self.obs_list = []    #观测值
        self.state_list = []  #状态
        self.N = 0   #状态数， =len(self.state_list)
        self.M = 0   #观测值数 =len(self.obs_list)
        self.initial_prob = None     #初始状态概率矩阵
        self.transition_prob = None  #状态转移概率矩阵
        self.emission_prob = None    #发射概率矩阵
        self.epsilon = 1e-8 #偏置，防止log0或乘0

    def load_data(self, data_path):
        obs_seq_list = []  # 观测序列列表
        state_seq_list = []  # 状态序列列表
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                char, label = line.strip().split()  # 按空格分隔每个字符和标签
                obs_seq_list.append(char)
                state_seq_list.append(label)

                if char not in self.obs_list:
                    self.obs_list.append(char)
                if label not in self.state_list:
                    self.state_list.append(label)
        
        if 'UNK' not in self.obs_list:
            self.obs_list.append('UNK')

        self.M = len(self.obs_list)
        self.N = len(self.state_list)

        return obs_seq_list, state_seq_list

    def train(self, obs_seq_list, state_seq_list):
        '''
        训练模型
        '''
        #初始化概率矩阵
        self.initial_prob = np.zeros(self.N)
        self.transition_prob = np.zeros((self.N, self.N))
        self.emission_prob = np.zeros((self.N, self.M))

        print("initialize training...")
        #计算初始概率矩阵
        state_index_dict = defaultdict(int)
        for i, state in enumerate(self.state_list):
            state_index_dict[state] = i
        for state_seq in state_seq_list:
            self.initial_prob[state_index_dict[state_seq]] += 1
        self.initial_prob[self.initial_prob == 0] = self.epsilon
        self.initial_prob /= len(state_seq_list)
        #计算状态转移概率矩阵
        for i in range(len(state_seq_list) - 1):
            curr_state = state_index_dict[state_seq_list[i]]
            next_state = state_index_dict[state_seq_list[i + 1]]
            self.transition_prob[curr_state][next_state] += 1
        #计算发射概率矩阵
        for i in range(len(state_seq_list)):
            state = state_index_dict[state_seq_list[i]]
            obs = self.obs_list.index(obs_seq_list[i])
            self.emission_prob[state][obs] += 1

        for i in range(self.N):
            self.transition_prob[i][self.transition_prob[i] == 0] = self.epsilon
            self.emission_prob[i][self.emission_prob[i] == 0] = self.epsilon
            self.transition_prob[i] = self.transition_prob[i] / sum(self.transition_prob[i])
            self.emission_prob[i] = self.emission_prob[i] / sum(self.emission_prob[i])

        self.initial_prob = np.log(self.initial_prob)
        self.transition_prob = np.log(self.transition_prob)
        self.emission_prob = np.log(self.emission_prob)
        print("DONE!")

    def viterbi(self, obs_seq):
        """
        维特比解码算法
        """
        T = len(obs_seq)
        delta = np.zeros((T, self.N))
        psi = np.zeros((T, self.N))

        # 初始化 delta 和 psi 的第一列
        for i in range(self.N):
            delta[0][i] = self.initial_prob[i] + self.emission_prob[i][obs_seq[0]]
            psi[0][i] = 0

        # 计算 delta 和 psi 的其他列
        for t in range(1, T):
            for j in range(self.N):
                max_delta = float('-inf') #前一时刻状态转移的最大概率值
                max_index = None #最大概率所对应的前一时刻隐藏状态
                for i in range(self.N):
                    prob = delta[t - 1][i] + self.transition_prob[i][j]
                    if prob > max_delta:
                        max_delta = prob
                        max_index = i
                delta[t][j] = max_delta + self.emission_prob[j][obs_seq[t]]
                psi[t][j] = max_index

        # 回溯找出最佳路径
        state_seq = np.zeros(T, dtype=int) # 最优路径序列
        state_seq[T - 1] = np.argmax(delta[T - 1])
        for t in range(T - 2, -1, -1):
            state_seq[t] = psi[t + 1][state_seq[t + 1]]

        return [self.state_list[i] for i in state_seq]
    
    def predict(self, test_file, output_file):
        obs_seq_list = []   #观测序列列表
        state_seq_list = [] #状态序列列表

        with open(test_file, "r", encoding="utf-8") as f:
            test_lines = f.readlines()
        
        with open(output_file, "w", encoding="utf-8") as f:
            for line in test_lines:
                line = line.strip()
                if line:
                    char, label = line.split()
                    if char in self.obs_list:
                        obs_seq_list.append(self.obs_list.index(char))
                    else:
                        obs_seq_list.append(self.obs_list.index("UNK"))
                    state_seq_list.append(label)
            
            pred_state_seq_list = self.viterbi(obs_seq_list)

            i = 0
            for line in test_lines:
                line = line.strip()
                if not line:
                    f.write("\n")
                else:
                    char, __ = line.split()
                    f.write(char + " " + pred_state_seq_list[i] + "\n")
                    i += 1

    def predict1(self, test_file):
        # 加载测试数据
        obs_seq_list = []   #观测序列列表
        state_seq_list = [] #状态序列列表

        with open(test_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                char, label = line.split()
                if char in self.obs_list:
                    obs_seq_list.append(self.obs_list.index(char))
                else:
                    obs_seq_list.append(self.obs_list.index("UNK"))
                state_seq_list.append(label)

        # 利用维特比算法进行解码
        pred_state_seq_list = self.viterbi(obs_seq_list)

        accuracy = sum(1 for i in range(len(state_seq_list)) if state_seq_list[i] == pred_state_seq_list[i]) / len(state_seq_list)
        return accuracy, pred_state_seq_list


if __name__ == '__main__':
    hmm = HMM()
    train_path = "HMM/Project2/NER/Chinese/train.txt"
    #train_path = "HMM/Project2/NER/English/train.txt"
    validation_path = "HMM/Project2/NER/Chinese/validation.txt"
    #validation_path = "HMM/Project2/NER/English/validation.txt"
    my_path = "HMM/my_pred_Chinese.txt"
    #my_path = "HMM/my_pred_English.txt"
    
    obs_seq_list, state_seq_list = hmm.load_data(train_path)
    hmm.train(obs_seq_list, state_seq_list)
    hmm.predict(validation_path,my_path)
    #acc, pred= hmm.predict1(validation_path)
    #print(f"accuracy:{(acc*100):.4f}%")
