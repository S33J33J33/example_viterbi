'''
作者：宋建军
时间：2021年2月6日
简介：使用了维特比（viterbi）算法
题目：
从前有个村儿，村里的人的身体情况只有两种可能：健康或者发烧。
假设这个村儿的人没有体温计或者百度这种神奇东西，他唯一判断他身体情况的途径就是到村头我的偶像金正月的小诊所询问。
月儿通过询问村民的感觉，判断她的病情，再假设村民只会回答正常、头晕或冷。
有一天村里奥巴驴就去月儿那去询问了。
第一天她告诉月儿她感觉正常。
第二天她告诉月儿感觉有点冷。
第三天她告诉月儿感觉有点头晕。
那么问题来了，月儿如何根据阿驴的描述的情况，推断出这三天中阿驴的一个身体状态呢?

因此，从HMM隐马尔科夫模型的角度来说，健康或者发烧 是身体隐含的状态state，而正常，头晕或者冷是观察序列，

因此这三天的观察序列为：正常。冷，头晕，而每天的身体隐含状况则未知，这在HMM问题中属于解码问题，可以通过维特比算法，
利用动态规划算法，来求解出整个HMM链中每天的隐含状态。

已知情况：
隐含的身体状态 = {健康，发烧}
可观察的感觉状态 = {正常，冷，头晕}
初始状态序列={健康：0.6，发烧：0.4}
转换概率
健康->健康：0.7
健康->发烧：0.3
发烧->健康：0.4
发烧->发烧：0.6
观测序列分布
    正常 冷 头晕
健康：0.5 0.4 0.1
发烧：0.1 0.3 0.6
'''


import numpy as np


# 定义算法函数
def viterbi(p_start, p_tran, p_observe, r_observe):
    # 根据观察到结果矩阵判断需要迭代多少次循环
    temp = len(r_observe)
    # 用来存放算法得出的结果，每一天是什么状态
    r_real = []
    # 用以存放中间过程中的概率
    P_temp = 0

    # 以下一部分是第一次迭代的过程
    P_t0 = p_observe[[0],[r_observe[0]-1]]*p_start[0]
    # 假设这一天是健康状态下发生当前情况的概率
    # print(P_t0)
    P_t1 = p_observe[[1],[r_observe[0]-1]]*p_start[1]
    # 假设这一天是发烧状态下发生当前情况的概率
    # print(p_observe[1,0])
    if P_t0>P_t1:
        # 如果健康的概率高，那么这一天就是健康的，结果矩阵添加0
        r_real.append(0)
        P_temp = P_t0
    else:
        r_real.append(1)
        P_temp = P_t1

    # 以下部分是迭代剩余过程
    for i in range(1,temp):
        P_t0 = P_temp*p_observe[0,[r_observe[i]-1]]*p_tran[r_real[-1],0]
        P_t1 = P_temp*p_observe[1,[r_observe[i]-1]]*p_tran[r_real[-1],1]
        if P_t0>P_t1:
            r_real.append(0)
            P_temp = P_t0
        else:
            r_real.append(1)
            P_temp = P_t1

    # 打印算法预测的结果
    # 0：健康 1：发烧
    print(r_real)
    return r_real


# 初始状态矩阵
p_start = np.array([0.6, 0.4])
# 状态转移矩阵
p_tran = np.array([[0.7, 0.3], [0.4, 0.6]])
# 观测矩阵
p_observe = np.array([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]])
# 实际观察到每天结果的矩阵
# 1：健康 2：冷 3：头晕
r_observe = np.array([1, 2, 3])

viterbi(p_start ,p_tran ,p_observe ,r_observe)
