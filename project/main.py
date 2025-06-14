# 导入相关包
import os
import random
import numpy as np
from Maze import Maze
from Runner import Runner
from QRobot import QRobot
from ReplayDataSet import ReplayDataSet
from torch_py.MinDQNRobot import MinDQNRobot as TorchRobot # PyTorch版本
from keras_py.MinDQNRobot import MinDQNRobot as KerasRobot # Keras版本
import matplotlib.pyplot as plt

import numpy as np

# 机器人移动方向
move_map = {
    'u': (-1, 0), # up
    'r': (0, +1), # right
    'd': (+1, 0), # down
    'l': (0, -1), # left
}


# 迷宫路径搜索树
class SearchTree(object):


    def __init__(self, loc=(), action='', parent=None):
        """
        初始化搜索树节点对象
        :param loc: 新节点的机器人所处位置
        :param action: 新节点的对应的移动方向
        :param parent: 新节点的父辈节点
        """

        self.loc = loc  # 当前节点位置
        self.to_this_action = action  # 到达当前节点的动作
        self.parent = parent  # 当前节点的父节点
        self.children = []  # 当前节点的子节点

    def add_child(self, child):
        """
        添加子节点
        :param child:待添加的子节点
        """
        self.children.append(child)

    def is_leaf(self):
        """
        判断当前节点是否是叶子节点
        """
        return len(self.children) == 0


def expand(maze, is_visit_m, node):
    """
    拓展叶子节点，即为当前的叶子节点添加执行合法动作后到达的子节点
    :param maze: 迷宫对象
    :param is_visit_m: 记录迷宫每个位置是否访问的矩阵
    :param node: 待拓展的叶子节点
    """
    can_move = maze.can_move_actions(node.loc)
    for a in can_move:
        new_loc = tuple(node.loc[i] + move_map[a][i] for i in range(2))
        if not is_visit_m[new_loc]:
            child = SearchTree(loc=new_loc, action=a, parent=node)
            node.add_child(child)


def back_propagation(node):
    """
    回溯并记录节点路径
    :param node: 待回溯节点
    :return: 回溯路径
    """
    path = []
    while node.parent is not None:
        path.insert(0, node.to_this_action)
        node = node.parent
    return path


def breadth_first_search(maze):
    """
    对迷宫进行广度优先搜索
    :param maze: 待搜索的maze对象
    """
    start = maze.sense_robot()
    root = SearchTree(loc=start)
    queue = [root]  # 节点队列，用于层次遍历
    h, w, _ = maze.maze_data.shape
    is_visit_m = np.zeros((h, w), dtype=np.int)  # 标记迷宫的各个位置是否被访问过
    path = []  # 记录路径
    while True:
        current_node = queue[0]
        is_visit_m[current_node.loc] = 1  # 标记当前节点位置已访问

        if current_node.loc == maze.destination:  # 到达目标点
            path = back_propagation(current_node)
            break

        if current_node.is_leaf():
            expand(maze, is_visit_m, current_node)

        # 入队
        for child in current_node.children:
            queue.append(child)

        # 出队
        queue.pop(0)

    return path
    
def my_search(maze):
    """
    任选深度优先搜索算法、最佳优先搜索（A*)算法实现其中一种
    :param maze: 迷宫对象
    :return :到达目标点的路径 如：["u","u","r",...]
    """

    path = []

    # -----------------请实现你的算法代码--------------------------------------
    start = maze.sense_robot() 
    root = SearchTree(loc=start)
    stack = [root]  # 使用栈结构实现DFS
    h, w, _ = maze.maze_data.shape  
    is_visit_m = np.zeros((h,  w), dtype=np.int) 
    
    while stack:
        current_node = stack.pop()   # 弹出栈顶元素
        
        # 终点判断必须放在标记访问前，否则可能漏判最后一个节点
        if current_node.loc  == maze.destination: 
            path = back_propagation(current_node)
            break 
        
        if is_visit_m[current_node.loc]  == 1:
            continue 
        
        is_visit_m[current_node.loc]  = 1  # 标记当前节点为已访问 
        
        # 叶子节点扩展时需要生成子节点 
        if current_node.is_leaf(): 
            expand(maze, is_visit_m, current_node)
        
        # 逆序压栈保证子节点按原顺序处理（例如u方向优先于r方向）
        for child in reversed(current_node.children): 
            # 压栈前检查子节点是否已被访问（避免重复路径）
            if is_visit_m[child.loc] == 0:  
                stack.append(child) 
    # -----------------------------------------------------------------------
    return path


import torch
import torch.nn  as nn
import numpy as np 
from collections import deque
 
class DQN(nn.Module):
    """ 深度Q网络结构设计 """
    '''def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net  = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, state):
        return self.net(state) 
    '''
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        
        # 特征提取层：深度网络结构
        self.feature_net  = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),  # 层归一化提升稳定性 
            nn.GELU(),          # 高级激活函数
            
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )
        
        # Dueling DQN架构高效实现
        self.value_head  = nn.Linear(128, 1)         # 状态价值函数 V(s)
        self.advantage_head  = nn.Linear(128, output_dim)  # 动作优势函数 A(s,a)
        
        # 优化的参数初始化 
        for m in self.modules(): 
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight,  nonlinearity='relu')
                nn.init.constant_(m.bias,  0.1)
 
    def forward(self, state):
        # 特征提取
        features = self.feature_net(state) 
        
        # Dueling DQN分支计算 
        state_value = self.value_head(features) 
        action_advantages = self.advantage_head(features) 
        
        # 组合价值与优势：Q(s,a) = V(s) + [A(s,a) - mean(A(s,a))]
        return state_value + (action_advantages - action_advantages.mean(dim=0,  keepdim=True))

from QRobot import QRobot

class Robot(QRobot):

    def __init__(self, maze):
        """
        初始化 Robot 类
        :param maze:迷宫对象
        """
        super(Robot, self).__init__(maze)
        self.maze = maze
        
        self.state_dim  = 2  # 坐标(x,y)
        self.action_map  = {'u':0, 'd':1, 'l':2, 'r':3}
        self.actions  = list(self.action_map.keys()) 
        self.pos_arr = []
        self.prev_reward = 0
        # 关键超参数优化 
        self.gamma  = 0.99     # 折扣因子 
        self.epsilon  = 0.8    # 初始探索率（提高）
        self.epsilon_min  = 0.01  # 最小探索率 
        self.epsilon_decay  = 0.995  # 衰减率（减缓）
        self.batch_size  = 64   # 批次大小 
        self.memory  = deque(maxlen=10000)  # 经验回放池 
        
        # 网络架构 
        self.policy_net  = DQN(self.state_dim,  len(self.actions)) 
        self.target_net  = DQN(self.state_dim,  len(self.actions)) 
        self.target_net.load_state_dict(self.policy_net.state_dict()) 
        self.optimizer  = torch.optim.Adam(self.policy_net.parameters(),  lr=0.0005)  # 降低学习率 
        
        # 训练状态跟踪 
        self.previous_pos  = None  # 记录前一个位置 
        self.update_counter  = 0   # 网络更新计数器 
        self.target_update_freq  = 100  # 目标网络更新频率 
        

    def train_update(self):
        """
        以训练状态选择动作并更新Deep Q network的相关参数
        :return :action, reward 如："u", -1
        """
        #action, reward = "u", -1.0

        # -----------------请实现你的算法代码--------------------------------------
                
        state = np.array(self.sense_state()) 
        # ε-greedy策略 
        valid_actions = self.current_state_valid_actions() 
        if np.random.rand()  < self.epsilon: 
            action = np.random.choice(valid_actions) 
        else:
            with torch.no_grad(): 
                state_tensor = torch.FloatTensor(state)
                q_values = self.policy_net(state_tensor) 
                mask = [0 if a in valid_actions else -np.inf  for a in self.actions] 
                masked_q = q_values + torch.tensor(mask) 
                action_idx = torch.argmax(masked_q).item() 
                action = self.actions[action_idx] 

        # 执行动作获取奖励
         
        reward = self.maze.move_robot(action)
        
        self.pos_arr=(self.pos_arr+[self.maze.sense_robot()])[-5:]
        if len(self.pos_arr)==5:
            p0,p1,p2,p3,p4=enumerate(self.pos_arr)
            # 容差比较关键点是否相等
            if (p0[1]==p2[1] and p2[1]==p4[1] and p1[1]==p3[1]):
                #print(p0,p1,p2,p3,p4)
                reward = -3.0
        if self.prev_reward==-3.0 and reward !=self.prev_reward:
                reward = 1.0
        
        self.prev_reward=reward
        next_state = np.array(self.sense_state()) 
        done = (self.maze.sense_robot() == self.maze.destination)

        # 存储经验 
        self.memory.append((state,  self.action_map[action],  reward, next_state, done))

        # 经验回放
        if len(self.memory)  >= self.batch_size: 
            self._replay_experience()

        # 衰减探索率 
        self.epsilon  = max(self.epsilon_min, self.epsilon  * self.epsilon_decay) 
        # -----------------------------------------------------------------------
        #print("train",action,reward)
        if(reward==50):print("manmanmanamanaanamnamnam")
        return action, reward

    def _replay_experience(self):
        batch = np.random.choice(len(self.memory),  self.batch_size,  replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in batch])

        states = torch.FloatTensor(np.array(states)) 
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states)) 
        dones = torch.BoolTensor(dones)

        # 计算目标Q值 
        with torch.no_grad(): 
            target_q = self.target_net(next_states).max(1)[0] 
            target_q[dones] = 0.0
            target = rewards + self.gamma  * target_q

        # 计算当前Q值
        current_q = self.policy_net(states).gather(1,  actions).squeeze()

        # 优化网络 
        loss = nn.MSELoss()(current_q, target)
        self.optimizer.zero_grad() 
        loss.backward() 
        
        # 梯度裁剪防止爆炸 
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(),  1.0)
        
        self.optimizer.step() 

        # 周期性更新目标网络 
        self.update_counter  += 1 
        if self.update_counter  % self.target_update_freq  == 0:
            print("what can i say")
            self.target_net.load_state_dict(self.policy_net.state_dict()) 
    
    def test_update(self):
        """
        以测试状态选择动作并更新Deep Q network的相关参数
        :return : action, reward 如："u", -1
        """
        #action, reward = "u", -1.0

        # -----------------请实现你的算法代码--------------------------------------

        state = np.array(self.sense_state()) 
    
        with torch.no_grad(): 
            state_tensor = torch.FloatTensor(state)
            q_values = self.policy_net(state_tensor) 

            # 过滤非法动作
            valid_mask = [0 if a in self.current_state_valid_actions()  else -np.inf  
                         for a in self.actions] 
            masked_q = q_values + torch.tensor(valid_mask) 
            action_idx = torch.argmax(masked_q).item() 
            action = self.actions[action_idx] 

        reward = self.maze.move_robot(action) 
        # -----------------------------------------------------------------------
        print("test",action,reward)
        return action, reward
