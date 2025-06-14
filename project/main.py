import torch
import torch.nn  as nn
import numpy as np 
from collections import deque
import matplotlib.pyplot as plt
class DQN(nn.Module):
    """ 深度Q网络结构设计 """
    '''
    def __init__(self, input_dim, output_dim):
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
        #print("state.shape: ",state.shape)
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
        #self.pos_arr = []
        #self.prev_reward = 0
        # 关键超参数优化 
        self.gamma  = 0.85     # 折扣因子 
        self.epsilon  = 1.0    # 初始探索率
        self.epsilon_min  = 0.1  # 最小探索率 
        self.epsilon_decay  = 0.85  # 衰减率
        self.batch_size  = 8   # 批次大小 
        self.memory  = deque(maxlen=200)  # 经验回放池 
        
        # 网络架构 
        self.policy_net  = DQN(self.state_dim,  len(self.actions)) 
        self.target_net  = DQN(self.state_dim,  len(self.actions)) 
        self.target_net.load_state_dict(self.policy_net.state_dict()) 
        self.optimizer  = torch.optim.Adam(self.policy_net.parameters(),  lr=0.02)  # 降低学习率 
        
        # 训练状态跟踪 
        #self.previous_pos  = None  # 记录前一个位置 
        self.update_counter  = 0   # 网络更新计数器 
        self.target_update_freq  = 5  # 目标网络更新频率 
        

    def train_update(self):
        """
        以训练状态选择动作并更新Deep Q network的相关参数
        :return :action, reward 如："u", -1
        """
        #action, reward = "u", -1.0

        # -----------------请实现你的算法代码--------------------------------------
                
        state = np.array(self.sense_state()) 
        print("state:",state)
        valid_actions = self.current_state_valid_actions() 
        if np.random.rand()  < self.epsilon: 
            action = np.random.choice(valid_actions) 
            print("random move")
        else:
            with torch.no_grad(): 
                state_tensor = torch.FloatTensor(state)#.unsqueeze(0)
                q_values = self.policy_net(state_tensor) 
                mask = [0 if a in valid_actions else -np.inf  for a in self.actions] 
                masked_q = q_values + torch.tensor(mask) 
                action_idx = torch.argmax(masked_q).item() 
                action = self.actions[action_idx] 
                print("masked_q:",[round(x, 2) for x in masked_q.squeeze().tolist()]," action_idx:",action_idx," action:",action)

        # 执行动作获取奖励
        reward = self.maze.move_robot(action)
        #self.prev_reward=reward
        next_state = np.array(self.sense_state()) 
        done = (self.maze.sense_robot() == self.maze.destination)

        # 存储经验 
        self.memory.append((state,  self.action_map[action],  reward, next_state, done))

        # 经验回放
        if len(self.memory)  >= self.batch_size: 
            self._replay_experience()
        else:
            print(f"Current memory size: {len(self.memory)} / {self.batch_size}")

        # 衰减探索率 
        self.epsilon  = max(self.epsilon_min, self.epsilon  * self.epsilon_decay) 
        # -----------------------------------------------------------------------
       
        #print("train",action,reward)
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
            #print("what can i say")
            self.target_net.load_state_dict(self.policy_net.state_dict()) 
        return current_q,target_q
    def test_update(self):
        """
        以测试状态选择动作并更新Deep Q network的相关参数
        :return : action, reward 如："u", -1
        """
        #action, reward = "u", -1.0

        # -----------------请实现你的算法代码--------------------------------------

        state = np.array(self.sense_state()) 
    
        with torch.no_grad(): 
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor) 

            # 过滤非法动作
            valid_mask = [0 if a in self.current_state_valid_actions()  else -np.inf  
                         for a in self.actions] 
            masked_q = q_values + torch.tensor(valid_mask) 
            action_idx = torch.argmax(masked_q).item() 
            action = self.actions[action_idx] 

        reward = self.maze.move_robot(action) 
        # -----------------------------------------------------------------------
        #print("test",action,reward)
        return action, reward
