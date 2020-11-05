class Actor(nn.Module):
  #action_dim：一个时间点上往往进行多个动作，以完成运动，ex跑步需要同时动手动脚
  #max_action：把action添加噪声后，clip合并到一个范围
  def __init__(self, state_dim, action_dim, max_action):
    #激活继承
    super(Actor, self).__init__()
    #网络结构见paper
    self.layer_1 = nn.Linear(state_dim, 400)
    self.layer_2 = nn.Linear(400, 300)
    self.layer_3 = nn.Linear(300, action_dim)
    self.max_action = max_action

  def forward(self, x):
    x = F.relu(self.layer_1(x))
    x = F.relu(self.layer_2(x))
    #因为tanh返回值在-1到1，通过self.max_action让值回到连续动作值域的范围
    x = self.max_action * torch.tanh(self.layer_3(x))
    return x
    
class Critic(nn.Module):
  #由于只返回一个Q值，所以不需要max_action
  def __init__(self, state_dim, action_dim):
    super(Critic, self).__init__()
    # Defining the first Critic neural network
    self.layer_1 = nn.Linear(state_dim + action_dim, 400)
    self.layer_2 = nn.Linear(400, 300)
    self.layer_3 = nn.Linear(300, 1)
    # Defining the second Critic neural network
    self.layer_4 = nn.Linear(state_dim + action_dim, 400)
    self.layer_5 = nn.Linear(400, 300)
    self.layer_6 = nn.Linear(300, 1)

  def forward(self, x, u):
    #合并state x和action u
    #1：在vertical方向进行连接；0：水平方向
    xu = torch.cat([x, u], 1)
    # Forward-Propagation on the first Critic Neural Network
    x1 = F.relu(self.layer_1(xu))
    x1 = F.relu(self.layer_2(x1))
    x1 = self.layer_3(x1)
    # Forward-Propagation on the second Critic Neural Network
    x2 = F.relu(self.layer_4(xu))
    x2 = F.relu(self.layer_5(x2))
    x2 = self.layer_6(x2)
    return x1, x2
  #额外的函数
  #代表第一个critic model输出的Q值，仅用它来gradient ascent，来更新actor model
  def Q1(self, x, u):
    xu = torch.cat([x, u], 1)
    x1 = F.relu(self.layer_1(xu))
    x1 = F.relu(self.layer_2(x1))
    x1 = self.layer_3(x1)
    return x1
