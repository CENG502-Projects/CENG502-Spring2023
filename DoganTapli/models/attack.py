import torch.nn as nn
import torch

from torch.autograd import Variable

# DAP Attack Network
# Input: Victim Agent's actions and state
# Output: Lure action and switch 
class AttackNetwork(nn.Module):
    def __init__(self, inp_channel, num_actions):

        super().__init__()
        self.conv1 = nn.Conv2d(inp_channel, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.lin1 = nn.LazyLinear(512)
        #self.lin2 = nn.Linear(512, num_actions)

        self.lin3 = nn.Linear(num_actions, 512) # !!
        self.lin4 = nn.Linear(512, 256) # !!
        self.lstm = nn.LSTM(input_size=768, hidden_size=128, num_layers=256) # ... 256 cells ? A3C paper

        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()

        self.lure_head = nn.LazyLinear(num_actions)
        self.switch_head = nn.LazyLinear(1)
        
    def init_state(self):
        return (Variable(torch.zeros(256, 128)),
                Variable(torch.zeros(256, 128)))


    def forward(self, state, victim_policy, hidden):
        x1 = self.conv1(state)
        x1 = self.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.relu(x1)
        x1 = self.conv3(x1)
        x1 = self.relu(x1)
        x1 = self.flatten(x1)
        x1 = self.lin1(x1)
        x1 = self.relu(x1)
        #x1 = self.lin2(x1)
        #x1 = self.relu(x1)

        x2 = self.lin3(victim_policy)
        x2 = self.relu(x2)
        x2 = self.lin4(x2)
        x2 = self.relu(x2)

        x = torch.cat((x1, x2), dim=1)

        x, hidden = self.lstm(x, hidden)
        # relu?

        # softmax?
        action = torch.softmax(self.lure_head(x), dim=1)
        switch = torch.softmax(self.switch_head(x), dim=1)

        return action, switch, hidden
