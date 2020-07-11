import torch
from torch import nn
from torch.nn import functional as F


class Expert(nn.Module):
    def __init__(self):
        super(Expert, self).__init__()

        self.input_layer = nn.Linear(32 + 62, 256) # z is of size 32
        self.layer_1 = nn.Linear(32 + 256, 256)
        self.layer_2 = nn.Linear(32 + 256, 256)
        self.layer_3 = nn.Linear(32 + 256, 62)

    def forward(self, z, prev_pose):
        inp = torch.cat((z, prev_pose), 0)

        x = torch.cat((z, F.elu(self.input_layer(inp))), 0)
        x = torch.cat((z, F.elu(self.layer_1(x))), 0)
        x = torch.cat((z, F.elu(self.layer_2(x))), 0)

        return self.layer_3(x)


class GatingNetwork(nn.Module):
    def __init__(self):
        super(GatingNetwork, self).__init__()

        self.input_layer = nn.Linear(32 + 62, 256)
        self.layer_1 = nn.Linear(256, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, 6)

    def forward(self, z, prev_pose):
        inp = torch.cat((z, prev_pose), 0)

        x = F.elu(self.input_layer(inp))
        x = F.elu(self.layer_1(x))
        x = F.elu(self.layer_2(x))

        return self.layer_3(x)


class MVAE(nn.Module):
    def __init__(self):
        super(MVAE, self).__init__()

        # one frame/pose contains 62 pieces of data; 2 poses are input in one pass
        self.input_layer = nn.Linear(62 + 62, 256)
        self.encoder_layer_1 = nn.Linear(256, 256)
        self.encoder_layer_2 = nn.Linear(256, 256)
        self.encoder_layer_3_mean = nn.Linear(256, 32)
        self.encoder_layer_3_std = nn.Linear(256, 32)

        self.gating_network = GatingNetwork()

        self.expert_1 = Expert()
        self.expert_2 = Expert()
        self.expert_3 = Expert()
        self.expert_4 = Expert()
        self.expert_5 = Expert()
        self.expert_6 = Expert()

    def encode(self, prev_pose, curr_pose):
        inp = torch.cat((prev_pose, curr_pose), 0)
        x = F.elu(self.input_layer(inp))
        x = F.elu(self.encoder_layer_1(x))
        x = F.elu(self.encoder_layer_2(x))
        return self.encoder_layer_3_mean(x), self.encoder_layer_3_std(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, prev_pose):
        gating_output = self.gating_network(z, prev_pose)

        expert_1_output = self.expert_1(z, prev_pose)
        expert_2_output = self.expert_2(z, prev_pose)
        expert_3_output = self.expert_3(z, prev_pose)
        expert_4_output = self.expert_4(z, prev_pose)
        expert_5_output = self.expert_5(z, prev_pose)
        expert_6_output = self.expert_6(z, prev_pose)

        return  expert_1_output * gating_output[0] + expert_2_output * gating_output[1] + \
                expert_3_output * gating_output[2] + expert_4_output * gating_output[3] + \
                expert_5_output * gating_output[4] + expert_6_output * gating_output[5]

    def forward(self, prev_pose, curr_pose):
        mu, logvar = self.encode(prev_pose, curr_pose)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, prev_pose), mu, logvar


# thing = GatingNetwork()
# print(thing(torch.tensor([0.5]), torch.rand(28)))

# thing = Expert()
# print(thing(torch.tensor([0.5]), torch.rand(28)))

thing = MVAE()
thing(torch.rand(28), torch.rand(28))
