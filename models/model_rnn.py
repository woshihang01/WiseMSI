import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms, datasets
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import torch.nn.functional as F


# # 首先建立一个简单的循环神经网络：输入维度为1024， 输出维度是512， 一层的双向网络
# basic_rnn = nn.RNN(input_size=1024, hidden_size=512, num_layers=2, bidirectional=False)
# """
# 网络会接收一个序列输入 x_{t} 和记忆输入 h_{0},x_{t} 的维度是 (seq, batch, feature)，分别表示序列长度、批量和输入的特征维度，
# h_{0}也叫隐藏状态，它的维度是 (layer*direction,batch,hidden) ，分别表示层数乘方向（如果单向就是1，双向就是2）、批量和输出的维度。
# 网络会输出 output和 h_{n} , output 表示网络实际的输出，维度是 (seq,batch,hidden*direction) ，分别表示序列长度、批量和输出维度上方向，
# h_{n} 表示记忆单元，维度是 (layer*direction,batch,hidden) ，分别表示层数乘方向、批量和输出维度。
# """
# print(basic_rnn.weight_ih_l0.size(), basic_rnn.bias_ih_l0.size(), basic_rnn.weight_hh_l0.size(), basic_rnn.bias_hh_l0.size())
#
# # 随机初始化输入和隐藏状态
# toy_input = torch.randn(100, 64, 1024)
# h_0 = torch.randn(2, 64, 512)
#
# # 将输入和隐藏状态传入网络，得到输出和更新之后的隐藏状态，输出维度是(100, 1, 50),隐藏维度为(2,1,50)
# toy_output, h_n = basic_rnn(toy_input, h_0)
# print(toy_output.size())
#
# # print(h_n)
# print(h_n.size())

# data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
#
# train_set = datasets.MNIST(train=True, transform=data_tf, download=True)
# test_set = datasets.MNIST(train=False, transform=data_tf, download=True)
# train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_set, batch_size=64, shuffle=False)


class rnn_classify(nn.Module):
    def __init__(self, input_size=1024, hidden_layer_size=256, num_class=2):
        super(rnn_classify, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_layer_size, batch_first=True)
        self.module = [
            nn.Linear(hidden_layer_size, hidden_layer_size//4),
            nn.ReLU(),
            nn.Linear(hidden_layer_size//4, num_class),
        ]
        self.module = nn.Sequential(*self.module)

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.to(torch.float32))
        logits = self.module(lstm_out[:, -1, :])
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict

    # def forward(self, feature, feature_len):
    #     order_idx = np.argsort(feature_len.numpy())[::-1]
    #     order_feature = feature[order_idx.tolist()]
    #     order_seq = feature_len[order_idx.tolist()]
    #     pack_data = pack_padded_sequence(order_feature, order_seq, batch_first=True)
    #     out, _ = self.rnn(pack_data.float())
    #     unpacked = pad_packed_sequence(out)
    #     out, bz = unpacked[0], unpacked[1]
    #     out = out[bz-1, list(range(out.shape[1])), :]
    #     out = self.dense_1(out)
    #     out = self.activate(out)
    #     logits = self.dense_2(out)
    #     Y_hat = torch.topk(logits, 1, dim=1)[1]
    #     Y_prob = F.softmax(logits, dim=1)
    #     results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
    #     return results_dict

#
# net = rnn_classify(28, 100, 10)
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adadelta(net.parameters(), 1e-1)
#
#
# def get_acc(output, label):
#     total = output.shape[0]
#     _, pred_label = output.max(1)
#     num_correct = (pred_label == label).sum().data
#     # print(num_correct, total)
#     return num_correct
#
#
# def train(net, train_loader, valid_loader, num_epochs, optimizer, criterion):
#     if torch.cuda.is_available():
#         net = net.cuda()
#     for i in range(num_epochs):
#         train_loss = 0
#         train_acc = 0
#         net = net.train()
#         for im, label in train_loader:
#             if torch.cuda.is_available():
#                 im = im.cuda()
#                 label = label.cuda()
#             output = net(im)
#             total = output.shape[0]
#             loss = loss_fn(output, label)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.data.cpu().numpy() / float(total)
#             train_acc += get_acc(output, label).cpu().numpy() / float(total)
#         if valid_loader is not None:
#             valid_loss = 0
#             valid_acc = 0
#             net = net.eval()
#             for im, label in valid_loader:
#                 if torch.cuda.is_available():
#                     im = im.cuda()
#                     label = label.cuda()
#                 output = net(im)
#                 total = output.shape[0]
#                 loss = loss_fn(output, label)
#                 valid_loss += loss.data.cpu().numpy() / float(total)
#                 valid_acc += get_acc(output, label).cpu().numpy() / float(total)
#             print("epoch: %d, train_loss: %f, train_acc: %f, valid_loss: %f, valid_acc:%f"
#                   % (i, train_loss / len(train_loader), train_acc / len(train_loader),
#                      valid_loss / len(valid_loader), valid_acc / len(valid_loader)))
#         else:
#             print("epoch= ", i, "train_loss= ", train_loss / len(train_loader), "train_acc= ",
#                   train_acc / len(train_loader))
#
#
# train(net, train_loader, test_loader, 10, optimizer, loss_fn)
