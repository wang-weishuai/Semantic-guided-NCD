import sys
import torch.nn.functional as F
import torch

# Based on: https://github.com/xu-ji/IIC/blob/master/code/utils/cluster/IID_losses.py

def mutual_information_loss(x_out, y_out, lamb=1.0, EPS=sys.float_info.epsilon):
  # has had softmax applied
  _, k = x_out.size()
  _, m = y_out.size()
  p_i_j = compute_joint(x_out, y_out)
  assert (p_i_j.size() == (k, m))

  p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, m).clone()
  p_j = p_i_j.sum(dim=0).view(1, m).expand(k, m).clone()

  # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
  p_i_j[(p_i_j < EPS).data] = EPS
  p_j[(p_j < EPS).data] = EPS
  p_i[(p_i < EPS).data] = EPS

  loss = - p_i_j * (torch.log(p_i_j) \
                    - lamb * torch.log(p_j) \
                    - lamb * torch.log(p_i))

  loss = loss.sum()

  loss_no_lamb = - p_i_j * (torch.log(p_i_j) \
                            - torch.log(p_j) \
                            - torch.log(p_i))

  loss_no_lamb = loss_no_lamb.sum()

  return loss, loss_no_lamb


def compute_joint(x_out, y_out):
  # produces variable that requires grad (since args require grad)

  bn, k = x_out.size()
  assert (y_out.size(0) == bn)

  p_i_j = x_out.unsqueeze(2) * y_out.unsqueeze(1)  # bn, k, m
  p_i_j = p_i_j.sum(dim=0)  # k, m
  p_i_j = p_i_j / p_i_j.sum()  # normalise

  return p_i_j
