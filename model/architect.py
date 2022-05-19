# Differentiable Architecture Search
# Code accompanying the paper
# > [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055)\
# > Hanxiao Liu, Karen Simonyan, Yiming Yang.\
# > _arXiv:1806.09055_.

import torch
import numpy as np
from torch.autograd import Variable
from config.configs import Ellen_Config


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, optimizer):
        self.network_momentum = Ellen_Config['nas']['momentum']
        self.network_weight_decay = Ellen_Config['nas']['structure_optim_weight_decay']
        self.model = model
        self.optimizer = optimizer

    def _compute_unrolled_model(self, input, target, lr, network_optimizer):
        loss = self.model.get_loss(input, target)
        net_weight = _concat(self.model.net_param).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.net_param).mul_(
                self.network_momentum)
        except:
            moment = torch.zeros_like(net_weight)
        dw = _concat(torch.autograd.grad(loss, self.model.net_param)).data + self.network_weight_decay * net_weight
        net_weight = net_weight.sub(other=moment + dw, alpha=lr)
        unrolled_model = self._construct_model_from_theta(_concat([self.model.structure_param, net_weight]))
        return unrolled_model

    def step(self, input_train, target_train, input_valid, target_valid, lr, network_optimizer):
        self.optimizer.zero_grad()
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, lr, network_optimizer)
        self.optimizer.step()

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, lr, network_optimizer):
        unrolled_model = self._compute_unrolled_model(input_train, target_train, lr, network_optimizer)
        unrolled_loss = unrolled_model.get_loss(input_valid, target_valid)

        unrolled_loss.backward()
        dalpha = [v.grad.data for v in set([unrolled_model.structure_param])]
        vector = [v.grad.data for v in unrolled_model.net_param]
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(other=ig.data, alpha=lr)

        for v, g in zip(set([self.model.structure_param]), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        epsilon = r / _concat(vector).norm()
        for p, v in zip(self.model.net_param, vector):
            p.data.add_(other=v, alpha=epsilon)
        loss = self.model.get_loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model.structure_param)

        for p, v in zip(self.model.net_param, vector):
            p.data.sub_(other=v, alpha=2 * epsilon)
        loss = self.model.get_loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model.structure_param)

        for p, v in zip(self.model.net_param, vector):
            p.data.add_(other=v, alpha=epsilon)

        return [(x - y).div_(2 * epsilon) for x, y in zip(grads_p, grads_n)]
