""" Architect controls architecture of cell by computing gradients of alphas """
import copy

import torch
import torch.nn as nn


class Architect:
    """ Compute gradients of alphas """
    def __init__(self, args, criterion, teacher, student):
        self.args = copy.deepcopy(args)
        self.criterion = criterion
        self.teacher = teacher.cuda()
        self.student = student.cuda()
        self.v_teacher = copy.deepcopy(teacher)
        self.v_student = copy.deepcopy(student)
        self.w_momentum = self.args.w_momentum
        self.w_weight_decay = self.args.w_weight_decay

    def critere(self, pred, true, data_count, reduction='mean'):
        weights = self.teacher.arch[data_count:data_count + pred.shape[0]]
        weights = torch.softmax(weights, dim=0) ** 0.5
        if reduction != 'mean':
            crit = nn.MSELoss(reduction=reduction)
            return crit(pred * weights, true * weights).mean(dim=(-1, -2))
        else:
            return self.criterion(pred * weights, true * weights)

    def virtual_step(self, trn_data, next_data, xi, w_optim_teacher, w_optim_student, data_count):
        # forward & calc loss
        pred = self.teacher(trn_data[0])
        unreduced_loss = self.critere(pred, trn_data[1], data_count, reduction='none')
        gradients = torch.autograd.grad(unreduced_loss.mean(), self.teacher.W(), retain_graph=True)
        with torch.no_grad():
            for w, vw, g in zip(self.teacher.W(), self.v_teacher.W(), gradients):
                m = w_optim_teacher.state[w].get('momentum_buffer', 0.) * self.w_momentum
                vw.copy_(w - xi * (m + g + self.w_weight_decay * w))
                w.grad = g
            for a, va in zip(self.teacher.A(), self.v_teacher.A()):
                va.copy_(a)

        pred_teacher = self.teacher(next_data[0])
        pred = self.student(next_data[0])
        unreduced_loss_s = self.critere(pred, torch.cat([trn_data[1][:, :self.args.dec_seq_len, :],
                                        pred_teacher[:, -self.args.pred_len:, :]], dim=1), data_count, reduction='none')
        gradients = torch.autograd.grad(unreduced_loss.mean(), self.student.W())
        with torch.no_grad():
            for w, vw, g in zip(self.student.W(), self.v_student.W(), gradients):
                m = w_optim_student.optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                vw.copy_(w - xi * (m + g + self.w_weight_decay * w))
                w.grad = g
            for a, va in zip(self.student.A(), self.v_student.A()):
                va.copy_(a)
        return unreduced_loss, unreduced_loss_s

    def unrolled_backward(self, args_in, trn_data, val_data, next_data, xi, w_optim_teacher, w_optim_student, data_count):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # init config
        args = args_in
        # do virtual step (calc w`)
        unreduced_loss, unreduced_loss_s = self.virtual_step(trn_data, next_data, xi, w_optim_teacher, w_optim_student, data_count)
        if torch.isfinite(unreduced_loss).float().min() < 1:
            print('DAMGER 007')
        hessian = torch.zeros(args.batch_size, args.pred_len, trn_data[1].shape[-1]).cuda()

        # calc unrolled loss
        pred = self.v_student(val_data[0])
        loss = self.criterion(pred, val_data[:, -self.args.pred_len, :])
        # compute gradient
        v_W = list(self.v_student.W())
        dw = list(torch.autograd.grad(loss, v_W))
        for d in dw:
            if torch.isfinite(d).float().min() < 1:
                print('DAMGER 008')
        hessian = self.compute_hessian(dw, trn_data, data_count)

        # hessian clip
        max_norm = float(args.max_hessian_grad_norm)
        hessian_clip = copy.deepcopy(hessian)
        for n, (h_c, h) in enumerate(zip(hessian_clip, hessian)):
            h_norm = torch.norm(h.detach(), dim=-1)
            max_coeff = h_norm / max_norm
            max_coeff[max_coeff < 1.0] = torch.tensor(1.0).cuda(args.gpu)
            hessian_clip[n] = torch.div(h, max_coeff.unsqueeze(-1))
        hessian = hessian_clip
        if torch.isfinite(hessian).float().min() < 1:
            print('DAMGER 009')

        dw_list = []
        for i in range(self.args.batch_size):
            dw_list.append(torch.autograd.grad(unreduced_loss[i], self.teacher.W(), retain_graph=i != self.args.batch_size - 1))

        da = torch.zeros_like(self.teacher.arch).cuda()
        pred = self.v_teacher(trn_data[0])
        pseudo_loss = (pred * hessian).sum()
        if torch.isfinite(pseudo_loss).float().min() < 1:
            print('DAMGER 010')
        dw0 = torch.autograd.grad(pseudo_loss, self.v_teacher.W())
        for d in dw0:
            if torch.isfinite(d).float().min() < 1:
                print('DAMGER 011')
        for i in range(self.args.batch_size):
            for a, b in zip(dw_list[i], dw0):
                da[data_count+i] += (a*b).sum()

        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():
            self.teacher.arch.grad = da * xi * xi
        return unreduced_loss.mean()

    def compute_hessian(self, dw, trn_data, data_count):
        """
        dw = dw` { L_val(alpha, w`, h`) }, dh = dh` { L_val(alpha, w`, h`) }
        w+ = w + eps_w * dw, h+ = h + eps_h * dh
        w- = w - eps_w * dw, h- = h - eps_h * dh
        hessian_w = (dalpha { L_trn(alpha, w+, h) } - dalpha { L_trn(alpha, w-, h) }) / (2*eps_w)
        hessian_h = (dalpha { L_trn(alpha, w, h+) } - dalpha { L_trn(alpha, w, h-) }) / (2*eps_h)
        eps_w = 0.01 / ||dw||, eps_h = 0.01  ||dh||
        """
        norm_w = torch.cat([w.view(-1) for w in dw]).norm()
        eps_w = 0.01 / norm_w
        trn_data[1].requires_grad = True

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.teacher.W(), dw):
                p += eps_w * d
        pred = self.student(trn_data[0])
        loss = self.critere(pred, trn_data[1], data_count)
        dE_pos = torch.autograd.grad(loss, [trn_data[1]])[0][:, -self.args.pred_len:, :]

        with torch.no_grad():
            for p, d in zip(self.teacher.W(), dw):
                p -= 2. * eps_w * d
        pred = self.student(trn_data[0])
        loss = self.critere(pred, trn_data[1], data_count)
        dE_neg = torch.autograd.grad(loss, [trn_data[1]])[0][:, -self.args.pred_len:, :]

        # recover w
        with torch.no_grad():
            for p, d in zip(self.teacher.W(), dw):
                p += eps_w * d

        hessian = (dE_pos - dE_neg) / (2. * eps_w)
        trn_data[1].requires_grad = False
        return hessian

