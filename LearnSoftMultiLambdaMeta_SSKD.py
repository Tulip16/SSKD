import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

import faulthandler


class LearnSoftMultiLambdaMeta(object):
    """
    Implementation of Data Selection Strategy class which serves as base class for other
    dataselectionstrategies for general learning frameworks.
    Parameters
        ----------
        trainloader: class
            Loading the training data using pytorch dataloader
        model: class
            Model architecture used for training
        num_classes: int
            Number of target classes in the dataset
    """

    def __init__(self, trainloader, valloader, model, num_classes, N_trn, loss, device, fit, \
                 teacher_model, criterion_red, temp):
        """
        Constructer method
        """
        self.trainloader = trainloader  # assume its a sequential loader.
        self.valloader = valloader
        self.model = model
        self.N_trn = N_trn

        self.num_classes = num_classes
        self.device = device

        self.fit = fit
        # self.batch = batch
        # self.dist_batch = dist

        self.teacher_model = teacher_model
        self.criterion = loss
        self.criterion_red = criterion_red #nn.CrossEntropyLoss(reduction='sum')#
        self.temp = temp

        # setting the following to default values given in the SSKD paper 
        self.temp_T = 4
        self.temp_SS = 0.5
        self.ratio_ss = 0.75
        self.ratio_tf = 1
        self.num_teachers = 1
        print(N_trn)

    def update_model(self, model_params):
        """
        Update the models parameters

        Parameters
        ----------
        model_params: OrderedDict
            Python dictionary object containing models parameters
        """
        self.model.load_state_dict(model_params)

    
    def get_lambdas(self, eta,lam, lam_ss, lam_t):

        offset = 0
        batch_wise_indices = list(self.trainloader.batch_sampler)
        #eta =0.1

        lambdas = lam.cuda(0)#, device=self.device)
        lambdas_ss = lam_ss.cuda(0)#, device=self.device)
        lambdas_t = lam_t.cuda(0)#, device=self.device)
        soft_lam = F.softmax(lambdas, dim=1)

        with torch.no_grad():

            for batch_idx, (inputs, targets, _) in enumerate(self.valloader):
                # print(inputs.size())
                inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                # print(inputs.size())
                inputs = inputs[:,0,:,:,:].cuda()
                if batch_idx == 0:
                    out, l1, _, _ = self.model(inputs)
                    self.init_out = out
                    self.init_l1 = l1
                    self.y_val = targets  # .view(-1, 1)
                    tea_out_val, _, _, _ = self.teacher_model(inputs)
                else:
                    out, l1, _, _ = self.model(inputs)
                    self.init_out = torch.cat((self.init_out, out), dim=0)
                    self.init_l1 = torch.cat((self.init_l1, l1), dim=0)
                    self.y_val = torch.cat((self.y_val, targets), dim=0)
                    tea_out_val_temp, _, _, _ = self.teacher_model(inputs)
                    tea_out_val = torch.cat((tea_out_val, tea_out_val_temp), dim=0)

            # val_loss_SL = self.criterion_red(self.init_out,self.y_val)

            #val_loss_SL = torch.sum(self.criterion(self.init_out, self.y_val))

            # print(val_loss_SL)

            '''val_loss_KD = nn.KLDivLoss(reduction='none')(F.log_softmax(self.init_out / self.temp, dim=1),\
                F.softmax(tea_out_val / self.temp, dim=1))
            val_loss_KD = self.temp*self.temp*torch.sum (val_loss_KD)#torch.mean(torch.sum (val_loss_KD,dim=1))'''

            # val_loss_KD = self.temp*self.temp*nn.KLDivLoss(reduction='batchmean')(F.log_softmax(self.init_out / self.temp, dim=1),\
            #    F.softmax(tea_out_val / self.temp, dim=1))

            self.init_out = self.init_out.cuda(0)
            self.init_l1 = self.init_l1.cuda(0)
            self.y_val = self.y_val.cuda(0)
            tea_out_val = tea_out_val.cuda(0)

        KD_grads = [0]*self.num_teachers
        grad_t = [0]*self.num_teachers
        grad_ss = [0]*self.num_teachers

        c_temp = self.temp
        for batch_idx, (inputs, target,indices) in enumerate(self.trainloader):

            #batch_wise_indices = list(self.trainloader.batch_sampler)
            # print("input for SL grads before tranformation", inputs.size())
            inputs, target = inputs.to(self.device), target.to(self.device, non_blocking=True)
            # inputs = inputs[:,0,:,:,:].cuda()
            c,h,w = inputs.size()[-3:]
            inputs = inputs.view(-1,c,h,w).cuda()
            inputs = inputs[::4,:,:,:]
            # print("input for SL grads after tranformation", inputs.size())

            outputs, l1, i_, _ = self.model(inputs)
            # custom_target=torch.cat((target,target),dim=0)
            # custom_target=torch.cat((custom_target,custom_target),dim=0)
            # print("outputs sz", outputs.size())
            
            # for i in range(targets.size()[0]):
            #     custom_target[i][target[i]]=1
            #     custom_target[i+64][target[i]]=1
            #     custom_target[i+128][target[i]]=1
            #     custom_target[i+196][target[i]]=1




            # print(outputs.size())
            #print(outputs)
            #print(custom_target)
            # print(custom_target.size())
            #print(i_.size())
            #print(_.size())

            
            loss_SL = self.criterion_red(outputs, target)  # self.criterion(outputs, target).sum()

            l0_grads = (torch.autograd.grad(loss_SL, outputs)[0]).detach().clone().cuda(0)
            l0_expand = torch.repeat_interleave(l0_grads, l1.shape[1], dim=1)
            l1_grads = l0_expand * l1.repeat(1, self.num_classes).cuda(0)

            if batch_idx % self.fit == 0:
                with torch.no_grad():
                    train_out = outputs.cuda(0)
                    train_l1 = l1.cuda(0)
                    train_target = target.cuda(0)
                SL_grads = torch.cat((l0_grads, l1_grads), dim=1)
                batch_ind = list(indices) #batch_wise_indices[batch_idx]
            else:
                with torch.no_grad():
                    train_out = torch.cat((train_out,outputs.cuda(0)), dim=0)
                    train_l1 = torch.cat((train_l1,l1.cuda(0)), dim=0)
                    train_target = torch.cat((train_target,target.cuda(0)), dim=0)
                SL_grads = torch.cat((SL_grads, torch.cat((l0_grads, l1_grads), dim=1)), dim=0)
                batch_ind.extend(list(indices))#batch_wise_indices[batch_idx])

            for m in range(self.num_teachers):
                with torch.no_grad():
                    teacher_outputs, _, _, _ = self.teacher_model(inputs)
                loss_KD = self.temp * self.temp * nn.KLDivLoss(reduction='batchmean')(
                    F.log_softmax(outputs / self.temp, dim=1), \
                    F.softmax(teacher_outputs / self.temp, dim=1))

                l0_grads = (torch.autograd.grad(loss_KD, outputs)[0]).detach().clone().cuda(0)
                l0_expand = torch.repeat_interleave(l0_grads, l1.shape[1], dim=1)
                l1_grads = l0_expand * l1.repeat(1, self.num_classes).cuda(0)

                if batch_idx % self.fit == 0:
                    KD_grads[m] = torch.cat((l0_grads, l1_grads), dim=1)
                else:
                    KD_grads[m] = torch.cat((KD_grads[m], torch.cat((l0_grads, l1_grads), dim=1)), dim=0)

            ''' T and SS components of the loss '''
            for m in range(self.num_teachers):
                c,h,w = inputs.size()[-3:]
                x = inputs.view(-1,c,h,w).cuda()
                # _, num_transformations, _, _, _ = inputs.size()
                # x = inputs[:,1:num_trasformations,:,:,:]

                batch = int(x.size(0) / 4)
                nor_index = (torch.arange(4*batch) % 4 == 0).cuda()
                aug_index = (torch.arange(4*batch) % 4 != 0).cuda()

                output, l1, s_feat, _ = self.model(x, bb_grad=True)
                # log_nor_output = F.log_softmax(output[nor_index] / args.kd_T, dim=1)
                log_aug_output = F.log_softmax(output[aug_index] / self.temp_T, dim=1)
                with torch.no_grad():
                    knowledge, _, t_feat, _ = self.teacher_model(x)
                    # nor_knowledge = F.softmax(knowledge[nor_index] / args.kd_T, dim=1)
                    aug_knowledge = F.softmax(knowledge[aug_index] / self.temp_T, dim=1)
          
                special_target = target[::4] # might be target[:target.size()[0]/4]
                aug_target = special_target.unsqueeze(1).expand(-1,3).contiguous().view(-1).long().cuda()
                rank = torch.argsort(aug_knowledge, dim=1, descending=True)
                rank = torch.argmax(torch.eq(rank, aug_target.unsqueeze(1)).long(), dim=1)  # groundtruth label's rank
                index = torch.argsort(rank)
                tmp = torch.nonzero(rank, as_tuple=True)[0]
                wrong_num = tmp.numel()
                correct_num = 3*batch - wrong_num
                wrong_keep = int(wrong_num * self.ratio_tf)
                index = index[:correct_num+wrong_keep]
                distill_index_tf = torch.sort(index)[0]

                s_nor_feat = s_feat[nor_index]
                s_aug_feat = s_feat[aug_index]
                s_nor_feat = s_nor_feat.unsqueeze(2).expand(-1,-1,3*batch).transpose(0,2)
                s_aug_feat = s_aug_feat.unsqueeze(2).expand(-1,-1,1*batch)
                s_simi = F.cosine_similarity(s_aug_feat, s_nor_feat, dim=1)

                t_nor_feat = t_feat[nor_index]
                t_aug_feat = t_feat[aug_index]
                t_nor_feat = t_nor_feat.unsqueeze(2).expand(-1,-1,3*batch).transpose(0,2)
                t_aug_feat = t_aug_feat.unsqueeze(2).expand(-1,-1,1*batch)
                t_simi = F.cosine_similarity(t_aug_feat, t_nor_feat, dim=1)
                t_simi = t_simi.detach()
                aug_target = torch.arange(batch).unsqueeze(1).expand(-1,3).contiguous().view(-1).long().cuda()
                rank = torch.argsort(t_simi, dim=1, descending=True)
                rank = torch.argmax(torch.eq(rank, aug_target.unsqueeze(1)).long(), dim=1)  # groundtruth label's rank
                index = torch.argsort(rank)
                tmp = torch.nonzero(rank, as_tuple=True)[0]
                wrong_num = tmp.numel()
                correct_num = 3*batch - wrong_num
                wrong_keep = int(wrong_num * self.ratio_ss)
                index = index[:correct_num+wrong_keep]
                distill_index_ss = torch.sort(index)[0]

                log_simi = F.log_softmax(s_simi / self.temp_SS, dim=1)
                simi_knowledge = F.softmax(t_simi / self.temp_SS, dim=1)

                loss_T = self.temp_T * self.temp_T * nn.KLDivLoss(reduction='batchmean')(
                    log_aug_output[distill_index_tf], \
                    aug_knowledge[distill_index_tf])
                
                loss_SS = self.temp_SS * self.temp_SS * nn.KLDivLoss(reduction='batchmean')(
                    log_simi[distill_index_ss], \
                    simi_knowledge[distill_index_ss])

                l0_grads = (torch.autograd.grad(loss_T, output)[0]).detach().clone().cuda(0)
                l0_expand = torch.repeat_interleave(l0_grads, l1.shape[1], dim=1)
                l1_grads = l0_expand * l1.repeat(1, self.num_classes).cuda(0)

                if batch_idx % self.fit == 0:
                    grad_t[m] = torch.cat((l0_grads, l1_grads), dim=1)
                else:
                    grad_t[m] = torch.cat((grad_t[m], torch.cat((l0_grads, l1_grads), dim=1)), dim=0)
                

                l0_grads = (torch.autograd.grad(loss_SS, s_feat,allow_unused=True)[0]).detach().clone().cuda(0)
                l0_expand = l0_grads.repeat(1, self.num_classes).cuda(0)

                if batch_idx % self.fit == 0:
                    grad_ss[m] = l0_expand
                else:
                    grad_ss[m] = torch.cat((grad_ss[m], l0_expand), dim=0)
                
            
            if (batch_idx + 1) % self.fit == 0 or batch_idx + 1 == len(self.trainloader):

                for r in range(10):
                    #print("Before",lambdas[batch_ind[0]].item(),lambdas[batch_ind[-1]].item())
                    print("soft lam", soft_lam[batch_ind,0][:,None].size())
                    print("SL_grads", SL_grads.size())
                    comb_grad = soft_lam[batch_ind,0][:,None]*SL_grads 
                    #comb_grad = lambdas[batch_ind,0][:,None]*SL_grads 
                    
                    for m in range(self.num_teachers):
                        comb_grad += soft_lam[batch_ind,m+1][:,None]*KD_grads[m]
                        #comb_grad += lambdas[batch_ind,m+1][:,None]*KD_grads[m]

                    comb_grad = comb_grad.sum(0)

                    out_vec_val = self.init_out - (eta * comb_grad[:self.num_classes].view(1, -1).\
                        expand(self.init_out.shape[0], -1))

                    out_vec_val = out_vec_val - (eta * torch.matmul(self.init_l1, comb_grad[self.num_classes:].\
                        view(self.num_classes, -1).transpose(0, 1)))

                    #out_vec_val.requires_grad = True
                    '''loss_SL_val = self.criterion_red(out_vec_val, self.y_val)  # self.criterion(outputs, target).sum()

                    l0_grads = (torch.autograd.grad(loss_SL_val, out_vec_val)[0]).detach().clone().cuda(1)'''

                    loss_KD_val = c_temp * c_temp *nn.KLDivLoss(reduction='batchmean')(F.log_softmax(\
                    out_vec_val/c_temp , dim=1), F.softmax(tea_out_val/c_temp, dim=1))

                    l0_grads = (torch.autograd.grad(loss_KD_val, out_vec_val)[0]).detach().clone().cuda(0)
                    #print(round(loss_KD_val.item(),4), end=",")

                    #print(round(loss_SL_val.item(),4), end=",")
                    l0_expand = torch.repeat_interleave(l0_grads, self.init_l1.shape[1], dim=1)
                    l1_grads = l0_expand * self.init_l1.repeat(1, self.num_classes).cuda(0)
                    up_grads_val = torch.cat((l0_grads, l1_grads), dim=1).sum(0)
                    up_grads_val_ss = self.init_l1.repeat(1, self.num_classes).cuda(0).sum(0)

                    out_vec = train_out - (eta * comb_grad[:self.num_classes].view(1, -1).expand(train_out.shape[0], -1))

                    out_vec = out_vec - (eta * torch.matmul(train_l1, comb_grad[self.num_classes:].\
                        view(self.num_classes, -1).transpose(0, 1)))

                    # out_vec.requires_grad = True

                    loss_SL = self.criterion_red(out_vec, train_target)  # self.criterion(outputs, target).sum()

                    #print(round(loss_SL_val.item(),4),"+",round(loss_SL.item(),4), end=",")
                    print(round(loss_KD_val.item(),4),"+",round(loss_SL.item(),4), end=",")

                    l0_grads = (torch.autograd.grad(loss_SL, out_vec)[0]).detach().clone().cuda(0)
                    l0_expand = torch.repeat_interleave(l0_grads, train_l1.shape[1], dim=1)
                    l1_grads = l0_expand * train_l1.repeat(1, self.num_classes).cuda(0)
                    up_grads = torch.cat((l0_grads, l1_grads), dim=1).sum(0)
                    up_grads_ss = train_l1.repeat(1, self.num_classes).cuda(0).sum(0)

                    combined = (0.75*up_grads_val+0.25*up_grads).T
                    combined_ss = (0.75*up_grads_val_ss+0.25*up_grads_ss).T

                    grad = ((1-soft_lam[batch_ind,0])*soft_lam[batch_ind,0])[:,None]*SL_grads
                    #grad = SL_grads 
                    for m_1 in range(self.num_teachers):
                        grad -= (soft_lam[batch_ind,0]*soft_lam[batch_ind,m_1+1])[:,None]*KD_grads[m_1]
                        #grad -= KD_grads[m_1]
                    alpha_grads = torch.matmul(grad,combined)
                    lambdas[batch_ind,0] = lambdas[batch_ind,0] +  500*eta*alpha_grads #9*eta*
                    
                    for m in range(self.num_teachers):
                        grad = (-soft_lam[batch_ind,0]*soft_lam[batch_ind,m+1])[:,None]*SL_grads 
                        #grad = -SL_grads 
                        for m_1 in range(self.num_teachers):
                            if m_1 == m:
                                grad += ((1-soft_lam[batch_ind,m_1+1])*soft_lam[batch_ind,m_1+1])[:,None]*KD_grads[m_1]
                                #grad += KD_grads[m_1]
                            else:
                                grad -= (soft_lam[batch_ind,m+1]*soft_lam[batch_ind,m_1+1])[:,None]*KD_grads[m_1]
                                #grad -= KD_grads[m_1]
                        grad_ss = torch.zeros(len(grad_ss)) + grad_ss
                        grad_t = torch.tensor(len(grad_t)) + grad_t
                        print("_____________________________")
                        print(len(lambdas))
                        print(len(lambdas[0]))
                        print(len(lambdas_ss))
                        print(len(lambdas_ss[0]))
                        print(len(lambdas_t))
                        print(len(lambdas_t[0]))
                        alpha_grads = torch.matmul(grad,combined)
                        alpha_grads_ss = torch.matmul(grad_ss, combined_ss)
                        alpha_grads_t = torch.matmul(grad_t, combined)
                        lambdas[batch_ind,m+1] = lambdas[batch_ind,m+1] +  500*eta*alpha_grads #9*eta*
                        lambdas_ss[batch_ind,m] = lambdas_ss[batch_ind,m] +  500*eta*alpha_grads_ss #9*eta*
                        lambdas_t[batch_ind,m] = lambdas_t[batch_ind,m] +  500*eta*alpha_grads_t #9*eta*
                    #print("After",lambdas[batch_ind[0]].item(),lambdas[batch_ind[-1]].item())
                    #lambdas.clamp_(min=1e-7,max=1-1e-7)
                    soft_lam[batch_ind] = F.softmax(lambdas[batch_ind], dim=1)
                print()#"End for loop")

        #lambdas.clamp_(min=0.01,max=0.99)
        return lambdas.cuda(0), lambdas_ss.cuda(0), lambdas_t.cuda(0)
