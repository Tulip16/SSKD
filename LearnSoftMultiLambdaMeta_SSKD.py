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
        #print(N_trn)

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
        counter=0
        offset = 0
        batch_wise_indices = list(self.trainloader.batch_sampler)
        #eta =0.1

        lambdas = lam.cuda(0)#, device=self.device)
        lambdas_ss = lam_ss.cuda(0)#, device=self.device)
        lambdas_t = lam_t.cuda(0)#, device=self.device)
        #soft_lam = F.softmax(lambdas, dim=1)
        #soft_lam_ss = F.softmax(lambdas_ss, dim=1)
        #soft_lam_t = F.softmax(lambdas_t, dim=1)

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

        torch.cuda.empty_cache()
        KD_grads = [0]*self.num_teachers
        grad_t = [0]*self.num_teachers
        grad_ss = [0]*self.num_teachers

        c_temp = self.temp
        for batch_idx, (inputs, target,indices) in enumerate(self.trainloader):

            #batch_wise_indices = list(self.trainloader.batch_sampler)
            # print("input for SL grads before tranformation", inputs.size())
            if torch.cuda.is_available():
                inputs, target = inputs.to(self.device), target.to(self.device, non_blocking=True)
                # inputs = inputs[:,0,:,:,:].cuda()
                c,h,w = inputs.size()[-3:]
                inputs = inputs.view(-1,c,h,w).cuda()
                #inputs = inputs[::4,:,:,:] 
                # print("input for SL grads after tranformation", inputs.size())
                batch = int(inputs.size(0) / 4)
                nor_index = (torch.arange(4*batch) % 4 == 0)
                aug_index = (torch.arange(4*batch) % 4 != 0)

                outputs, l1, s_feat, _ = self.model(inputs)
                # custom_target=torch.cat((target,target),dim=0)
                # custom_target=torch.cat((custom_target,custom_target),dim=0)
                # print("outputs sz", outputs.size())

                # for i in range(targets.size()[0]):
                #     custom_target[i][target[i]]=1
                #     custom_target[i+64][target[i]]=1
                #     custom_target[i+128][target[i]]=1
                #     custom_target[i+196][target[i]]=1




                #print("output size")
                #print(outputs.size())
                #print(outputs)
                #print(custom_target)
                # print(custom_target.size())
                #print(i_.size())
                #print(_.size())


                loss_SL = self.criterion_red(outputs[::4, :], target)  # self.criterion(outputs, target).sum()

                l0_grads = (torch.autograd.grad(loss_SL, outputs)[0]).detach().clone().cuda(0)
                l0_grads = l0_grads[::4,:]
                l0_expand = torch.repeat_interleave(l0_grads, l1[::4,:].shape[1], dim=1)
                l1_grads = l0_expand * l1[::4,:].repeat(1, self.num_classes).cuda(0)
                torch.cuda.empty_cache()

                if batch_idx % self.fit == 0:
                    with torch.no_grad():
                        train_out = outputs[::4,:].cuda(0)
                        train_l1 = l1[::4,:].cuda(0)
                        train_target = target.cuda(0)
                    SL_grads = torch.cat((l0_grads, l1_grads), dim=1)
                    batch_ind = list(indices) #batch_wise_indices[batch_idx]

                else:
                    with torch.no_grad():
                        train_out = torch.cat((train_out,outputs[::4,:].cuda(0)), dim=0)
                        train_l1 = torch.cat((train_l1,l1[::4,:].cuda(0)), dim=0)
                        train_target = torch.cat((train_target,target.cuda(0)), dim=0)
                    SL_grads = torch.cat((SL_grads, torch.cat((l0_grads, l1_grads), dim=1)), dim=0)
                    batch_ind.extend(list(indices))#batch_wise_indices[batch_idx])

                for m in range(self.num_teachers):
                    with torch.no_grad():
                        knowledge, _, t_feat, _ = self.teacher_model(inputs)
                        # nor_knowledge = F.softmax(knowledge[nor_index] / args.kd_T, dim=1)
                        aug_knowledge = F.softmax(knowledge[aug_index] / self.temp_T, dim=1)
                        
                    loss_KD = self.temp * self.temp * nn.KLDivLoss(reduction='batchmean')(
                        F.log_softmax(outputs[::4,:] / self.temp, dim=1), \
                        F.softmax(knowledge[nor_index] / self.temp, dim=1))

                    l0_grads = (torch.autograd.grad(loss_KD, outputs)[0]).detach().clone().cuda(0)
                    l0_grads = l0_grads[::4,:]
                    l0_expand = torch.repeat_interleave(l0_grads, l1[::4,:].shape[1], dim=1)
                    l1_grads = l0_expand * l1[::4,:].repeat(1, self.num_classes).cuda(0)
                    torch.cuda.empty_cache()

                    if batch_idx % self.fit == 0:
                        KD_grads[m] = torch.cat((l0_grads, l1_grads), dim=1)
                    else:
                        KD_grads[m] = torch.cat((KD_grads[m], torch.cat((l0_grads, l1_grads), dim=1)), dim=0)

                    ''' T and SS components of the loss '''
                    #c,h,w = inputs.size()[-3:]
                    #x = inputs.view(-1,c,h,w).cuda()
                    # _, num_transformations, _, _, _ = inputs.size()
                    # x = inputs[:,1:num_trasformations,:,:,:]

                    #batch = int(x.size(0) / 4)
                    #nor_index = (torch.arange(4*batch) % 4 == 0).cuda()
                    #aug_index = (torch.arange(4*batch) % 4 != 0).cuda()

                    #output, l1, s_feat, _ = self.model(x, bb_grad=True)
                    # log_nor_output = F.log_softmax(outputs[nor_index] / args.kd_T, dim=1)
                    log_aug_output = F.log_softmax(outputs[aug_index] / self.temp_T, dim=1)
                    #with torch.no_grad():
                        #knowledge, _, t_feat, _ = self.teacher_model(x)
                        # nor_knowledge = F.softmax(knowledge[nor_index] / args.kd_T, dim=1)
                        #aug_knowledge = F.softmax(knowledge[aug_index] / self.temp_T, dim=1)

                    special_target = target # might be target[:target.size()[0]/4]
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

                    l0_grads = (torch.autograd.grad(loss_T, outputs)[0]).detach().clone().cuda(0)
                    l0_expand = torch.repeat_interleave(l0_grads, l1.shape[1], dim=1)
                    l1_grads = l0_expand * l1.repeat(1, self.num_classes).cuda(0)
                    torch.cuda.empty_cache()

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
                    #print("new")
                    for r in range(5):
                        #print("Before",lambdas[batch_ind[0]].item(),lambdas[batch_ind[-1]].item())
                        #print("soft lam", soft_lam[batch_ind,0][:,None].size())
                        #print("SL_grads", SL_grads.size())
                        comb_grad = lambdas[batch_ind,0][:,None]*SL_grads 

                        for m in range(self.num_teachers):
                            comb_grad += lambdas[batch_ind,m+1][:,None]*KD_grads[m]
                            #comb_grad += lambdas[batch_ind,m+1][:,None]*KD_grads[m]
                        #for m in range(self.num_teachers - 1):
                            #comb_grad_t += soft_lam_t[batch_ind,m][:,None]*grad_t[m]
                            #comb_grad_ss += soft_lam_ss[batch_ind,m][:,None]*grad_ss[m]


                        comb_grad = comb_grad.sum(0)
                        #comb_grad_t = comb_grad_t.sum(0)
                        #comb_grad_ss = comb_grad_ss.sum(0)

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
                        torch.cuda.empty_cache()
                        #print(torch.cuda.memory_summary(device=None, abbreviated=False))
                        #print(torch.cuda.memory_stats(device=None))
                        if counter <= 10:
                            print(torch.cuda.memory_allocated(device=None))
                        counter += 1
                        up_grads_val = torch.cat((l0_grads, l1_grads), dim=1).sum(0)
                        up_grads_val_ss = self.init_l1.repeat(1, self.num_classes).cuda(0).sum(0)

                        out_vec = train_out - (eta * comb_grad[:self.num_classes].view(1, -1).expand(train_out.shape[0], -1))

                        out_vec = out_vec - (eta * torch.matmul(train_l1, comb_grad[self.num_classes:].\
                            view(self.num_classes, -1).transpose(0, 1)))
                        del comb_grad
                        torch.cuda.empty_cache()

                        #out_vec.requires_grad = True

                        loss_SL = self.criterion_red(out_vec, train_target)  # self.criterion(outputs, target).sum()

                        #print(round(loss_SL_val.item(),4),"+",round(loss_SL.item(),4), end=",")
                        # print(round(loss_KD_val.item(),4),"+",round(loss_SL.item(),4), end=",")

                        l0_grads = (torch.autograd.grad(loss_SL, out_vec)[0]).detach().clone().cuda(0)
                        l0_expand = torch.repeat_interleave(l0_grads, train_l1.shape[1], dim=1)
                        l1_grads = l0_expand * train_l1.repeat(1, self.num_classes).cuda(0)
                        torch.cuda.empty_cache()
                        up_grads = torch.cat((l0_grads, l1_grads), dim=1).sum(0)
                        up_grads_ss = train_l1.repeat(1, self.num_classes).cuda(0).sum(0)

                        combined = (0.75*up_grads_val+0.25*up_grads).T
                        combined_ss = (0.75*up_grads_val_ss+0.25*up_grads_ss).T
                        del out_vec_val
                        del out_vec
                        del up_grads_val
                        del up_grads
                        del up_grads_ss
                        del up_grads_val_ss
                        torch.cuda.empty_cache()
                       
                        one_index = (torch.arange(4*batch*self.fit) % 4 == 1)
                        two_index = (torch.arange(4*batch*self.fit) % 4 == 2)
                        three_index = (torch.arange(4*batch*self.fit) % 4 == 3)
                        grad_SS = (grad_ss[0][one_index]+grad_ss[0][two_index]+grad_ss[0][three_index])/3
                        grad_T = (grad_t[0][one_index]+grad_t[0][two_index]+grad_t[0][three_index])/3
                        del one_index
                        del two_index
                        del three_index
                        torch.cuda.empty_cache()

                        alpha_grads_ss = torch.matmul(grad_SS, combined_ss)
                        alpha_grads_t = torch.matmul(grad_T, combined)
                        lambdas_ss[batch_ind,0] = lambdas_ss[batch_ind,0] +  100*eta*alpha_grads_ss #9*eta*
                        lambdas_t[batch_ind,0] = lambdas_t[batch_ind,0] +  100*eta*alpha_grads_t #9*eta*

                        for m in range(self.num_teachers):
                            grad = KD_grads[m] - SL_grads
                           
                            alpha_grads = torch.matmul(grad,combined)
                            lambdas[batch_ind,m] = lambdas[batch_ind,m] +  100*eta*alpha_grads #9*eta*

                        #print("After",lambdas[batch_ind[0]].item(),lambdas[batch_ind[-1]].item())
                        lambdas.clamp_(min=1e-7,max=1-1e-7)
                        lambdas[batch_ind,0] = 1- torch.max(lambdas[batch_ind,1:],dim=1).values
                        
                        del alpha_grads_ss
                        del alpha_grads_t
                        del alpha_grads
                        del grad
                        del grad_SS
                        del grad_T
                        del l0_grads
                        del l1_grads
                        del l0_expand
                        torch.cuda.empty_cache()
                    #print()#"End for loop")
                    try:
                        del alpha_grads_ss
                        del alpha_grads_t
                        del alpha_grads
                        del grad
                        del grad_SS
                        del grad_T
                        del loss_SL
                        del loss_SS
                        del loss_KD
                        del loss_T
                        del outputs
                        del l0_grads
                        del l1_grads
                        del l0_expand
                        torch.cuda.empty_cache()
                    except:
                        pass
            try:
                del alpha_grads_ss
                del alpha_grads_t
                del alpha_grads
                del grad
                del grad_SS
                del grad_T
                del loss_SL
                del loss_SS
                del loss_KD
                del loss_T
                del outputs
                del l0_grads
                del l1_grads
                del l0_expand
                del alpha_grads_ss
                torch.cuda.empty_cache()
            except:
                pass
        try:
            del alpha_grads_t
            del alpha_grads
            del grad
            del grad_SS
            del grad_T
            del loss_SL
            del loss_SS
            del loss_KD
            del loss_T
            del outputs
            del l0_grads
            del l1_grads
            del l0_expand
            torch.cuda.empty_cache()
        except:
            pass
                

        #lambdas.clamp_(min=0.01,max=0.99)
        return lambdas.cuda(0), lambdas_ss.cuda(0), lambdas_t.cuda(0)
