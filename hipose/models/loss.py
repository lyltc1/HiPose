import torch.nn as nn
import torch


#################### loss ######################
################################################
class BinaryCodeLoss(nn.Module):
    def __init__(self, binary_code_loss_type, mask_binary_code_loss, divided_number_each_iteration, use_histgramm_weighted_binary_loss=False):
        super().__init__()
        self.binary_code_loss_type = binary_code_loss_type
        self.mask_binary_code_loss = mask_binary_code_loss
        self.divided_number_each_iteration = divided_number_each_iteration
        self.use_histgramm_weighted_binary_loss = use_histgramm_weighted_binary_loss

        if self.use_histgramm_weighted_binary_loss: # this Hammloss will be used in both case, for loss, or for histogramm
            self.hamming_loss = HammingLoss()

        if binary_code_loss_type == "L1":
            self.loss = nn.L1Loss(reduction="mean")
        elif binary_code_loss_type == "BCE":
            self.loss = nn.BCEWithLogitsLoss(reduction="mean")
        elif binary_code_loss_type == "CE":
            self.loss = nn.CrossEntropyLoss(reduction="mean")
        else:
            raise NotImplementedError(f"unknown mask loss type: {binary_code_loss_type}")

        if self.use_histgramm_weighted_binary_loss:
            assert binary_code_loss_type == "BCE"  # currently only have the implementation with BCEWithLogitsLoss
            self.loss= BinaryLossWeighted(nn.BCEWithLogitsLoss(reduction='none'))

        self.histogram= None
        
    def forward(self, pred_binary_code, pred_mask, groundtruth_code):
        ## calculating hamming loss and bit error histogram for loss weighting
        if self.use_histgramm_weighted_binary_loss:
            loss_hamm, histogram_new = self.hamming_loss(pred_binary_code, groundtruth_code, pred_mask.clone().detach())
            if self.histogram is None:
                self.histogram  = histogram_new
            else:
                self.histogram = histogram_new*0.05+self.histogram*0.95
            
            ## soft bin weigt decrease 
            hist_soft = torch.minimum(self.histogram,0.51-self.histogram)
            bin_weights = torch.exp(hist_soft*3).clone()    

        if self.mask_binary_code_loss:
            pred_binary_code = pred_mask.clone().detach() * pred_binary_code

        if self.binary_code_loss_type == "L1":
            pred_binary_code = pred_binary_code.reshape(-1, 1, pred_binary_code.shape[2], pred_binary_code.shape[3])
            pred_binary_code = torch.sigmoid(pred_binary_code)
            groundtruth_code = groundtruth_code.view(-1, 1, groundtruth_code.shape[2], groundtruth_code.shape[3])
        elif self.binary_code_loss_type == "BCE" and not self.use_histgramm_weighted_binary_loss:
            pred_binary_code = pred_binary_code.reshape(-1, pred_binary_code.shape[2], pred_binary_code.shape[3])
            groundtruth_code = groundtruth_code.view(-1, groundtruth_code.shape[2], groundtruth_code.shape[3])
        elif self.binary_code_loss_type == "CE":
            pred_binary_code = pred_binary_code.reshape(-1, self.divided_number_each_iteration, pred_binary_code.shape[2], pred_binary_code.shape[3])
            groundtruth_code = groundtruth_code.view(-1, groundtruth_code.shape[2], groundtruth_code.shape[3])
            groundtruth_code = groundtruth_code.long()
        
        if self.use_histgramm_weighted_binary_loss:
            loss = self.loss(pred_binary_code, groundtruth_code, bin_weights)
        else:
            loss = self.loss(pred_binary_code, groundtruth_code)
    
        return loss


class BinaryLossWeighted(nn.Module):
    def __init__(self, baseloss):
        #the base loss should have reduction 'none' 
        super().__init__()
        self.base_loss = baseloss

    def forward(self,input,target,weight):
        base_output=self.base_loss(input,target)
        assert base_output.ndim == 4
        output = base_output.mean([0,2,3])
        output = torch.sum(output*weight)/torch.sum(weight)
        return output


class MaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss(reduction="mean")
       
    def forward(self, pred_mask, groundtruth_mask): 
        pred_mask = pred_mask[:, 0, :, :]
        pred_mask = torch.sigmoid(pred_mask)
        
        return self.loss(pred_mask, groundtruth_mask)


class HammingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,predicted_code_prob,GT_code,mask):
        assert predicted_code_prob.ndim ==4
        mask_hard  = mask.round().clamp(0,1) # still kept round and clamp for safety
        code1_hard = torch.sigmoid(predicted_code_prob).round().clamp(0,1)
        code2_hard = GT_code.round().clamp(0,1) # still kept round and clamp for safety
        hamm = torch.abs(code1_hard-code2_hard)*mask_hard
        histogram = hamm.sum([0,2,3])/(mask_hard.sum()+1)
        hamm_loss = histogram.mean()
        
        return hamm_loss,histogram.detach()
    

class ConsistentMaskLoss(nn.Module):
    """ for all point mask > 0.5 or image mask < 0.5, loss = L1(point_mask, mask) """
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss(reduction="mean")

    def forward(self, pred_point_mask_output, pred_img_visib_mask_output, consistency_map):
        pred_point_mask_output = pred_point_mask_output.squeeze(-1).squeeze(1)  # [bsz, 2730]
        gathered_pts_mask = torch.gather(pred_img_visib_mask_output.reshape(pred_point_mask_output.shape[0], -1), 1, consistency_map)  # [bsz, 2730]
        loss = self.loss(torch.sigmoid(pred_point_mask_output), torch.sigmoid(gathered_pts_mask))
        return loss

class ConsistentCodeLoss(nn.Module):
    """  point code should consistent with corresponding image code 
         point code means the code output according to 3D points,
         image code means the code output according to 2D points,
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss(reduction="mean")
    def forward(self, pred_point_code_output, pred_img_code_output, consistency_map):
        pred_point_code_output = pred_point_code_output.reshape(pred_point_code_output.shape[0], pred_point_code_output.shape[1], -1)
        gathered_pts_code = torch.gather(pred_img_code_output.reshape(pred_img_code_output.shape[0], pred_img_code_output.shape[1], -1), 2, consistency_map.unsqueeze(1).repeat(1,16,1))
        loss = self.loss(torch.sigmoid(pred_point_code_output), torch.sigmoid(gathered_pts_code))
        return loss