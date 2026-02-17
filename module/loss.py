import torch.nn.functional as F
from torch import nn
import torch
import numpy as np

class ContrastiveLoss(nn.Module):
    def __init__(self, init_temperature, alpha, beta, eeg_l2norm:bool, img_l2norm:bool, text_l2norm:bool, learnable:bool, is_softplus:bool, multi_positive_loss: bool = False):
        super(ContrastiveLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eeg_l2norm = eeg_l2norm
        self.img_l2norm = img_l2norm
        self.text_l2norm = text_l2norm
        self.multi_positive_loss = multi_positive_loss
        
        self.is_softplus = is_softplus
        
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_mse = nn.MSELoss()
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / init_temperature), requires_grad=learnable)
        self.softplus = nn.Softplus()

    def _multi_positive_ce(self, sim_matrix: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        # sim_matrix: (N, N); group_ids: (N,) with same id => multi-positive
        logp = F.log_softmax(sim_matrix, dim=1)
        mask = (group_ids[:, None] == group_ids[None, :]).to(logp.dtype)
        mask = mask / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return -(mask * logp).sum(dim=1).mean()

    def forward(self, eeg_feature, image_feature, text_feature, group_ids: torch.Tensor | None = None):
        # L2 normalize embeddings
        if self.eeg_l2norm:
            eeg_feature = F.normalize(eeg_feature, p=2, dim=1)
        if self.img_l2norm:
            image_feature = F.normalize(image_feature, p=2, dim=1)
        if self.beta != 1.0:
            if self.text_l2norm:
                text_feature = F.normalize(text_feature, p=2, dim=1)

        # Calculate similarity matrix (N x N)
        if self.is_softplus:
            logit_scale = self.softplus(self.logit_scale)
        else:
            logit_scale = torch.exp(self.logit_scale)
        similarity_matrix_ie = torch.matmul(eeg_feature, image_feature.T) * logit_scale
        if self.beta != 1.0:
            similarity_matrix_te = torch.matmul(eeg_feature, text_feature.T) * logit_scale

        # Calculate two parts of the loss
        if self.multi_positive_loss:
            if group_ids is None:
                raise ValueError("multi_positive_loss=True requires group_ids (e.g., object_idx*100 + image_idx).")
            group_ids = group_ids.to(eeg_feature.device).long()
            loss_eeg_ie = self._multi_positive_ce(similarity_matrix_ie, group_ids)
            loss_img_ie = self._multi_positive_ce(similarity_matrix_ie.T, group_ids)
        else:
            labels = torch.arange(eeg_feature.shape[0], device=eeg_feature.device)
            loss_eeg_ie = self.criterion_cls(similarity_matrix_ie, labels)
            loss_img_ie = self.criterion_cls(similarity_matrix_ie.T, labels)
        if self.beta != 1.0:
            if self.multi_positive_loss:
                loss_eeg_te = self._multi_positive_ce(similarity_matrix_te, group_ids)
                loss_img_te = self._multi_positive_ce(similarity_matrix_te.T, group_ids)
            else:
                loss_eeg_te = self.criterion_cls(similarity_matrix_te, labels)
                loss_img_te = self.criterion_cls(similarity_matrix_te.T, labels)
            
        if self.alpha != 1.0:
            loss_mse = self.criterion_mse(eeg_feature, image_feature)
        
        # Total loss is the average
        if self.beta != 1.0:
            loss_contrastive_ie = (loss_eeg_ie + loss_img_ie) / 2
            loss_contrastive_te = (loss_eeg_te + loss_img_te) / 2
            loss_contrastive = self.beta * loss_contrastive_ie + (1 - self.beta) * loss_contrastive_te
        else:
            loss_contrastive = (loss_eeg_ie + loss_img_ie) / 2
        
        if self.alpha != 1.0:
            loss = self.alpha * loss_contrastive + (1 - self.alpha) * loss_mse
        else:
            loss = loss_contrastive
        
        return loss