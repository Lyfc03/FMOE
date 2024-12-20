import torch
from torch import nn, Tensor
from model.modules import ModalFuseModule


class UMEPP(nn.Module):
    def __init__(self,
                 model_id: str,
                 available_type: str,
                 v_dim: int,
                 t_dim: int,
                 a_dim: int,
                 h_dim: int):

        super(UMEPP, self).__init__()
        self.model_id = model_id
        self.available_type = available_type
        self.modal_fuse_module = ModalFuseModule(t_dim=t_dim,
                                                v_dim=v_dim,
                                                a_dim=a_dim,
                                                h_dim=h_dim,
                                                available_type=available_type)
        self.predict_layer = nn.Sequential(
            nn.Linear(h_dim*len(available_type), 200),
            nn.Linear(200, 1)
        )

    def forward(self, v_f_seq: Tensor, t_f: Tensor, a_f: Tensor,training=True):
        available_tensor, loss_kl, loss_u = self.modal_fuse_module(t_f, v_f_seq, a_f,training=training)
        output = self.predict_layer(available_tensor)
        output = output.squeeze(-1)
        return  output, loss_kl, loss_u
            

    def compute_loss(self, v_f_seq: Tensor, t_f: Tensor, a_f: Tensor, label: Tensor,loss_ratio: dict,training=True):
        label = label.unsqueeze(-1)
        output, loss_kl, loss_u = self.forward(v_f_seq=v_f_seq, t_f=t_f, a_f=a_f,training=training)
        loss = nn.MSELoss()(output, label)
        loss = loss + loss_ratio['kl'] * loss_kl + loss_ratio['u'] * loss_u
        return loss
