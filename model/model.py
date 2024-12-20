import torch
from torch import nn, Tensor
from model.modules import ModalFuseModule


class FMOE(nn.Module):
    def __init__(self,
                 model_id: str,
                 available_type: str,
                 v_dim: int,
                 t_dim: int,
                 a_dim: int,
                 h_dim: int):

        super(FMOE, self).__init__()
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
        self.predict_layer_t = nn.Sequential(
            nn.Linear(h_dim, 200),
            nn.Linear(200, 1)
        )
        self.predict_layer_v = nn.Sequential(
            nn.Linear(h_dim, 200),
            nn.Linear(200, 1)
        )
        self.predict_layer_a = nn.Sequential(
            nn.Linear(h_dim, 200),
            nn.Linear(200, 1)
        )

    def forward(self, v_f_seq: Tensor, t_f: Tensor, a_f: Tensor,first_stage):
        if first_stage:
            T_f_star, V_f_star, A_f_star = self.modal_fuse_module(t_f, v_f_seq, a_f,first_stage=first_stage)
            out_t=self.predict_layer_t(T_f_star)
            out_v=self.predict_layer_v(V_f_star)
            out_a=self.predict_layer_a(A_f_star)
            return  out_t, out_v, out_a
        else:
            available_tensor, loss_kl, loss_u = self.modal_fuse_module(t_f, v_f_seq, a_f,first_stage=first_stage)
            output = self.predict_layer(available_tensor)
            output = output.squeeze(-1)
            return  output, loss_kl, loss_u
            

    def compute_loss(self, v_f_seq: Tensor, t_f: Tensor, a_f: Tensor, label: Tensor,loss_ratio: dict,first_stage):
        label = label.unsqueeze(-1)
        
        if first_stage:
            out_t, out_v, out_a = self.forward(v_f_seq=v_f_seq, t_f=t_f, a_f=a_f,first_stage=first_stage)
            loss_t=nn.MSELoss()(out_t, label)
            loss_v=nn.MSELoss()(out_v, label)
            loss_a=nn.MSELoss()(out_a, label)
            return loss_t, loss_v, loss_a
        else:
            output, loss_kl, loss_u = self.forward(v_f_seq=v_f_seq, t_f=t_f, a_f=a_f,first_stage=first_stage)
            loss = nn.MSELoss()(output, label)
            loss = loss + loss_ratio['kl'] * loss_kl + loss_ratio['u'] * loss_u
        return loss
