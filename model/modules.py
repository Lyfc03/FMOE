import torch
from torch import nn, Tensor

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=2,
            attn_drop=0.0,
            proj_drop=0.0,
            mlp_ratio=1.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.scale = head_dim ** -0.5
        self.q, self.k, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop_a, self.proj_drop_t, self.proj_drop_v = nn.Dropout(proj_drop), nn.Dropout(proj_drop), nn.Dropout(proj_drop)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=proj_drop,
        )


    def forward(self, x):
        B, seq_len, C = x.shape

        q = self.q(x).reshape(B, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q.float() @ k.float().transpose(-2, -1))  # [B, heads, s, s]
        x_out = (attn @ v).transpose(1, 2).reshape(B, seq_len, C)
        x_out = x_out + self.mlp(x_out)
        return x_out


class FeedForward(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.mu = nn.Linear(input_dim,output_dim)
        self.logvar = nn.Linear(input_dim,output_dim)

    def reparameterise(self,mu, std):
        """
        mu : [batch_size,z_dim]
        std : [batch_size,z_dim]        
        """        
        # get epsilon from standard normal
        eps = torch.randn_like(std)
        return mu + std*eps

    def KL_loss(self,mu,logvar):
        return (-(1+logvar-mu.pow(2)-logvar.exp())/2).sum(dim=1).mean()

    def forward(self,x):
        mu=self.mu(x)
        logvar=self.logvar(x)
        z = self.reparameterise(mu,logvar)
        kl_loss = self.KL_loss(mu,logvar)
        return kl_loss ,torch.exp(logvar)
    
    
class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ModalFuseModule(nn.Module):
    """
    This class is used to fuse the textual and visual features.

    Args:
        t_dim: The dimension of the textual feature.
        v_dim: The dimension of the visual feature.

    Inputs:
        t_f: The textual feature.
        v_f: The visual feature.

    Outputs:
        fused_features: The fused features.
    """

    def __init__(self, t_dim: Tensor, v_dim: Tensor, a_dim: Tensor, h_dim: Tensor, available_type):
        super(ModalFuseModule, self).__init__()
        self.tahn = nn.Tanh()
        self.v_dim = v_dim
        self.t_dim = t_dim
        self.a_dim = a_dim
        self.h_dim = h_dim
        self.drop_rate = 0
        self.mlp_ratio=2
        self.num_heads=4
        self.attn_drop_rate=0.1
        self.available_type = available_type
        self.visual_embedding = nn.Linear(v_dim, v_dim)
        self.textual_embedding = nn.Linear(t_dim, t_dim)
        self.dual_attention_linear_1 = nn.Linear(v_dim * 2, v_dim)
        self.dual_attention_linear_2 = nn.Linear(t_dim * 2, t_dim)
        self.cross_modal_linear_1 = nn.Linear(t_dim * 2, t_dim)
        self.cross_modal_linear_2 = nn.Linear(t_dim * 2, t_dim)
        self.uni_modal_linear_1 = nn.Linear(h_dim, 1)
        self.uni_modal_linear_2 = nn.Linear(h_dim, 1)
        self.uni_modal_linear_3 = nn.Linear(h_dim, 1)
        self.gelu = nn.GELU()
        
        self.t_in_proj = nn.Sequential(nn.Linear(t_dim, h_dim))
        self.v_in_proj = nn.Sequential(nn.Linear(v_dim, h_dim))
        self.a_in_proj = nn.Sequential(nn.Linear(a_dim, h_dim))
        
        self.router_t = Mlp(
            in_features=h_dim,
            hidden_features=int(h_dim * self.mlp_ratio),
            out_features=3,
            drop=self.drop_rate
        )
        self.router_v = Mlp(
            in_features=h_dim,
            hidden_features=int(h_dim * self.mlp_ratio),
            out_features=3,
            drop=self.drop_rate
        )
        self.router_a = Mlp(
            in_features=h_dim,
            hidden_features=int(h_dim * self.mlp_ratio),
            out_features=3,
            drop=self.drop_rate
        )
        
        self.Transformer_a = Attention(
            dim=h_dim,
            num_heads=self.num_heads,
            attn_drop=self.attn_drop_rate,
            proj_drop=self.drop_rate,
            mlp_ratio=self.mlp_ratio,
        )
        self.Transformer_t = Attention(
            dim=h_dim,
            num_heads=self.num_heads,
            attn_drop=self.attn_drop_rate,
            proj_drop=self.drop_rate,
            mlp_ratio=self.mlp_ratio,
        )
        self.Transformer_v = Attention(
            dim=h_dim,
            num_heads=self.num_heads,
            attn_drop=self.attn_drop_rate,
            proj_drop=self.drop_rate,
            mlp_ratio=self.mlp_ratio,
        )

        self.ff_t = FeedForward(h_dim,h_dim)
        self.ff_v = FeedForward(h_dim,h_dim)
        self.ff_a = FeedForward(h_dim,h_dim)
    def uni_modal_attention(self,
                            v_f: Tensor,
                            t_f: Tensor,
                            a_f: Tensor):
        alpha_v = torch.softmax(self.uni_modal_linear_1(v_f) / self.h_dim, dim=1)
        V_f_star = torch.matmul(alpha_v.transpose(1, 2), v_f)
        alpha_t = torch.softmax(self.uni_modal_linear_2(t_f) / self.h_dim, dim=1)
        T_f_star = torch.matmul(alpha_t.transpose(1, 2), t_f)
        alpha_a = torch.softmax(self.uni_modal_linear_3(a_f) / self.h_dim, dim=1)
        A_f_star = torch.matmul(alpha_a.transpose(1, 2), a_f)
        return V_f_star, T_f_star, A_f_star
    
    def joint(self, x):
        x_cross_t, x_cross_v, x_cross_a = torch.clone(x), torch.clone(x), torch.clone(x)
        
        x_cross_t = x_cross_t + self.Transformer_t(x_cross_t)
        x_cross_v = x_cross_v + self.Transformer_v(x_cross_v)
        x_cross_a = x_cross_a + self.Transformer_a(x_cross_a)
        
        kl_t, sigma_t = self.ff_t(x_cross_t)
        kl_v, sigma_v = self.ff_v(x_cross_v)
        kl_a, sigma_a = self.ff_a(x_cross_a)
        loss_kl = (kl_t + kl_v + kl_a)/3
        joint=torch.cat([x_cross_t, x_cross_v, x_cross_a], dim=-1)
        sigma=torch.cat([sigma_t, sigma_v, sigma_a], dim=-1)
        return joint, sigma, loss_kl

    def forward(self, t_f: Tensor, v_f: Tensor, a_f: Tensor,first_stage=True):
        B, seq_len, _= v_f.shape
        
        x_t = self.t_in_proj(t_f)
        x_v = self.v_in_proj(v_f)
        x_a = self.a_in_proj(a_f)
        
        
        if first_stage:
            x_out_t = x_t + self.Transformer_t(x_t)
            x_out_v = x_v + self.Transformer_v(x_v)
            x_out_a = x_a + self.Transformer_a(x_a)
        else:
            weight_t, weight_v, weight_a= self.router_t(x_t), self.router_v(x_v), self.router_a(x_a)
            weight_t = torch.softmax(weight_t, dim=-1)
            weight_v = torch.softmax(weight_v, dim=-1)
            weight_a = torch.softmax(weight_a, dim=-1)
            
            x_t, sigma_t, kl_t = self.joint(x_t)
            x_v, sigma_v, kl_v = self.joint(x_v)
            x_a, sigma_a, kl_a = self.joint(x_a)
            loss_kl = (kl_t + kl_v + kl_a)/3
            
            weight_a = weight_a.unsqueeze(-1).repeat(1, 1, 1, self.h_dim)
            weight_t = weight_t.unsqueeze(-1).repeat(1, 1, 1, self.h_dim)
            weight_v = weight_v.unsqueeze(-1).repeat(1, 1, 1, self.h_dim)
            
            x_unweighted_a = x_a.reshape(B, seq_len, 3, self.h_dim)
            x_unweighted_t = x_t.reshape(B, seq_len, 3, self.h_dim)
            x_unweighted_v = x_v.reshape(B, seq_len, 3, self.h_dim)
            
            x_out_a = torch.sum(weight_a * x_unweighted_a, dim=2)
            x_out_t = torch.sum(weight_t * x_unweighted_t, dim=2)
            x_out_v = torch.sum(weight_v * x_unweighted_v, dim=2)
            
            Uncertainty_a = sigma_a.reshape(B, seq_len, 3, self.h_dim)
            Uncertainty_t = sigma_t.reshape(B, seq_len, 3, self.h_dim)
            Uncertainty_v = sigma_v.reshape(B, seq_len, 3, self.h_dim)
            
            loss_u_a = torch.sum(weight_a * Uncertainty_a, dim=2).mean()
            loss_u_t = torch.sum(weight_t * Uncertainty_t, dim=2).mean()
            loss_u_v = torch.sum(weight_v * Uncertainty_v, dim=2).mean()
            
            loss_u = loss_u_t + loss_u_v + loss_u_a
            loss_u = loss_u/3
            
        V_f_star, T_f_star, A_f_star = self.uni_modal_attention(x_out_v, x_out_t, x_out_a)
        if first_stage:
            return T_f_star, V_f_star, A_f_star
        else:
            available_types = list(self.available_type)
            type_tensors = {
                'T': T_f_star,
                'V': V_f_star,
                'A': A_f_star,
            }
            tensors_to_cat = [type_tensors[t] for t in available_types]
            available_tensor = torch.cat(tensors_to_cat, dim=-1)
            return available_tensor,loss_kl, loss_u
 
