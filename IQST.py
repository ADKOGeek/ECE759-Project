#########################################################
# IQ Signal Transformer Model
#########################################################

import torch
import torch.nn as nn
from itertools import repeat

#Patch splitting module
class PatchBlock(nn.Module):
    def __init__(self, num_patches):
        super().__init__()
        self.patch_size = int(512 / num_patches)
        self.patch = nn.Unfold(kernel_size=[2,self.patch_size], stride=self.patch_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.patch(x)
        x = torch.transpose(torch.transpose(x, dim0=0, dim1=2), dim0=1, dim1=2)

        return x

#Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        super().__init__()
        self.attn_block = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=False)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim,embed_dim)
        )
        
    def forward(self, x):
        normed = self.layer_norm(x)
        attended, weights = self.attn_block(normed, normed, normed) 
        x = attended + x
        x = self.ffn(x) + x #FFN block and residual connection

        return x

    
#task head for each estimated parameter
class TaskHead(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1,out_channels=embed_dim, kernel_size=3, padding=1, padding_mode='circular')
        self.lin_out = nn.Linear(embed_dim*embed_dim,1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.lin_out(x)

        return x
    
#IQ Signal Transformer Class
class IQST(nn.Module):
    def __init__(self, device):
        super().__init__()
        # TODO: Implement all elements of the full Vision Transform
        self.num_patches = 8 #default 8
        self.patch_size = int(2*512/self.num_patches)
        self.embed_dim = 768 #default 128
        self.hidden_dim = 768 #default 128
        self.num_heads = 8 #default 8
        self.num_layers = 3 #default 3
        self.dropout = 0.25 #default 0.25
        self.num_classes = 5
        self.device = device

        self.patch_block = PatchBlock(self.num_patches)

        #shared token for classification and regression
        self.class_token = nn.Parameter(torch.randn(1,1,self.embed_dim))

        #create random learnable position embeddings. This is different from the paper, which uses cos/sin embeddings
        self.position_embeddings = nn.Parameter(torch.randn(self.num_patches+1,1,self.embed_dim))

        #linear projector into embed dim
        self.linear_projector = nn.Linear(self.patch_size, self.embed_dim)

        #encoder and decoder blocks
        self.encoder = nn.ModuleList([
            EncoderBlock(embed_dim=self.embed_dim, hidden_dim=self.hidden_dim, num_heads=self.num_heads, dropout=self.dropout)
            for i in range(0,self.num_layers)])
        #self.decoder = DecoderBlock(embed_dim=self.embed_dim, hidden_dim=self.hidden_dim, num_heads=self.num_heads, dropout=self.dropout)

        #parameter-specific layers
        self.num_pulses = TaskHead(self.embed_dim, self.dropout)
        self.pulse_width = TaskHead(self.embed_dim, self.dropout)
        self.time_delay = TaskHead(self.embed_dim, self.dropout)
        self.repetition_interval = TaskHead(self.embed_dim, self.dropout)
        
        self.class_MLP = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim,self.num_classes),
            nn.Softmax(dim=1))


    def forward(self, x):
        #create patch embeddings
        patches = self.patch_block(x)
        embeddings = self.linear_projector(patches)

        #create class token
        cls_tokens = self.class_token.repeat(1,x.shape[0],1)
        embeddings = torch.concat((cls_tokens, embeddings), dim=0) #add class token to embeddings along #tokens axis (num_tokens x batch_size x embed_dim)

        #add positional embedding to input
        embeddings = torch.add(embeddings, self.position_embeddings)

        #put embeddings through encoder
        for i in range(0,self.num_layers):
            embeddings = self.encoder[i](embeddings)

        #perform classification with cls token
        shared_embed = torch.mean(embeddings, 0) #take only cls token out of embeddings
        p_type = self.class_MLP(shared_embed)

        #Estimate radar params
        #np = self.num_pulses(shared_embed) #should be batch x 1
        np = self.num_pulses(shared_embed)
        pw = self.pulse_width(shared_embed)
        td = self.time_delay(shared_embed)
        ri = self.repetition_interval(shared_embed)
        rad_params = torch.cat((np,pw,td,ri), dim=1) 

        #return class and estimates
        return p_type, rad_params



