import numpy as np
import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pygnn
from performer_pytorch import SelfAttention
from torch_geometric.data import Batch
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import to_dense_batch
from torch_geometric.utils import degree

from graphgps.layer.bigbird_layer import SingleBigBirdLayer
from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvESLapPE

from torch_scatter import scatter_mean, scatter_std, scatter_add

@torch.no_grad()
def get_log_deg(batch):
    if "log_deg" in batch:
        log_deg = batch.log_deg
    elif "deg" in batch:
        deg = batch.deg
        log_deg = torch.log(deg + 1).unsqueeze(-1)
    else:
        #warnings.warn("Compute the degree on the fly; Might be problematric if have applied edge-padding to complete graphs")
        deg = degree(batch.edge_index[1],
                               num_nodes=batch.num_nodes,
                               dtype=torch.float
                               )
        log_deg = torch.log(deg + 1)
    log_deg = log_deg.view(batch.num_nodes, 1)
    return log_deg

class CALayer(nn.Module):  #input shape: n, h, w, c

    def __init__(self, features, reduction=4, scatter_option='mean', use_bias=True):
        super().__init__()
        self.features = features
        self.reduction = reduction
        self.use_bias = use_bias
        self.Conv_0 = nn.Linear(self.features, self.features//self.reduction, bias=self.use_bias)  #1*1 conv
        self.relu = nn.ReLU()
        self.Conv_1 = nn.Linear(self.features//self.reduction, self.features, bias=self.use_bias)  #1*1 conv
        self.sigmoid = nn.Sigmoid()
        self.option=scatter_option
        if self.option == 'add' :
            self.scatter_cal = scatter_add
        elif self.option == 'mean' :
            self.scatter_cal = scatter_mean

    def forward(self, x, batch):
        y = x
        y=self.scatter_cal(y, batch, dim=0)
        y=y[batch]
        y = self.Conv_0(y)
        y = self.relu(y)
        y = self.Conv_1(y)
        y = self.sigmoid(y)
        return x*y







class TiGTLayer(nn.Module):
    """Local MPNN + full graph attention x-former layer.
    """

    def __init__(self, dim_h,
                 local_gnn_type, global_model_type, num_heads, act='relu',
                 pna_degrees=None, equivstable_pe=False, dropout=0.0,
                 attn_dropout=0.0, layer_norm=False, batch_norm=True,
                 bigbird_cfg=None, log_attn_weights=False, type_calayer='no_res', scatter_option='mean', local_mode='single', deg_scaler=False):
  
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.activation = register.act_dict[act]
        self.local_mode = local_mode
        self.log_attn_weights = log_attn_weights

        self.deg_scaler = deg_scaler
        if log_attn_weights and global_model_type not in ['Transformer',
                                                          'BiasedTransformer']:
            raise NotImplementedError(
                f"Logging of attention weights is not supported "
                f"for '{global_model_type}' global attention model."
            )

        # # Local message-passing model.
        self.local_gnn_with_edge_attr = True
        if local_gnn_type == 'None':
            self.local_model = None
        else:
            if self.local_mode == 'single' or self.local_mode == 'nono':
                self.local_model, self.local_gnn_with_edge_attr = self._init_gnn_model(local_gnn_type, dim_h, num_heads=num_heads, equivstable_pe=self.equivstable_pe, dropout=dropout, act=act, pna_degrees=pna_degrees)
                self.local_model2 = self.local_model
            elif local_mode == 'double':
                self.local_model, self.local_gnn_with_edge_attr = self._init_gnn_model(local_gnn_type, dim_h, num_heads=num_heads, equivstable_pe=self.equivstable_pe, dropout=dropout, act=act, pna_degrees=pna_degrees)
                self.local_model2, self.local_gnn_with_edge_attr2 = self._init_gnn_model(local_gnn_type, dim_h, num_heads=num_heads, equivstable_pe=self.equivstable_pe, dropout=dropout, act=act, pna_degrees=pna_degrees)
        self.local_gnn_type = local_gnn_type

        # Global attention transformer-style model.
        if global_model_type == 'None':
            self.self_attn = None
        elif global_model_type in ['Transformer', 'BiasedTransformer']:
            self.self_attn = torch.nn.MultiheadAttention(
                dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
        else:
            raise ValueError(f"Unsupported global x-former model: "
                             f"{global_model_type}")
        self.global_model_type = global_model_type

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            self.norm1_local = pygnn.norm.LayerNorm(dim_h)
            self.norm1_attn = pygnn.norm.LayerNorm(dim_h)
            self.norm2_local = pygnn.norm.LayerNorm(dim_h)
            self.norm2_attn = pygnn.norm.LayerNorm(dim_h)
            # self.norm1_local = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_attn = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_local = pygnn.norm.InstanceNorm(dim_h)
            # self.norm1_attn = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
            self.norm2_local = nn.BatchNorm1d(dim_h)
            self.norm2_attn = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        self.act_fn_ff = self.activation()
        if self.layer_norm:
            self.norm2 = pygnn.norm.LayerNorm(dim_h)
            # self.norm2 = pygnn.norm.GraphNorm(dim_h)
            # self.norm2 = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

        #self.type_calayer = None, 'no_res', 'res', 'init_res'
        self.type_calayer =type_calayer
        self.calayer=CALayer(dim_h, 4, scatter_option=scatter_option)

        if self.deg_scaler:
            self.deg_coef = nn.Parameter(torch.zeros(1, dim_h, 2))
            nn.init.xavier_normal_(self.deg_coef)

    def forward(self, batch):
        h = batch.x
        h_in1 = h  # for first residual connection

        h_out_list = []

        # Processing for edge attributes using self.local_model.
        h_local, batch.edge_attr = self._process_local(self.local_model, h, batch.edge_index, batch.edge_attr, self.norm1_local,batch=batch, h_in=h_in1)
        h_out_list.append(h_local)
        if self.local_mode != 'nono':
            # Processing for cycle edge attributes using self.local_model_c.
            h_local_c, batch.cycle_attr = self._process_local(self.local_model2, h, batch.cycle_index, batch.cycle_attr, self.norm2_local, batch=batch, h_in=h_in1)
            h_out_list.append(h_local_c)

        # Multi-head attention.
        if self.self_attn is not None:
            h_dense, mask = to_dense_batch(h, batch.batch)
            if self.global_model_type == 'Transformer':
                h_attn = self._sa_block(h_dense, None, ~mask)[mask]
            elif self.global_model_type == 'BiasedTransformer':
                # Use Graphormer-like conditioning, requires `batch.attn_bias`.
                h_attn = self._sa_block(h_dense, batch.attn_bias, ~mask)[mask]
            elif self.global_model_type == 'Performer':
                h_attn = self.self_attn(h_dense, mask=mask)[mask]
            elif self.global_model_type == 'BigBird':
                h_attn = self.self_attn(h_dense, attention_mask=mask)
            else:
                raise RuntimeError(f"Unexpected {self.global_model_type}")

            h_attn = self.dropout_attn(h_attn)
            h_attn = h_in1 + h_attn  # Residual connection.
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn, batch.batch)
            if self.batch_norm:
                h_attn = self.norm1_attn(h_attn)
            h_out_list.append(h_attn)

        # Combine local and global outputs.
        # h = torch.cat(h_out_list, dim=-1)
        h = sum(h_out_list)

        # degree scaler
        if self.deg_scaler:
            log_deg = get_log_deg(batch)
            h = torch.stack([h, h * log_deg], dim=-1)
            h = (h * self.deg_coef).sum(dim=-1)

        # Feed Forward block.
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)

        if self.type_calayer == 'res' :
            h = h + self.calayer(h, batch.batch)
        elif self.type_calayer == 'no_res' :
            h = self.calayer(h, batch.batch)
        elif self.type_calayer == 'init_res' :
            h = h + self.calayer(h_in1, batch.batch)
        elif self.type_calayer == 'nono':
            h=h

        batch.x = h
        return batch

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block.
        """
        if not self.log_attn_weights:
            x = self.self_attn(x, x, x,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask,
                               need_weights=False)[0]
        else:
            # Requires PyTorch v1.11+ to support `average_attn_weights=False`
            # option to return attention weights of individual heads.
            x, A = self.self_attn(x, x, x,
                                  attn_mask=attn_mask,
                                  key_padding_mask=key_padding_mask,
                                  need_weights=True,
                                  average_attn_weights=False)
            self.attn_weights = A.detach().cpu()
        return x

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = f'summary: dim_h={self.dim_h}, ' \
            f'local_gnn_type={self.local_gnn_type}, ' \
            f'global_model_type={self.global_model_type}, ' \
            f'heads={self.num_heads}'
        return s
    
    def _process_local(self, model, h, edge_index, edge_attr, norm_local, batch=None, h_in=None):
        # Local MPNN helper function.
        if model is None:
            return None, edge_attr

        model: pygnn.conv.MessagePassing  # Typing hint.
        if self.local_gnn_type == 'CustomGatedGCN':
            es_data = batch.pe_EquivStableLapPE if self.equivstable_pe else None
            h_local, edge_attr = model(Batch(batch=batch,
                                    x=h,
                                    edge_index=edge_index,
                                    edge_attr=edge_attr,
                                    pe_EquivStableLapPE=es_data))

            # GatedGCN does residual connection and dropout internally.
        else:
            if self.local_gnn_with_edge_attr:
                if self.equivstable_pe:
                    h_local = model(h,
                                    edge_index,
                                    edge_attr,
                                    batch.pe_EquivStableLapPE)
                else:
                    h_local = model(h,
                                    edge_index,
                                    edge_attr)
            else:
                h_local = model(h, edge_index)
            h_local = self.dropout_local(h_local)
            if h_in != None:
                h_local = h_in + h_local  # Residual connection.
        if self.layer_norm:
            h_local = norm_local(h_local, batch.batch)
        if self.batch_norm:
            h_local = norm_local(h_local)
        
        return h_local, edge_attr

    def _init_gnn_model(self, model_type, dim_h, num_heads=None, equivstable_pe=False, dropout=0.0, act='relu', pna_degrees=None):
        """
        Initialize a GNN layer based on the given model type.
        """
        if model_type == 'GCN':
            return pygnn.GCNConv(dim_h, dim_h), False  # Return model and a flag indicating if it supports edge attributes
        elif model_type == 'GIN':
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h), self.activation(), Linear_pyg(dim_h, dim_h))
            return pygnn.GINConv(gin_nn), False
        elif model_type == 'GENConv':
            return pygnn.GENConv(dim_h, dim_h), True
        elif model_type == 'GINE':
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h), self.activation(), Linear_pyg(dim_h, dim_h))
            if equivstable_pe:
                return GINEConvESLapPE(gin_nn), True
            else:
                return pygnn.GINEConv(gin_nn), True
        elif model_type == 'GAT':
            return pygnn.GATConv(in_channels=dim_h, out_channels=dim_h // num_heads, heads=num_heads, edge_dim=dim_h), True
        elif model_type == 'PNA':
            aggregators = ['mean', 'max', 'sum']
            scalers = ['identity']
            deg = torch.from_numpy(np.array(pna_degrees))
            return pygnn.PNAConv(dim_h, dim_h, aggregators=aggregators, scalers=scalers, deg=deg, edge_dim=min(128, dim_h),
                                towers=1, pre_layers=1, post_layers=1, divide_input=False), True
        elif model_type == 'CustomGatedGCN':
            return GatedGCNLayer(dim_h, dim_h, dropout=dropout, residual=True, act=act, equivstable_pe=equivstable_pe), True
        elif model_type == 'None':
            return None, False
        else:
            raise ValueError(f"Unsupported local GNN model: {model_type}")
        




import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
from torch_geometric.graphgym.register import register_network

#from graphgps.layer.gps_layer import GPSLayer
from graphgps.encoder.type_dict_encoder import TypeDictCycleEncoder

class TiGTFeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in):
        super(TiGTFeatureEncoder, self).__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_inner, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
            # Update dim_in to reflect the new dimension of the node features
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            # Hard-limit max edge dim for PNA.
            if 'PNA' in cfg.gt.layer_type:
                cfg.gnn.dim_edge = min(128, cfg.gnn.dim_inner)
            else:
                cfg.gnn.dim_edge = cfg.gnn.dim_inner
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[
                cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_edge, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
        if cfg.posenc_CY2C.enable == True:
            self.cycle_encoder = TypeDictCycleEncoder(cfg.gnn.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_edge, -1, -1, has_act=False,
                                    has_bias=False, cfg=cfg))

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
    
@register_network('TiGTModel')
class TiGTModel(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = TiGTFeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if not cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in:
            raise ValueError(
                f"The inner and hidden dims must match: "
                f"embed_dim={cfg.gt.dim_hidden} dim_inner={cfg.gnn.dim_inner} "
                f"dim_in={dim_in}"
            )

        try:
            local_gnn_type, global_model_type = cfg.gt.layer_type.split('+')
        except:
            raise ValueError(f"Unexpected layer type: {cfg.gt.layer_type}")
        layers = []
        for _ in range(cfg.gt.layers):
            layers.append(TiGTLayer(
                dim_h=cfg.gt.dim_hidden,
                local_gnn_type=local_gnn_type,
                global_model_type=global_model_type,
                num_heads=cfg.gt.n_heads,
                act=cfg.gnn.act,
                pna_degrees=cfg.gt.pna_degrees,
                equivstable_pe=cfg.posenc_EquivStableLapPE.enable,
                dropout=cfg.gt.dropout,
                attn_dropout=cfg.gt.attn_dropout,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
                bigbird_cfg=cfg.gt.bigbird,
                log_attn_weights=cfg.train.mode == 'log-attn-weights',
                type_calayer=cfg.cy2.type_calayer, #'no_res', # new 
                scatter_option=cfg.cy2.scatter_option, #'mean', #new
                local_mode = cfg.cy2.local_mode,
                deg_scaler = cfg.cy2.deg_scaler
            ))
        self.layers = torch.nn.ModuleList(layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

        
        if cfg.cy2.pe_act == 'tanh':
            self.pe_activation = torch.nn.Tanh()
        elif cfg.cy2.pe_act == 'relu':
            self.pe_activation = torch.nn.ReLU()# register.act_dict[cfg.cy2.pe_act] #new
        else:
            print('error in pe activation')


        # # Local message-passing model.
        self.local_gnn_type = cfg.cy2.local_gnn_type #new 
        self.dim_h = cfg.cy2.dim_hidden #new 
        self.num_heads = cfg.cy2.n_heads #new
        self.equivstable_pe =False
        self.dropout = cfg.cy2.dropout #new
        self.act = cfg.cy2.act  #new
        self.activation = register.act_dict[cfg.cy2.act]
        self.pna_degrees=None 
        self.local_mode = cfg.cy2.pe_local_mode #new
        self.local_gnn_with_edge_attr = True 

        if self.local_gnn_type == 'None':
            self.local_model = None
        else:
            # This block is common for both 'single' and 'double'
            self.local_model, self.local_gnn_with_edge_attr = self._init_gnn_model(self.local_gnn_type,self.dim_h,num_heads=self.num_heads,equivstable_pe=self.equivstable_pe,dropout=self.dropout,act=self.act,pna_degrees=self.pna_degrees)
            if self.local_mode == 'single':
                self.local_model2 = self.local_model
            elif self.local_mode == 'double':
                self.local_model2, self.local_gnn_with_edge_attr = self._init_gnn_model(self.local_gnn_type,self.dim_h,num_heads=self.num_heads,equivstable_pe=self.equivstable_pe,dropout=self.dropout,act=self.act,pna_degrees=self.pna_degrees)
        self.coef = nn.Parameter(torch.ones(1,self.dim_h, 2)/2)
        if cfg.cy2.pe_norm == 'layer':
            self.layer_norm=True
            self.batch_norm=False
            self.pe1_local = pygnn.norm.LayerNorm(self.dim_h)
            self.pe2_local = pygnn.norm.LayerNorm(self.dim_h)            
        elif cfg.cy2.pe_norm == 'batch' : 
            self.layer_norm=False
            self.batch_norm=True
            self.pe1_local = nn.BatchNorm1d(self.dim_h)
            self.pe2_local = nn.BatchNorm1d(self.dim_h)
        else : 
            print('error')
        self.dropout_local = nn.Dropout(cfg.cy2.dropout)



    def forward(self, batch):
        batch = self.encoder(batch)
        h, batch.pos_edge_attr = self._process_local(self.local_model, batch.x, batch.edge_index, batch.pos_edge_attr, self.pe1_local,batch=batch, h_in=batch.x)
        h_c, batch.cycle_attr = self._process_local(self.local_model2, batch.x, batch.cycle_index, batch.cycle_attr, self.pe2_local,batch=batch, h_in=batch.x)
        pe = (torch.stack([h, h_c], dim=-1)*self.coef).sum(-1)
        #print(pe)
        pe = self.pe_activation(pe)
        batch.x = batch.x + pe
        for i, layer in enumerate(self.layers):
            batch = layer(batch)

        batch = self.post_mp(batch)

        return batch



    def _process_local(self, model, h, edge_index, edge_attr, norm_local, batch=None, h_in=None):
        # Local MPNN helper function.
        if model is None:
            return None, edge_attr

        model: pygnn.conv.MessagePassing  # Typing hint.

        if self.local_gnn_type == 'CustomGatedGCN':
            es_data = batch.pe_EquivStableLapPE if self.equivstable_pe else None
            h_local, edge_attr = model(Batch(batch=batch,
                                    x=h,
                                    edge_index=edge_index,
                                    edge_attr=edge_attr,
                                    pe_EquivStableLapPE=es_data))

            # GatedGCN does residual connection and dropout internally.
        else:
            if self.local_gnn_with_edge_attr:
                if self.equivstable_pe:
                    h_local = model(h,
                                    edge_index,
                                    edge_attr,
                                    batch.pe_EquivStableLapPE)
                else:
                    h_local = model(h,
                                    edge_index,
                                    edge_attr)
            else:
                h_local = model(h, edge_index)
            h_local = self.dropout_local(h_local)
            if h_in != None:
                h_local = h_in + h_local  # Residual connection.
        if self.layer_norm:
            h_local = norm_local(h_local, batch.batch)
        if self.batch_norm:
            h_local = norm_local(h_local)
        
        return h_local, edge_attr

    def _init_gnn_model(self, model_type, dim_h, num_heads=None, equivstable_pe=False, dropout=0.0, act='relu', pna_degrees=None):
        """
        Initialize a GNN layer based on the given model type.
        """
        if model_type == 'GCN':
            return pygnn.GCNConv(dim_h, dim_h), False  # Return model and a flag indicating if it supports edge attributes
        elif model_type == 'GIN':
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h), self.activation(), Linear_pyg(dim_h, dim_h))
            return pygnn.GINConv(gin_nn), False
        elif model_type == 'GENConv':
            return pygnn.GENConv(dim_h, dim_h), True
        elif model_type == 'GINE':
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h), self.activation(), Linear_pyg(dim_h, dim_h))
            if equivstable_pe:
                return GINEConvESLapPE(gin_nn), True
            else:
                return pygnn.GINEConv(gin_nn), True
        elif model_type == 'GAT':
            return pygnn.GATConv(in_channels=dim_h, out_channels=dim_h // num_heads, heads=num_heads, edge_dim=dim_h), True
        elif model_type == 'PNA':
            aggregators = ['mean', 'max', 'sum']
            scalers = ['identity']
            deg = torch.from_numpy(np.array(pna_degrees))
            return pygnn.PNAConv(dim_h, dim_h, aggregators=aggregators, scalers=scalers, deg=deg, edge_dim=min(128, dim_h),
                                towers=1, pre_layers=1, post_layers=1, divide_input=False), True
        elif model_type == 'CustomGatedGCN':
            return GatedGCNLayer(dim_h, dim_h, dropout=dropout, residual=True, act=act, equivstable_pe=equivstable_pe), True
        elif model_type == 'None':
            return None, False
        else:
            raise ValueError(f"Unsupported local GNN model: {model_type}")
