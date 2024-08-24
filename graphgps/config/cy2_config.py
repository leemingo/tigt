from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_cy2')
def set_cfg_cy2(cfg):
    """Configuration for Graph Transformer-style models, e.g.:
    - Spectral Attention Network (SAN) Graph Transformer.
    - "vanilla" Transformer / Performer.
    - General Powerful Scalable (GPS) Model.
    """
    # Positional encodings argument group
    cfg.cy2 = CN()
    cfg.cy2.type_calayer = 'no_res'
    cfg.cy2.scatter_option = 'mean'
    cfg.cy2.local_mode='single'
    cfg.cy2.local_gnn_type = 'GINE'
    cfg.cy2.dim_hidden = 64
    cfg.cy2.n_heads = 4
    cfg.cy2.dropout = 0.0
    cfg.cy2.pe_local_mode = 'single'
    cfg.cy2.pe_act = 'tanh'
    cfg.cy2.act = 'relu'
    cfg.cy2.pe_norm = 'layer'
    cfg.cy2.cy2c_self = False
    #== New ==
    cfg.cy2.cy2c_same_attr = True
    cfg.cy2.cy2c_trans = False
    cfg.cy2.deg_scaler = False
    