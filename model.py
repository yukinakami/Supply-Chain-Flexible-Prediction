# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class SupplyChainGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=1, conv_type='GCN', num_layers=2):
        """
        图神经网络模型，用于预测企业柔性 h
        :param in_channels: 节点特征维度
        :param hidden_channels: 隐藏层维度
        :param out_channels: 输出维度（预测 h，1）
        :param conv_type: 'GCN' 或 'GAT'
        :param num_layers: GNN 层数
        """
        super(SupplyChainGNN, self).__init__()
        self.convs = nn.ModuleList()
        self.num_layers = num_layers
        self.conv_type = conv_type

        # 第一层
        if conv_type == 'GCN':
            self.convs.append(GCNConv(in_channels, hidden_channels))
        elif conv_type == 'GAT':
            self.convs.append(GATConv(in_channels, hidden_channels, heads=4, concat=True))
        else:
            raise ValueError("conv_type must be 'GCN' or 'GAT'")

        # 中间隐藏层
        for _ in range(num_layers - 1):
            if conv_type == 'GCN':
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            else:
                self.convs.append(GATConv(hidden_channels*4, hidden_channels, heads=4, concat=True))

        # 输出层
        self.out_lin = nn.Linear(hidden_channels if conv_type=='GCN' else hidden_channels*4, out_channels)

    def forward(self, data):
        """
        前向传播
        :param data: PyG Data 对象
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.out_lin(x)  # 输出预测 h
        return x

# if __name__ == "__main__":
#     from .graph import SupplyChainGraph

#     # 构建图数据
#     data_path = "data.xlsx"
#     sc_graph = SupplyChainGraph(data_path)
#     graph = sc_graph.get_graph()

#     # 模型初始化
#     in_channels = graph.x.shape[1]  # 节点特征维度
#     model = SupplyChainGNN(in_channels=in_channels, hidden_channels=64, out_channels=1, conv_type='GCN')

#     # 前向测试
#     pred = model(graph)
#     print("预测 h shape:", pred.shape)  # [num_nodes, 1]
