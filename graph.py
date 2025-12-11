# graph.py
import torch
from torch_geometric.data import Data
from data import SupplyChainGraphDataset

class SupplyChainGraph:
    def __init__(self, data_path):
        """
        初始化图数据
        :param data_path: Excel 数据路径
        """
        self.dataset = SupplyChainGraphDataset(data_path)
        self.graph = self.build_graph()

    def build_graph(self):
        """
        将节点特征和边特征构建为 PyG Data 对象
        """
        node_features, edge_index, edge_features, node_symbols = self.dataset.get_graph_data()
        self.node_symbols = node_symbols  # 保存到类属性，方便后续可视化


        # PyG Data 对象
        graph_data = Data(
            x=node_features,           # 节点特征 [num_nodes, num_node_features]
            edge_index=edge_index,     # 边索引 [2, num_edges]
            edge_attr=edge_features    # 边特征 [num_edges, num_edge_features]
        )
        print(graph_data.x.shape)
        return graph_data

    def get_graph(self):
        return self.graph

# if __name__ == "__main__":
#     data_path = "data.xlsx"
#     sc_graph = SupplyChainGraph(data_path)
#     graph = sc_graph.get_graph()
#     print("节点特征 shape:", graph.x.shape)
#     print("边索引 shape:", graph.edge_index.shape)
#     print("边特征 shape:", graph.edge_attr.shape)
