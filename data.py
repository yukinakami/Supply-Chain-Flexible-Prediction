# data.py
import pandas as pd
import numpy as np
import torch

class SupplyChainGraphDataset:
    def __init__(self, file_path):
        """
        初始化数据集
        :param file_path: xlsx 文件路径
        """
        self.file_path = file_path
        self.df = None
        self.node_features = None
        self.edge_index = None
        self.edge_features = None
        self.node_id_map = {}   # Symbol -> 节点ID
        self.node_symbols = []  # 节点顺序对应的 Symbol 列表
        self.load_data()
        self.process_data()

    def load_data(self):
        """读取Excel数据"""
        self.df = pd.read_excel(self.file_path)
        required_columns = [
            "Symbol", "EndDate", "Rank", "supply", "demanding", "h", 
            "demand_amount", "PurchaseAmount", "Gdp_second", "Gdp_percent", 
            "h_supply", "h_demanding"
        ]
        missing = [col for col in required_columns if col not in self.df.columns]
        if missing:
            raise ValueError(f"缺失必要字段: {missing}")

        # 填充缺失值
        for col in required_columns:
            if self.df[col].dtype in [np.float64, np.int64]:
                self.df[col] = self.df[col].fillna(0)
            else:
                self.df[col] = self.df[col].fillna("0")

        # 转换数据类型
        self.df["Symbol"] = self.df["Symbol"].astype(str).str.zfill(6)
        self.df["supply"] = self.df["supply"].astype(str).str.zfill(6)
        self.df["demanding"] = self.df["demanding"].astype(str).str.zfill(6)
        self.df["EndDate"] = self.df["EndDate"].astype(int)
        # 标准化 GDP
        self.df["Gdp_second"] = (self.df["Gdp_second"] - self.df["Gdp_second"].mean()) / self.df["Gdp_second"].std()
        self.df["Gdp_percent"] = (self.df["Gdp_percent"] - self.df["Gdp_percent"].mean()) / self.df["Gdp_percent"].std()

    def process_data(self):
        """处理数据，生成图结构"""
        # 1. 创建所有唯一节点
        all_symbols = pd.unique(
            self.df[["Symbol", "supply", "demanding"]].values.ravel()
        )
        self.node_symbols = all_symbols.tolist()  # 保存顺序对应的 Symbol
        self.node_id_map = {sym: idx for idx, sym in enumerate(all_symbols)}

        # 2. 构建节点特征
        node_features_list = []
        for sym in all_symbols:
            rows = self.df[self.df["Symbol"] == sym]
            if not rows.empty:
                row = rows.iloc[0]
                feat = [
                    row["h"], 
                    row["Gdp_second"], 
                    row["Gdp_percent"]
                ]
            else:
                feat = [0.0, 0.0, 0.0]
            node_features_list.append(feat)
        self.node_features = torch.tensor(node_features_list, dtype=torch.float)

        # 3. 构建边（supply -> Symbol -> demanding）
        edges = []
        edge_feats = []
        for _, row in self.df.iterrows():
            # 上游 -> 本企业
            edges.append([self.node_id_map[row["supply"]], self.node_id_map[row["Symbol"]]])
            edge_feats.append([row["h_supply"], row["PurchaseAmount"], row["Rank"]])
            # 本企业 -> 下游
            edges.append([self.node_id_map[row["Symbol"]], self.node_id_map[row["demanding"]]])
            edge_feats.append([row["h_demanding"], row["demand_amount"], row["Rank"]])

        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # shape: [2, num_edges]
        self.edge_features = torch.tensor(edge_feats, dtype=torch.float)

    def get_graph_data(self):
        """
        返回图数据
        :return: node_features, edge_index, edge_features, node_symbols
        """
        print("节点特征 shape:", self.node_features.shape)
        print("边索引 shape:", self.edge_index.shape)
        print("边特征 shape:", self.edge_features.shape)
        return self.node_features, self.edge_index, self.edge_features, self.node_symbols


# if __name__ == "__main__":
#     dataset = SupplyChainGraphDataset("data.xlsx")
#     node_features, edge_index, edge_features, node_symbols = dataset.get_graph_data()
#     print(node_symbols[:10])
