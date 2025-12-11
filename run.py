# run.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from graph import SupplyChainGraph
from model import SupplyChainGNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def compute_metrics(y_true, y_pred, tol=0.05):
    """
    计算回归指标，包括 MAE, RMSE, R2, Accuracy (容忍度)
    """
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    acc = np.mean(np.abs(y_true - y_pred) < tol)  # 容忍度 accuracy
    return mae, rmse, r2, acc

def train(model, data, train_idx, optimizer, criterion, device, tol=0.05):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[train_idx], data.y[train_idx])
    loss.backward()
    optimizer.step()
    mae, rmse, r2, acc = compute_metrics(data.y[train_idx], out[train_idx], tol)
    return loss.item(), mae, rmse, r2, acc

def evaluate(model, data, val_idx, criterion, device, tol=0.05):
    model.eval()
    with torch.no_grad():
        out = model(data)
        loss = criterion(out[val_idx], data.y[val_idx])
        mae, rmse, r2, acc = compute_metrics(data.y[val_idx], out[val_idx], tol)
    return loss.item(), mae, rmse, r2, acc

def main(data_path, epochs=100, lr=1e-3, hidden_channels=64, conv_type='GCN', test_size=0.2, tol=0.05):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载图数据
    sc_graph = SupplyChainGraph(data_path)
    graph = sc_graph.get_graph()

    # 2. 设置目标 y (假设 h 是节点特征的第一列)
    graph = graph.to(device)
    graph.y = graph.x[:, 0].unsqueeze(1).clone()

    # 3. 划分训练/验证集
    num_nodes = graph.num_nodes
    train_idx_np, val_idx_np = train_test_split(range(num_nodes), test_size=test_size, random_state=42)
    train_idx = torch.tensor(train_idx_np, dtype=torch.long).to(device)
    val_idx = torch.tensor(val_idx_np, dtype=torch.long).to(device)

    # 4. 初始化模型
    in_channels = graph.x.shape[1]
    model = SupplyChainGNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=1,
        conv_type=conv_type
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 5. 训练循环
    for epoch in range(1, epochs+1):
        train_loss, train_mae, train_rmse, train_r2, train_acc = train(model, graph, train_idx, optimizer, criterion, device, tol)
        val_loss, val_mae, val_rmse, val_r2, val_acc = evaluate(model, graph, val_idx, criterion, device, tol)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | "
                  f"Train Loss: {train_loss:.6f} | MAE: {train_mae:.6f} | RMSE: {train_rmse:.6f} | R2: {train_r2:.4f} | Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.6f} | MAE: {val_mae:.6f} | RMSE: {val_rmse:.6f} | R2: {val_r2:.4f} | Acc: {val_acc:.4f}")

    # 6. 保存模型
    torch.save(model.state_dict(), "supply_chain_gnn.pth")
    print("模型已保存: supply_chain_gnn.pth")

    # 7. 预测
    model.eval()
    with torch.no_grad():
        pred_h = model(graph).cpu().numpy()
    print("预测 h 示例:", pred_h[:10])

if __name__ == "__main__":
    data_path = "./final_with_GDP_H_Company_safe.xlsx"  # 你的数据文件路径
    main(data_path, epochs=1000, lr=1e-3, hidden_channels=64, conv_type='GCN')
