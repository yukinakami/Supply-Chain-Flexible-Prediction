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
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")  # Seaborn 风格，美观

def compute_metrics(y_true, y_pred, tol=0.05):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    acc = np.mean(np.abs(y_true - y_pred) < tol)
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

def plot_training_curves(metrics_history):
    epochs = range(1, len(metrics_history['train_loss']) + 1)

    plt.figure(figsize=(16, 12))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, metrics_history['train_loss'], label='Train Loss')
    plt.plot(epochs, metrics_history['val_loss'], label='Val Loss')
    plt.title('Loss 曲线')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, metrics_history['train_mae'], label='Train MAE')
    plt.plot(epochs, metrics_history['val_mae'], label='Val MAE')
    plt.title('MAE 曲线')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, metrics_history['train_r2'], label='Train R2')
    plt.plot(epochs, metrics_history['val_r2'], label='Val R2')
    plt.title('R² 曲线')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs, metrics_history['train_acc'], label='Train Acc')
    plt.plot(epochs, metrics_history['val_acc'], label='Val Acc')
    plt.title('Accuracy (容忍度 ±0.05) 曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_predictions(true_h, pred_h, symbols=None):
    # 散点图：预测 vs 真实值
    plt.figure(figsize=(8, 6))
    plt.scatter(true_h, pred_h, alpha=0.7)
    plt.plot([true_h.min(), true_h.max()], [true_h.min(), true_h.max()], 'r--', label='y=x')
    plt.xlabel('True h')
    plt.ylabel('Predicted h')
    plt.title('预测值 vs 真实值')
    plt.legend()
    plt.show()

    # 残差直方图
    residuals = pred_h - true_h
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, bins=30, kde=True)
    plt.xlabel('Residuals')
    plt.title('预测残差分布')
    plt.show()

    # 条形图：前 20 个节点预测 vs 真实
    n = min(20, len(true_h))
    indices = np.arange(n)
    plt.figure(figsize=(12, 6))
    plt.bar(indices - 0.2, true_h[:n], width=0.4, label='True h')
    plt.bar(indices + 0.2, pred_h[:n], width=0.4, label='Pred h')
    plt.xticks(indices, symbols[:n] if symbols is not None else indices, rotation=45)
    plt.ylabel('h 值')
    plt.title('前 20 个节点预测与真实对比')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main(data_path, epochs=100, lr=1e-3, hidden_channels=64, conv_type='GCN', test_size=0.2, tol=0.05):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载图数据
    sc_graph = SupplyChainGraph(data_path)
    graph = sc_graph.get_graph()
    symbols = sc_graph.node_symbols  # 企业代码或名称列表

    # 设置目标 y
    graph = graph.to(device)
    graph.y = graph.x[:, 0].unsqueeze(1).clone()

    # 划分训练/验证集
    num_nodes = graph.num_nodes
    train_idx_np, val_idx_np = train_test_split(range(num_nodes), test_size=test_size, random_state=42)
    train_idx = torch.tensor(train_idx_np, dtype=torch.long).to(device)
    val_idx = torch.tensor(val_idx_np, dtype=torch.long).to(device)

    # 初始化模型
    in_channels = graph.x.shape[1]
    model = SupplyChainGNN(in_channels=in_channels, hidden_channels=hidden_channels,
                            out_channels=1, conv_type=conv_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 训练循环，同时记录指标
    metrics_history = {
        'train_loss': [], 'val_loss': [],
        'train_mae': [], 'val_mae': [],
        'train_r2': [], 'val_r2': [],
        'train_acc': [], 'val_acc': []
    }

    for epoch in range(1, epochs+1):
        train_loss, train_mae, train_rmse, train_r2, train_acc = train(model, graph, train_idx, optimizer, criterion, device, tol)
        val_loss, val_mae, val_rmse, val_r2, val_acc = evaluate(model, graph, val_idx, criterion, device, tol)

        metrics_history['train_loss'].append(train_loss)
        metrics_history['val_loss'].append(val_loss)
        metrics_history['train_mae'].append(train_mae)
        metrics_history['val_mae'].append(val_mae)
        metrics_history['train_r2'].append(train_r2)
        metrics_history['val_r2'].append(val_r2)
        metrics_history['train_acc'].append(train_acc)
        metrics_history['val_acc'].append(val_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | "
                  f"Train Loss: {train_loss:.6f} | MAE: {train_mae:.6f} | RMSE: {train_rmse:.6f} | R2: {train_r2:.4f} | Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.6f} | MAE: {val_mae:.6f} | RMSE: {val_rmse:.6f} | R2: {val_r2:.4f} | Acc: {val_acc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "supply_chain_gnn.pth")
    print("模型已保存: supply_chain_gnn.pth")

    # 预测
    model.eval()
    with torch.no_grad():
        pred_h = model(graph).cpu().numpy()
        true_h = graph.y.cpu().numpy()

    # 输出前 10 个节点预测
    print("前 10 个节点预测示例:")
    print("{:<10} {:<15} {:<15}".format("Symbol", "True h", "Pred h"))
    for i in range(10):
        print("{:<10} {:<15.6f} {:<15.6f}".format(symbols[i], true_h[i][0], pred_h[i][0]))

    # 可视化
    plot_training_curves(metrics_history)
    plot_predictions(true_h[:,0], pred_h[:,0], symbols)

if __name__ == "__main__":
    data_path = "./final_with_GDP_H_Company_safe.xlsx"
    main(data_path, epochs=1000, lr=1e-3, hidden_channels=64, conv_type='GCN')
