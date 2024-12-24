import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np
from PSFAN import PSFAN

def train_source(model, source_data, target_data, num_epochs=100, batch_size=1024, lambda_mmd=0.1, lambda_domain=0.1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    source_data = source_data.to(device)
    target_data = target_data.to(device)

    source_loader = NeighborSampler(source_data.edge_index, sizes=[10, 10], batch_size=batch_size, shuffle=True, drop_last=True)
    target_loader = NeighborSampler(target_data.edge_index, sizes=[10, 10], batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    total_steps = num_epochs * len(source_loader)
    current_step = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for (batch_size_s, n_id_s, adjs_s), (batch_size_t, n_id_t, adjs_t) in zip(source_loader, target_loader):
            optimizer.zero_grad()

            current_step += 1
            p = current_step / total_steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # 获取源域节点特征和标签
            x_s = source_data.x[n_id_s].to(device)
            y_s = source_data.y[n_id_s[:batch_size_s]].to(device)
            adjs_s = [(adj.edge_index.to(device), adj.e_id.to(device), adj.size) for adj in adjs_s]

            # 获取目标域节点特征
            x_t = target_data.x[n_id_t].to(device)
            adjs_t = [(adj.edge_index.to(device), adj.e_id.to(device), adj.size) for adj in adjs_t]

            # 前向传播，传入 alpha
            s_pred, _, _, _, _ = model((x_s, adjs_s), (x_t, adjs_t), s_label=y_s, alpha=alpha)

            # 分类损失
            classification_loss = criterion(s_pred, y_s)

            # 提取源域和目标域特征
            source_feature = model.extract_features(x_s, adjs_s).view(-1, model.feature_dim)
            target_feature = model.extract_features(x_t, adjs_t).view(-1, model.feature_dim)

            # 计算 MMD 损失
            loss_mmd = model.mmd_loss.get_loss(source_feature, target_feature, y_s, None).to(device)

            # 域判别器预测
            source_domain_preds = model.domain_classifier(source_feature)
            target_domain_preds = model.domain_classifier(target_feature)
            domain_preds = torch.cat((source_domain_preds, target_domain_preds), dim=0)

            # 域标签
            domain_labels = torch.cat((
                torch.zeros(source_feature.size(0), dtype=torch.long).to(device),
                torch.ones(target_feature.size(0), dtype=torch.long).to(device)
            ))

            # 域损失
            domain_loss = criterion(domain_preds, domain_labels)

            # 总损失
            total_loss = classification_loss + lambda_mmd * loss_mmd + lambda_domain * domain_loss
            total_loss.backward()
            optimizer.step()

            num_batches += 1

        avg_loss = total_loss.item() / num_batches
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')


def train_fine_tune(model, target_data, num_epochs=100, batch_size=1024):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    target_data = target_data.to(device)

    target_loader = NeighborSampler(target_data.edge_index, sizes=[10, 10], batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch_size_t, n_id_t, adjs_t in target_loader:
            optimizer.zero_grad()

            # 设置 alpha（可以为 0 或其他值）
            alpha = 0.0

            # 获取目标域节点特征和标签
            x_t = target_data.x[n_id_t].to(device)
            y_t = target_data.y[n_id_t[:batch_size_t]].to(device) if hasattr(target_data, 'y') else None
            adjs_t = [(adj.edge_index.to(device), adj.e_id.to(device), adj.size) for adj in adjs_t]

            # 前向传播，传入 alpha
            _, t_pred, _, _, _ = model((None, None), (x_t, adjs_t), t_label=y_t, alpha=alpha)

            # 计算目标域的分类损失
            if y_t is not None:
                classification_loss = criterion(t_pred, y_t)
                classification_loss.backward()
                optimizer.step()
                total_loss += classification_loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f'Fine-tune Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')


def evaluate(model, data, batch_size=1024):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    data = data.to(device)

    loader = NeighborSampler(data.edge_index, sizes=[10, 10], batch_size=batch_size, shuffle=False)
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_size, n_id, adjs in loader:
            x = data.x[n_id].to(device)
            y = data.y[n_id[:batch_size]].cpu().numpy()
            adjs = [(adj.edge_index, adj.e_id, adj.size) for adj in adjs]
            adjs = [(edge_index.to(device), e_id.to(device), size) for edge_index, e_id, size in adjs]

            output = model.predict((x, adjs))  # 输出 logits
            probs = F.softmax(output, dim=1)  # 计算概率
            preds = output.max(1)[1].cpu().numpy()

            all_probs.append(probs.cpu().numpy())  # 收集概率
            all_preds.append(preds)
            all_labels.append(y)

    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)  # 形状：[num_samples, num_classes]
    all_labels = np.concatenate(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # 计算AUC（针对多分类，需要转换为一对多的形式）
    try:
        y_one_hot = np.eye(model.num_classes)[all_labels]  # 真实标签的 One-Hot 编码
        auc = roc_auc_score(y_one_hot, all_probs, average='weighted', multi_class='ovr')
    except Exception as e:
        print("AUC计算出错：", e)
        auc = None

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    if auc is not None:
        print(f'AUC: {auc:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载数据
    source_data = torch.load('2_source_240000.pt')
    target_data = torch.load('2_target_20%_50%.pt')  # 修改为目标域数据
    test_data = torch.load('2_test.pt')

    # 获取输入特征的维度
    num_features = source_data.num_node_features  # 或者 source_data.x.size(1)

    # 定义模型
    num_classes = 2  # 根据您的数据调整
    model = PSFAN(num_classes=num_classes, num_features=num_features).to(device)

    # 在源域上训练模型
    train_source(model, source_data, target_data, num_epochs=30, batch_size=1024, lambda_mmd=0, lambda_domain=0)

    # 在目标域数据上微调模型
    train_fine_tune(model, target_data, num_epochs=30, batch_size=1024)

    # 在目标域数据上评估模型（如果有标签）
    if hasattr(test_data, 'y'):
        print("在目标域数据上的评估结果：")
        evaluate(model, test_data, batch_size=1024)
