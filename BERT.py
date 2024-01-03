import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt

# 用于存储每个 epoch 的训练损失和验证准确度
train_losses = []
val_accuracies = []

# 加载数据
data = pd.read_csv("./Data/data.csv", encoding="utf-8")
# 将评分转换为分类
def score_to_category(score):
    if score >= 90:
        return 0  # 优秀
    elif score >= 75:
        return 1  # 良好
    elif score >= 60:
        return 2  # 一般
    else:
        return 3  # 不合格


data['种类'] = data['评分'].apply(score_to_category)

# 预处理数据
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


def encode_data(tokenizer, texts, max_len=128):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            max_length=max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
        # print(encoded)

    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)


max_len = 128  # 可以根据需要调整
input_ids, attention_masks = encode_data(tokenizer, data['内容'], max_len)

# 创建 PyTorch 数据集
labels = torch.tensor(data['种类'].values)
dataset = TensorDataset(input_ids, attention_masks, labels)

# 数据划分
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建数据加载器
batch_size = 32

train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
validation_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

# 加载预训练的 BERT 模型
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)

# GPU 支持
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 微调模型
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 5
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


# 训练过程
def train():
    model.train()
    total_loss = 0

    # 使用 tqdm 包装训练数据加载器
    progress_bar = tqdm(train_dataloader, desc="Training", leave=True)

    for batch in train_dataloader:
        b_input_ids, b_input_mask, b_labels = batch
        b_input_ids = b_input_ids.to(device)
        b_input_mask = b_input_mask.to(device)
        b_labels = b_labels.to(device)

        model.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # 更新进度条的描述
        progress_bar.set_description(f"Epoch {epoch + 1}")
        progress_bar.set_postfix(loss=loss.item())

    progress_bar.close()

    avg_train_loss = total_loss / len(train_dataloader)
    return avg_train_loss


# 验证过程
def evaluate():
    model.eval()
    predictions, true_labels = [], []

    for batch in validation_dataloader:
        b_input_ids, b_input_mask, b_labels = batch
        b_input_ids = b_input_ids.to(device)
        b_input_mask = b_input_mask.to(device)
        b_labels = b_labels.to(device)

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions.append(logits)
        true_labels.append(label_ids)

    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    return accuracy_score(np.argmax(predictions, axis=1), true_labels)


# 执行训练和评估
for epoch in range(0, epochs):
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
    train_loss = train()
    val_accuracy = evaluate()
    print("Training loss: {:.2f}".format(train_loss))
    print("Validation Accuracy: {:.2f}".format(val_accuracy))

    # 将损失和准确度加入到列表中
    train_losses.append(train_loss)
    val_accuracies.append(val_accuracy)

# 保存模型
model.save_pretrained("./Model")

# 绘制拟合曲线
plt.figure(figsize=(12, 6))

# 绘制训练损失
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 绘制验证准确度
plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()