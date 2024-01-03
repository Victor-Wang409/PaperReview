import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import pandas as pd

# 加载预训练的中文 BERT 分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 加载模型
model = BertForSequenceClassification.from_pretrained("./Model")
model.eval()  # 将模型设置为评估模式

# GPU 支持
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 准备待预测的数据
def prepare_data(tokenizer, text, max_len=128):
    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_id = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']
    return input_id, attention_mask

# 示例待预测文本
sample_text =  "本文围绕基于组织病理学图像的预后研究，提出一种用于预后分析的组织病理学全切片图像的新型表征模型和一种使用可信多模态回归模型预测患者癌症等级的方法。 论文选题属于学科前沿，但文献阅读不够全面，近年来深度学习领域相关最新论文较少，工作量不够。学生掌握了应有的基础理论和专业知识。论文写作不规范，对实验对比不足，分析也较欠缺。 论文未达到申请硕士学位的学术水平，不同意答辩。"
(input_id, attention_mask) = prepare_data(tokenizer, sample_text)

# 预测
input_id = input_id.to(device)
attention_mask = attention_mask.to(device)
with torch.no_grad():
    outputs = model(input_id, token_type_ids=None, attention_mask=attention_mask)
    logits = outputs[0]

# 将输出转换为分类
predicted_prob = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
predicted_class = np.argmax(predicted_prob, axis=1)
print("Predicted class:", predicted_class)
