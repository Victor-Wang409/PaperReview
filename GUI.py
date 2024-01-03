import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import scrolledtext
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jieba

import NBM

# 数据和标签
data = []
# data = ""
labels = []

def data_loader():
    paper_comments = pd.read_csv("./Data/paper_comments.csv", encoding="gbk")
    paper_content = paper_comments["内容"]
    paper_labels = paper_comments["评价"]
    # print(paper_content.head())
    # print(paper_labels.head())
    for c in paper_content:
        data.append(c)
    for l in paper_labels:
        labels.append(l)
    # print(data)
    # print(labels)

# 创建文本区域
def create_text_area(root):
    text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Arial", 10))
    text_area.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
    return text_area

# 文件上传和显示功能
def upload_and_display_file(text_area):
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if file_path:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
                text_area.config(state='normal')
                text_area.delete('1.0', tk.END)
                text_area.insert(tk.END, file_content)
                text_area.config(state='disabled')
                # 进行预测并弹出结果
                prediction = NBM.predict(model, vectorizer, [file_content])
                messagebox.showinfo("预测结果", f"这篇论文的评级是：{prediction[0]}")
        except Exception as e:
            messagebox.showerror("错误", f"读取文件时出错：{e}")

# GUI 设置
def setup_gui():
    root = tk.Tk()
    root.title("Upload and Predict Text File")
    root.geometry("800x600")

    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=1)

    text_area = create_text_area(root)

    upload_button = tk.Button(root, text="上传文件", command=lambda: upload_and_display_file(text_area))
    upload_button.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

    root.mainloop()

# 加载数据
data_loader()
# 训练模型
model, vectorizer = NBM.train_model(data, labels)
# 运行 GUI
setup_gui()
