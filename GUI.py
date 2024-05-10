import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from joblib import load
import pandas as pd


def load_model(model_path, vectorizer_path):
    # 从文件中加载模型和向量器
    model = load(model_path)
    vectorizer = load(vectorizer_path)
    return model, vectorizer


def create_text_area(root):
    # 创建文本区域用于显示文件内容
    text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Arial", 10))
    text_area.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
    return text_area


def upload_and_display_file(text_area, model_info):
    # 上传并显示文件内容，进行模型预测
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if file_path:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
                text_area.config(state='normal')
                text_area.delete('1.0', tk.END)
                text_area.insert(tk.END, file_content)
                text_area.config(state='disabled')

            # 加载模型和向量器
            model, vectorizer = load_model(*model_info)

            # 进行预测并弹出结果
            prediction = predict(model, vectorizer, [file_content])
            messagebox.showinfo("预测结果", f"这篇论文的评级是：{prediction[0]}")
        except Exception as e:
            messagebox.showerror("错误", f"读取文件时出错：{e}")


def predict(model, vectorizer, new_data):
    # 使用加载的模型进行预测
    new_data_vec = vectorizer.transform(new_data)
    return model.predict(new_data_vec)


def setup_gui():
    # 设置GUI界面
    root = tk.Tk()
    root.title("Upload and Predict Text File")
    root.geometry("800x600")
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=1)

    text_area = create_text_area(root)

    # 算法选择下拉列表
    algorithms = {'Naive Bayes': ('./model/nb_model.joblib', './model/nb_vectorizer.joblib'),
                  'SVM': ('./model/svm_model.joblib', './model/svm_vectorizer.joblib')}
    selected_algorithm = tk.StringVar(root)
    selected_algorithm.set('Naive Bayes')  # 默认选择朴素贝叶斯
    algorithm_menu = tk.OptionMenu(root, selected_algorithm, *algorithms.keys())
    algorithm_menu.grid(row=0, column=1, sticky="ew", padx=10, pady=10)

    upload_button = tk.Button(root, text="上传文件",
                              command=lambda: upload_and_display_file(text_area, algorithms[selected_algorithm.get()]))
    upload_button.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

    root.mainloop()


if __name__ == "__main__":
    setup_gui()
