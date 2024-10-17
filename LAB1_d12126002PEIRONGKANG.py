import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import re

# 清理数据的函数：假设文本数据位于特定列
def clean_text(text):
    # 清理和处理非标准字符或无用的符号
    text = re.sub(r'\t|\n', ' ', text)  # 移除 tab 和换行符
    return text

# 本地 CSV 文件路径
file_paths = [
    r"D:\OneDrive\國立台灣大學\NTU_113-1\NTU_113-1\td_freq_db_alt_atheism.csv",
    r"D:\OneDrive\國立台灣大學\NTU_113-1\NTU_113-1\td_freq_db_comp_graphics.csv",
    r"D:\OneDrive\國立台灣大學\NTU_113-1\NTU_113-1\td_freq_db_sci_med.csv",
    r"D:\OneDrive\國立台灣大學\NTU_113-1\NTU_113-1\td_freq_db_soc_religion_christian.csv"
]

# 添加类别标签
categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']

# 加载并处理数据
dataframes = []
for path, category in zip(file_paths, categories):
    df = pd.read_csv(path, header=None, names=['text'])  # 假设数据在第一列
    df['text'] = df['text'].apply(clean_text)  # 清理文本
    df['category'] = category  # 添加类别标签
    dataframes.append(df)

# 合并所有数据
data = pd.concat(dataframes, ignore_index=True)

# 显示清理后的数据
print(data.head())
# 文本向量化
tfidf_vect = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2)
X_tfidf = tfidf_vect.fit_transform(data['text'])

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, data['category'], test_size=0.3, random_state=42)

# 训练朴素贝叶斯模型
nb_tfidf = MultinomialNB()
nb_tfidf.fit(X_train, y_train)

# 进行预测
y_pred_tfidf = nb_tfidf.predict(X_test)

# 输出分类报告和准确率
print("Classification report (TF-IDF Vectorizer):")
print(classification_report(y_test, y_pred_tfidf))
print(f"Accuracy (TF-IDF Vectorizer): {accuracy_score(y_test, y_pred_tfidf)}")
