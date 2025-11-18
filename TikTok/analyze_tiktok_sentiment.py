# analyze_tiktok_sentiment.py
import pandas as pd
from transformers import pipeline
import time
from tqdm import tqdm

# 初始化情感分析模型
print("正在加载情感分析模型...")
classifier = pipeline("sentiment-analysis")

# 读取CSV文件
print("\n正在读取CSV文件...")
try:
    # 读取CSV，注意您的文件名
    df = pd.read_csv('tiktok_part_1.csv')
    print(f"成功读取 {len(df)} 条记录")
except FileNotFoundError:
    print("错误：找不到 tiktok_part_1.csv 文件")
    print("请确保CSV文件在当前目录下")
    exit()

# 查看数据概况
print(f"\n数据集包含 {len(df.columns)} 列")
print(f"文本列是否存在: {'text' in df.columns}")

# 提取文本列，过滤空值
texts = df['text'].dropna().tolist()
print(f"\n找到 {len(texts)} 条有效文本")

# 显示前3条文本作为示例
print("\n示例文本：")
for i, text in enumerate(texts[:3]):
    print(f"{i+1}. {text[:100]}...")

# 分析情感
print("\n开始情感分析（这可能需要几分钟）...")
results = []

# 使用进度条
for text in tqdm(texts, desc="分析进度"):
    try:
        # 截断过长的文本（模型有长度限制）
        truncated_text = text[:512] if len(text) > 512 else text
        
        # 分析情感
        result = classifier(truncated_text)[0]
        results.append({
            'text': text,
            'sentiment': result['label'],
            'confidence': result['score']
        })
        
    except Exception as e:
        # 处理可能的错误
        results.append({
            'text': text,
            'sentiment': 'ERROR',
            'confidence': 0.0
        })

# 创建结果DataFrame
results_df = pd.DataFrame(results)

# 统计分析
print("\n=== 情感分析统计 ===")
sentiment_counts = results_df['sentiment'].value_counts()
print(sentiment_counts)
print(f"\n积极比例: {(sentiment_counts.get('POSITIVE', 0) / len(results_df) * 100):.1f}%")
print(f"消极比例: {(sentiment_counts.get('NEGATIVE', 0) / len(results_df) * 100):.1f}%")

# 查看高置信度的积极和消极样本
print("\n=== 高置信度积极样本 (前5条) ===")
positive_samples = results_df[results_df['sentiment'] == 'POSITIVE'].nlargest(5, 'confidence')
for idx, row in positive_samples.iterrows():
    print(f"置信度: {row['confidence']:.2%}")
    print(f"文本: {row['text'][:150]}...")
    print("-" * 50)

print("\n=== 高置信度消极样本 (前5条) ===")
negative_samples = results_df[results_df['sentiment'] == 'NEGATIVE'].nlargest(5, 'confidence')
for idx, row in negative_samples.iterrows():
    print(f"置信度: {row['confidence']:.2%}")
    print(f"文本: {row['text'][:150]}...")
    print("-" * 50)

# 保存结果
output_file = 'tiktok_sentiment_results.csv'
results_df.to_csv(output_file, index=False, encoding='utf-8')
print(f"\n结果已保存到: {output_file}")

# 可选：将情感结果合并回原始数据
df_with_sentiment = df.copy()
df_with_sentiment['sentiment'] = None
df_with_sentiment['sentiment_confidence'] = None

# 匹配并更新情感结果
for idx, row in df_with_sentiment.iterrows():
    if pd.notna(row['text']):
        matching_result = results_df[results_df['text'] == row['text']]
        if not matching_result.empty:
            df_with_sentiment.at[idx, 'sentiment'] = matching_result.iloc[0]['sentiment']
            df_with_sentiment.at[idx, 'sentiment_confidence'] = matching_result.iloc[0]['confidence']

# 保存完整数据
complete_output_file = 'tiktok_complete_with_sentiment.csv'
df_with_sentiment.to_csv(complete_output_file, index=False, encoding='utf-8')
print(f"完整数据已保存到: {complete_output_file}")

print("\n分析完成！")