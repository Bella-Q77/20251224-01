import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import os

# 设置随机种子以保证结果可复现
np.random.seed(42)

# 1. 选择国家列表（除中国外的20个主要国家，包括美国、日本、加拿大等）
countries = [
    'United States', 'Japan', 'Germany', 'United Kingdom', 'France',
    'Italy', 'Canada', 'Australia', 'South Korea', 'Russia',
    'Brazil', 'India', 'Mexico', 'Indonesia', 'Turkey',
    'Saudi Arabia', 'South Africa', 'Argentina', 'Spain', 'Netherlands'
]

# 2. 生成2021-2023年的宏观指标数据
years = [2021, 2022, 2023]
indicators = ['GDP', 'PPI', 'CPI', 'M2', 'Exchange_Rate']

# 生成基础数据
data = []
for country in countries:
    for year in years:
        # GDP：以万亿美元为单位，根据国家经济规模生成
        base_gdp = {
            'United States': 23, 'Japan': 5, 'Germany': 4, 'United Kingdom': 3.5,
            'France': 2.9, 'Italy': 2.1, 'Canada': 2, 'Australia': 1.6,
            'South Korea': 1.8, 'Russia': 1.8, 'Brazil': 1.6, 'India': 3.2,
            'Mexico': 1.3, 'Indonesia': 1.1, 'Turkey': 1, 'Saudi Arabia': 0.8,
            'South Africa': 0.3, 'Argentina': 0.4, 'Spain': 1.4, 'Netherlands': 0.9
        }[country]
        gdp = base_gdp * (1 + np.random.uniform(-0.05, 0.1))  # 每年波动
        
        # PPI：生产者价格指数，以%为单位
        ppi = np.random.uniform(-2, 8)
        
        # CPI：消费者价格指数，以%为单位
        cpi = np.random.uniform(1, 8)
        
        # M2：货币供应量，以万亿美元为单位，根据GDP的一定比例生成
        m2 = gdp * np.random.uniform(0.5, 2.5)
        
        # Exchange_Rate：汇率，相对于美元
        if country == 'United States':
            exchange_rate = 1.0
        else:
            exchange_rate = np.random.uniform(0.5, 10)
        
        data.append({
            'Country': country,
            'Year': year,
            'GDP': round(gdp, 2),
            'PPI': round(ppi, 2),
            'CPI': round(cpi, 2),
            'M2': round(m2, 2),
            'Exchange_Rate': round(exchange_rate, 2)
        })

# 转换为DataFrame并保存
country_year_feature = pd.DataFrame(data)
country_year_feature.to_csv('country_year_feature.csv', index=False, encoding='utf-8-sig')
print("已生成country_year_feature.csv")

# 3. 数据清洗
# 3.1 缺失值检查
missing_values = country_year_feature.isnull().sum()
print("\n缺失值检查结果：")
print(missing_values)

# 3.2 异常值检测（使用Z-score）
numeric_cols = ['GDP', 'PPI', 'CPI', 'M2', 'Exchange_Rate']
z_scores = stats.zscore(country_year_feature[numeric_cols])
abs_z_scores = np.abs(z_scores)
outliers = (abs_z_scores > 3).any(axis=1)
print(f"\n检测到异常值数量：{outliers.sum()}")

# 3.3 重复值检查
duplicates = country_year_feature.duplicated().sum()
print(f"重复值数量：{duplicates}")

# 4. 离差标准化（Min-Max标准化）
normalized_data = country_year_feature.copy()
for year in years:
    year_data = normalized_data[normalized_data['Year'] == year]
    for col in numeric_cols:
        min_val = year_data[col].min()
        max_val = year_data[col].max()
        if max_val > min_val:
            normalized_data.loc[normalized_data['Year'] == year, col] = \
                (year_data[col] - min_val) / (max_val - min_val)

normalized_data.to_csv('Normalization.csv', index=False, encoding='utf-8-sig')
print("\n已生成Normalization.csv")

# 5. 相关性分析
correlation_results = {}
for year in years:
    year_data = country_year_feature[country_year_feature['Year'] == year][numeric_cols]
    corr_matrix = year_data.corr()
    correlation_results[year] = corr_matrix

# 保存相关性分析结果
with open('correlation_results.json', 'w') as f:
    json.dump({str(k): v.to_dict() for k, v in correlation_results.items()}, f, indent=2)
print("已生成correlation_results.json")

# 6. 变异系数法确定权重
weight_data = []
for year in years:
    year_data = country_year_feature[country_year_feature['Year'] == year][numeric_cols]
    # 计算变异系数
    cv = year_data.std() / year_data.mean()
    # 计算权重
    weights = cv / cv.sum()
    # 保存权重
    for col, weight in weights.items():
        weight_data.append({
            'Year': year,
            'Indicator': col,
            'Weight': round(weight, 4)
        })

weight_df = pd.DataFrame(weight_data)
weight_df.to_csv('weight.csv', index=False, encoding='utf-8-sig')
print("已生成weight.csv")

# 7. 计算综合评分
# 读取权重数据
weights = {} 
for year in years:
    year_weights = weight_df[weight_df['Year'] == year]
    weights[year] = {row['Indicator']: row['Weight'] for _, row in year_weights.iterrows()}

# 计算每个国家每年的综合评分
scores = []
for country in countries:
    for year in years:
        # 获取标准化后的数据
        normalized_row = normalized_data[(normalized_data['Country'] == country) & (normalized_data['Year'] == year)]
        # 计算综合评分
        score = 0
        for col in numeric_cols:
            score += normalized_row[col].values[0] * weights[year][col]
        scores.append({
            'Country': country,
            'Year': year,
            'Score': round(score, 4)
        })

scores_df = pd.DataFrame(scores)
scores_df.to_csv('macro_index_score.csv', index=False, encoding='utf-8-sig')
print("已生成macro_index_score.csv")

# 8. 生成分析报告
report = """
# 国家宏观指数综合评分分析报告

## 1. 数据概述
- 分析国家：20个主要经济体（除中国外）
- 分析年度：2021-2023年
- 分析指标：GDP、PPI、CPI、M2、汇率

## 2. 数据清洗结果
- 缺失值：无
- 异常值：{outliers_sum}个（使用Z-score方法检测）
- 重复值：无

## 3. 相关性分析
### 2021年相关性矩阵
{corr_2021}

### 2022年相关性矩阵
{corr_2022}

### 2023年相关性矩阵
{corr_2023}

## 4. 权重分析
### 2021年指标权重
{weights_2021}

### 2022年指标权重
{weights_2022}

### 2023年指标权重
{weights_2023}

## 5. 综合评分结果
### 2021年Top 5国家
{top5_2021}

### 2022年Top 5国家
{top5_2022}

### 2023年Top 5国家
{top5_2023}
""".format(
    outliers_sum=outliers.sum(),
    corr_2021=correlation_results[2021].to_string(),
    corr_2022=correlation_results[2022].to_string(),
    corr_2023=correlation_results[2023].to_string(),
    weights_2021=weight_df[weight_df['Year'] == 2021].to_string(index=False),
    weights_2022=weight_df[weight_df['Year'] == 2022].to_string(index=False),
    weights_2023=weight_df[weight_df['Year'] == 2023].to_string(index=False),
    top5_2021=scores_df[scores_df['Year'] == 2021].sort_values('Score', ascending=False).head(5).to_string(index=False),
    top5_2022=scores_df[scores_df['Year'] == 2022].sort_values('Score', ascending=False).head(5).to_string(index=False),
    top5_2023=scores_df[scores_df['Year'] == 2023].sort_values('Score', ascending=False).head(5).to_string(index=False)
)

with open('analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print("已生成analysis_report.txt")

print("\n所有任务已完成！")