import pandas as pd
import numpy as np
import json
import random
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 数据获取与生成
# 定义20个主要国家（除中国外），包括美国、日本、加拿大等主要经济大国
countries = [
    'United States', 'Japan', 'Germany', 'United Kingdom', 'France',
    'India', 'Italy', 'Brazil', 'Canada', 'Russia',
    'South Korea', 'Australia', 'Spain', 'Mexico', 'Indonesia',
    'Netherlands', 'Turkey', 'Switzerland', 'Saudi Arabia', 'Argentina'
]

# 定义指标列表
indicators = ['GDP', 'PPI', 'CPI', 'M2', 'ExchangeRate']

# 定义年份范围
years = [2021, 2022, 2023]

# 生成模拟数据（更符合现实范围的模拟数据）
def generate_country_data(countries, years, indicators):
    data = []
    # 按国家规模分类
    g7_countries = ['United States', 'Japan', 'Germany', 'United Kingdom', 'France', 'Italy', 'Canada']
    large_emerging = ['India', 'Brazil', 'Russia', 'South Korea', 'Australia', 'Spain', 'Mexico', 'Indonesia', 'Netherlands']
    medium_emerging = ['Turkey', 'Switzerland', 'Saudi Arabia', 'Argentina']
    
    for country in countries:
        for year in years:
            # 根据国家类型设置不同的数据范围
            if country == 'United States':
                gdp = np.random.uniform(22000, 26000)  # GDP: 22-26万亿美元
                m2 = np.random.uniform(20000, 22000)   # M2: 20-22万亿美元
                exchange_rate = 1.0                   # 美元本位
            elif country in ['Japan', 'Germany']:
                gdp = np.random.uniform(4000, 5000)    # GDP: 4-5万亿美元
                m2 = np.random.uniform(6000, 8000)     # M2: 6-8万亿美元
                exchange_rate = np.random.uniform(100, 140) if country == 'Japan' else np.random.uniform(0.9, 1.1)  # 日元/美元 或 欧元/美元
            elif country == 'United Kingdom':
                gdp = np.random.uniform(3000, 3500)    # GDP: 3-3.5万亿美元
                m2 = np.random.uniform(5000, 6000)     # M2: 5-6万亿美元
                exchange_rate = np.random.uniform(0.7, 0.85)  # 英镑/美元
            elif country == 'France':
                gdp = np.random.uniform(2800, 3200)    # GDP: 2.8-3.2万亿美元
                m2 = np.random.uniform(4000, 5000)     # M2: 4-5万亿美元
                exchange_rate = np.random.uniform(0.9, 1.1)  # 欧元/美元
            elif country == 'Italy':
                gdp = np.random.uniform(2000, 2500)    # GDP: 2-2.5万亿美元
                m2 = np.random.uniform(3000, 4000)     # M2: 3-4万亿美元
                exchange_rate = np.random.uniform(0.9, 1.1)  # 欧元/美元
            elif country == 'Canada':
                gdp = np.random.uniform(2000, 2500)    # GDP: 2-2.5万亿美元
                m2 = np.random.uniform(1500, 2000)     # M2: 1.5-2万亿美元
                exchange_rate = np.random.uniform(0.7, 0.85)  # 加元/美元
            elif country == 'India':
                gdp = np.random.uniform(3000, 3500)    # GDP: 3-3.5万亿美元
                m2 = np.random.uniform(20000, 25000)   # M2: 20-25万亿美元
                exchange_rate = np.random.uniform(70, 85)  # 卢比/美元
            elif country == 'Brazil':
                gdp = np.random.uniform(1500, 2000)    # GDP: 1.5-2万亿美元
                m2 = np.random.uniform(4000, 5000)     # M2: 4-5万亿美元
                exchange_rate = np.random.uniform(5, 6)  # 雷亚尔/美元
            elif country == 'Russia':
                gdp = np.random.uniform(1500, 2000)    # GDP: 1.5-2万亿美元
                m2 = np.random.uniform(3000, 4000)     # M2: 3-4万亿美元
                exchange_rate = np.random.uniform(60, 80)  # 卢布/美元
            elif country == 'South Korea':
                gdp = np.random.uniform(1800, 2200)    # GDP: 1.8-2.2万亿美元
                m2 = np.random.uniform(3000, 4000)     # M2: 3-4万亿美元
                exchange_rate = np.random.uniform(1100, 1300)  # 韩元/美元
            elif country == 'Australia':
                gdp = np.random.uniform(1500, 1800)    # GDP: 1.5-1.8万亿美元
                m2 = np.random.uniform(2000, 2500)     # M2: 2-2.5万亿美元
                exchange_rate = np.random.uniform(0.65, 0.75)  # 澳元/美元
            elif country == 'Spain':
                gdp = np.random.uniform(1400, 1600)    # GDP: 1.4-1.6万亿美元
                m2 = np.random.uniform(2000, 2500)     # M2: 2-2.5万亿美元
                exchange_rate = np.random.uniform(0.9, 1.1)  # 欧元/美元
            elif country == 'Mexico':
                gdp = np.random.uniform(1300, 1600)    # GDP: 1.3-1.6万亿美元
                m2 = np.random.uniform(2000, 2500)     # M2: 2-2.5万亿美元
                exchange_rate = np.random.uniform(18, 22)  # 比索/美元
            elif country == 'Indonesia':
                gdp = np.random.uniform(1200, 1500)    # GDP: 1.2-1.5万亿美元
                m2 = np.random.uniform(3000, 4000)     # M2: 3-4万亿美元
                exchange_rate = np.random.uniform(14000, 16000)  # 印尼卢比/美元
            elif country == 'Netherlands':
                gdp = np.random.uniform(1000, 1200)    # GDP: 1-1.2万亿美元
                m2 = np.random.uniform(1500, 2000)     # M2: 1.5-2万亿美元
                exchange_rate = np.random.uniform(0.9, 1.1)  # 欧元/美元
            elif country == 'Turkey':
                gdp = np.random.uniform(800, 1000)     # GDP: 0.8-1万亿美元
                m2 = np.random.uniform(1500, 2000)     # M2: 1.5-2万亿美元
                exchange_rate = np.random.uniform(20, 30)  # 里拉/美元
            elif country == 'Switzerland':
                gdp = np.random.uniform(800, 1000)     # GDP: 0.8-1万亿美元
                m2 = np.random.uniform(1000, 1500)     # M2: 1-1.5万亿美元
                exchange_rate = np.random.uniform(0.9, 1.1)  # 瑞郎/美元
            elif country == 'Saudi Arabia':
                gdp = np.random.uniform(800, 1000)     # GDP: 0.8-1万亿美元
                m2 = np.random.uniform(500, 800)       # M2: 0.5-0.8万亿美元
                exchange_rate = 3.75                   # 沙特里亚尔/美元（固定汇率）
            elif country == 'Argentina':
                gdp = np.random.uniform(400, 600)      # GDP: 0.4-0.6万亿美元
                m2 = np.random.uniform(800, 1200)      # M2: 0.8-1.2万亿美元
                exchange_rate = np.random.uniform(200, 300)  # 比索/美元
            else:
                # 默认值
                gdp = np.random.uniform(500, 1500)     # GDP: 0.5-1.5万亿美元
                m2 = np.random.uniform(1000, 2000)     # M2: 1-2万亿美元
                exchange_rate = np.random.uniform(1, 10)  # 汇率
            
            # 经济指标
            ppi = np.random.uniform(-2, 5)  # PPI: -2%到5%
            cpi = np.random.uniform(0, 10)  # CPI: 0%到10%
            
            row = {
                'Country': country,
                'Year': year,
                'GDP': gdp,
                'PPI': ppi,
                'CPI': cpi,
                'M2': m2,
                'ExchangeRate': exchange_rate
            }
            data.append(row)
    return pd.DataFrame(data)

# 生成数据
df = generate_country_data(countries, years, indicators)

# 由于是模拟数据，我们手动添加一些缺失值用于演示
# 在随机位置添加缺失值
np.random.seed(42)
for col in indicators:
    df.loc[np.random.choice(df.index, size=int(len(df)*0.05)), col] = np.nan

# 保存原始数据（包含缺失值）
original_data_path = 'country_year_feature.csv'
df.to_csv(original_data_path, index=False)
print(f'原始数据已保存到: {original_data_path}')

# 2. 数据清洗
print('\n=== 数据清洗 ===')

# 检查缺失值
print('\n缺失值检查:')
missing_values = df.isnull().sum()
print(missing_values)

# 缺失值填充：使用均值填充
print('\n缺失值填充（使用均值）:')
for col in indicators:
    mean_val = df[col].mean()
    df[col] = df[col].fillna(mean_val)
    print(f'{col} 均值: {mean_val:.2f}')

# 异常值检测：使用IQR方法
print('\n异常值检测（IQR方法）:')
for col in indicators:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f'{col} 异常值数量: {len(outliers)}')
    
    # 异常值处理：用中位数替换
    median_val = df[col].median()
    df.loc[(df[col] < lower_bound) | (df[col] > upper_bound), col] = median_val

# 重复值验证
print('\n重复值验证:')
duplicates = df.duplicated().sum()
print(f'重复行数量: {duplicates}')
if duplicates > 0:
    df = df.drop_duplicates()
    print(f'已删除 {duplicates} 条重复行')

# 3. 离差标准化（Min-Max标准化）
print('\n=== 离差标准化 ===')
normalized_dfs = []

for year in years:
    year_df = df[df['Year'] == year].copy()
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(year_df[indicators])
    normalized_df = pd.DataFrame(normalized_data, columns=indicators)
    normalized_df['Country'] = year_df['Country'].values
    normalized_df['Year'] = year_df['Year'].values
    normalized_dfs.append(normalized_df)

normalized_df = pd.concat(normalized_dfs)
normalized_df = normalized_df[['Country', 'Year'] + indicators]

# 保存标准化数据
normalized_data_path = 'Normalization.csv'
normalized_df.to_csv(normalized_data_path, index=False)
print(f'标准化数据已保存到: {normalized_data_path}')

# 4. 相关性分析
print('\n=== 相关性分析 ===')

# 计算所有指标的相关性矩阵
corr_matrix = df[indicators].corr()
print('\n相关性矩阵:')
print(corr_matrix)

# 绘制相关性热图
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('指标相关性矩阵')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300)
plt.close()

# 分析说明
print('\n相关性分析说明:')
print('1. 我们使用皮尔逊相关系数来衡量指标之间的线性关系')
print('2. 相关系数范围在[-1, 1]之间，绝对值越接近1表示相关性越强')
print('3. 正相关表示指标之间同向变化，负相关表示反向变化')
print('4. 我们可以看到不同指标之间的相关性程度，这有助于理解宏观经济变量之间的关系')

# 5. 使用变异系数法确定权重
print('\n=== 变异系数法确定权重 ===')

# 计算每个年度每个国家的变异系数
weights = []
for year in years:
    year_df = df[df['Year'] == year]
    # 计算每个指标的变异系数（标准差/均值）
    cv = year_df[indicators].std() / year_df[indicators].mean()
    # 归一化得到权重
    weights_cv = cv / cv.sum()
    # 转换为DataFrame
    weight_df = pd.DataFrame({'Indicator': indicators, 'Weight': weights_cv.values, 'Year': year})
    weights.append(weight_df)

weights_df = pd.concat(weights)

# 保存权重数据
weight_data_path = 'weight.csv'
weights_df.to_csv(weight_data_path, index=False)
print(f'权重数据已保存到: {weight_data_path}')

# 6. 计算宏观指数综合评分
print('\n=== 计算宏观指数综合评分 ===')

# 合并标准化数据和权重数据
merged_df = normalized_df.merge(weights_df, left_on=['Year'], right_on=['Year'])

# 计算每个国家每年的综合评分
scores = []
for year in years:
    year_df = merged_df[merged_df['Year'] == year]
    for country in countries:
        country_data = year_df[year_df['Country'] == country]
        if not country_data.empty:
            score = (country_data[indicators] * country_data['Weight'].values.reshape(-1, 1)).sum().sum()
            scores.append({'Country': country, 'Year': year, 'Score': score})

scores_df = pd.DataFrame(scores)

# 保存评分数据
scores_data_path = 'country_macro_score.csv'
scores_df.to_csv(scores_data_path, index=False)
print(f'宏观指数综合评分已保存到: {scores_data_path}')

# 显示结果
print('\n=== 最终结果展示 ===')
print('各国家2021-2023年宏观指数综合评分前10名:')
print(scores_df.sort_values('Score', ascending=False).head(10))

print('\n所有任务已完成！')
