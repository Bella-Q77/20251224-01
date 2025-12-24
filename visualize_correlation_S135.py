import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取correlation_results.json文件
with open('correlation_results.json', 'r') as f:
    correlation_data = json.load(f)

# 对每个年份的相关性矩阵进行可视化
for year, corr_dict in correlation_data.items():
    # 转换为DataFrame
    corr_matrix = pd.DataFrame(corr_dict)
    
    # 创建热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title(f'{year}年宏观指标相关性矩阵热力图')
    plt.xlabel('指标')
    plt.ylabel('指标')
    
    # 保存图片
    plt.savefig(f'correlation_heatmap_{year}.png', dpi=300, bbox_inches='tight')
    print(f'已生成{year}年相关性热力图：correlation_heatmap_{year}.png')
    
    # 显示图片（可选）
    # plt.show()
    plt.close()

print('所有相关性热力图已生成完成！')