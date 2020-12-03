import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
# 设置plt正确显示中文
from matplotlib.font_manager import FontProperties
my_font = FontProperties(fname=r'/Users/zhanglulu/Library/Fonts/msyh.ttf')
sns.set(font=my_font.get_name())  

# 数据加载，避免中文乱码问题
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
data_ori = pd.read_csv('/Users/zhanglulu/Desktop/极客_数据分析实战/EM_data-master/heros.csv', encoding='gb18030')

# 数据探索
print(data_ori.head())
print(data_ori.info())
print(data_ori['攻击范围'].value_counts())

features = [u'最大生命',u'生命成长',u'初始生命',u'最大法力', u'法力成长',u'初始法力',u'最高物攻',u'物攻成长',u'初始物攻',
            u'最大物防',u'物防成长',u'初始物防', u'最大每5秒回血', u'每5秒回血成长', u'初始每5秒回血', u'最大每5秒回蓝', 
            u'每5秒回蓝成长', u'初始每5秒回蓝', u'最大攻速', u'攻击范围']
data = data_ori[features]
# 对英雄属性之间的关系进行可视化分析
# 用热力图呈现features字段之间的相关性
corr = data[features].corr()
plt.figure(figsize=(14,14))
# annot=True显示每个方格的数据
sns.heatmap(corr, annot=True)
plt.show()

# 相关性大的属性只选一个，因此对属性进行降维
features_remain = [u'最大生命', u'初始生命', u'最大法力', u'最高物攻', u'初始物攻', u'最大物防', u'初始物防', 
                   u'最大每5秒回血', u'最大每5秒回蓝', u'初始每5秒回蓝', u'最大攻速', u'攻击范围']
data = data_ori[features_remain]
data[u'最大攻速'] = data[u'最大攻速'].apply(lambda x:float(x.strip('%'))/100)
data[u'攻击范围'] = data[u'攻击范围'].map({'近战':0, '远程':1})
# 采用Z_Score规范化数据，保证每个特征维度的数据均值为0，方差为1
ss = StandardScaler()
data = ss.fit_transform(data)

# 构造GMM聚类
gmm = GaussianMixture(n_components=30, covariance_type='full')
gmm.fit(data)
# 训练数据
prediction = gmm.predict(data)
# 将分组结果输出到csv文件中
data_ori.insert(0, '分组', prediction)
data_ori.to_csv('./hero_out.csv', index=False, sep=',')

from sklearn.metrics import calinski_harabasz_score
print(calinski_harabasz_score(data, prediction))
