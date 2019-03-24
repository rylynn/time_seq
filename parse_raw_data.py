from pandas import read_csv
from matplotlib import pyplot
# 加载数据集
dataset = read_csv('raw_data.csv', header=0, index_col=0)
values = dataset.values
# 指定要绘制的列
groups = [ 1, 2, 3, 5, 6, 7, 8]
i = 1
# 绘制每一列
pyplot.figure()
for group in groups:
    pyplot.subplot(len(groups), 1, i)
    pyplot.plot(values[:, group])
    pyplot.title(dataset.columns[group], y=0.5, loc='right')
    i += 1
pyplot.show()
