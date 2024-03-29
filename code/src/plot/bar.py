import matplotlib.pyplot as plt
import numpy as np

# 设置绘图风格
plt.style.use('ggplot')
# 处理中文乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# 取出城市名称
train_sizes = [5, 10, 15, 20, 25, 50, 75, 100, 200, 500]
models = ["CRF", "BiLSTM+CRF", "Prompt"]
# BERT
# p-tuning = [0.338876932, 0.514122577, 0.473390627, 0.588389468, 0.421102563, 0.573375685, 0.597694508, 0.710579865,
#        0.746300672, 0.660090527]
# bilstm_crf = [0.21848142, 0.157984241, 0.174556644, 0.255131488, 0.152710691, 0.243304169, 0.355899065, 0.34076533,
#               0.716460724, 0.76920203]
# p-tuning = [0.555619927, 0.728946049, 0.749752724, 0.767554578, 0.775195795, 0.811235511, 0.834981972, 0.840481661,
#           0.866761853, 0.889277739]


# MedBERT
# p-tuning = [0.216605965, 0.370935152, 0.42731971, 0.484238249, 0.432457966, 0.575287406, 0.628628132, 0.652827334,
#        0.708933316, 0.755626688]
# bilstm_crf = [0.176209631, 0.1676475, 0.195684206, 0.296526471, 0.279780293, 0.443146815, 0.552543094, 0.641947146,
#               0.707135921, 0.752753699]
# p-tuning = [0.366279324, 0.58270802, 0.655982184, 0.703158039, 0.712354588, 0.762442502, 0.791948947, 0.798728745,
#           0.83676061, 0.869888042]


# BART
crf = [0.225759436, 0.25193006, 0.275150319, 0.283955899, 0.280530199, 0.309642601, 0.322004205, 0.374642014,
       0.468966182, 0.63423463]
bilstm_crf = [0.233930642, 0.219406144, 0.228020655, 0.246726856, 0.266973882, 0.277493491, 0.287957097, 0.316191593,
              0.346951939, 0.557557065]
prompt = [0.402583391, 0.496534746, 0.54124772, 0.554707047, 0.560488367, 0.637539624, 0.666822406, 0.697963214,
          0.745667176, 0.793285198]
# 绘制水平交错条形图
bar_width = 0.2
plt.bar(x=np.arange(len(train_sizes)), height=bilstm_crf, label='BiLSTM-CRF', color='indianred',
        width=bar_width)
plt.bar(x=np.arange(len(train_sizes)) + bar_width, height=crf, label='CRF', color='steelblue', width=bar_width)
plt.bar(x=np.arange(len(train_sizes)) + bar_width * 2, height=prompt, label='Prompt', color='limegreen',
        width=bar_width)
plt.xticks(ticks=np.arange(len(train_sizes)), labels=train_sizes)
# 添加x轴刻度标签（向右偏移0.225）
# plt.xticks(np.arange(5) + 0.2, Cities)
# 添加y轴标签
plt.ylabel('f1')
plt.xlabel('train_sizes')
# 添加图形标题
plt.title('PLM:BART')
# 添加图例
plt.legend()
# 显示图形
plt.show()
