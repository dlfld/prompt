import pycrfsuite


# 定义特征提取函数
def extract_features(sentence):
    features = []
    for word in sentence:
        # 提取当前词的特征
        word_features = [
            # 添加你的特征
        ]
        features.append(word_features)
    return features


# 定义标签提取函数
def extract_labels(sentence):
    return [label for _, label in sentence]



# 加载训练数据
train_data = [
    # 添加你的训练数据，每个句子是一个由（词，标签）对组成的列表
    [
        ["脉","NR"],
        ["细","VA"],
        ["弱","VA"],
        ["右","LC"],
        ["细","VA"],
        ["弦","VA"],
    ]
]

# 提取训练特征和标签
X_train = [extract_features(sentence) for sentence in train_data]
y_train = [extract_labels(sentence) for sentence in train_data]

# 创建CRF模型
crf = pycrfsuite.Trainer(verbose=False)

# 添加训练数据到模型
for x, y in zip(X_train, y_train):
    crf.append(x, y)

# 设置模型参数
crf.set_params({
    'c1': 1.0,  # L1正则化参数
    'c2': 1e-3,  # L2正则化参数
    'max_iterations': 50,
    'feature.possible_transitions': True
})

# 训练CRF模型
crf.train('model.crfsuite')

# 加载测试数据
test_data = [
    # 添加你的测试数据，每个句子是一个由（词，标签）对组成的列表
    [
        ["脉", "NR"],
        ["细", "VA"],
        ["弱", "VA"],
        ["右", "LC"],
        ["细", "VA"],
        ["弦", "VA"],
    ]
]

# 提取测试特征和标签
X_test = [extract_features(sentence) for sentence in test_data]
y_test = [extract_labels(sentence) for sentence in test_data]

# 创建标注器
tagger = pycrfsuite.Tagger()
tagger.open('model.crfsuite')

# 预测标签
y_pred = [tagger.tag(x) for x in X_test]

# 打印预测结果
for i, sentence in enumerate(test_data):
    print('Sentence:', ' '.join([word for word, _ in sentence]))
    print('Predicted labels:', ' '.join(y_pred[i]))
    print('True labels:     ', ' '.join(y_test[i]))
    print()
