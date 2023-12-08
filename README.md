# 基于隐马尔可夫的提示调优词性标注方法

## 基本相关知识

1. 当前研究是在few-shot场景下进行研究的，因此，训练数据集有5-shot、10-shot、15-shot、20-shot、25-shot多组（n-shot表示训练数据是n条，一条数据代表的是数据集中的一个句子，例如：肘痛，按压时）
2. 当前实验采用五折交叉验证
3. 当前研究使用了三个数据集：中医切诊描述数据集、CTB8.0数据集、UD数据集
4. 当前研究的数据集不是在训练的时候划分的，为了达到对比效果，数据集都是提前划分好的，划分为训练集和测试集
5. 数据集的划分是，先将整体数据集划分为训练集和测试集，再在训练集中采样5n条作为n-shot。（5n是因为5折交叉验证）

## 项目结构

```
└─📁prompt
    └─📁code ✅ 目前所有代码都在当前目录下
        ├─📁data 🧪存放当前的实验数据
        │  └─📁ctb ctb8.0数据集
        │  │  └─📁split_data 数据集划分保存的地方
        │  │  │  └─📁few-shot
        │  │  │  │  └─📁fold 存放[基本相关知识1]中提到的，以及划分好的数据集。里面有两类文件：n.data和n_ids.data
        │  │  │  │  │  └─📃5.data 表示这是训练样本为5条时的训练数据
        │  │  │  │  │  └─📃5_ids.data 为了后续对结果分析更加方便，存储了当前划分的这些数据在原数据集中的id
        │  │  │  │  │  └─📃。。。。。。
        │  │  │  │  └─📃ctb_test.data 测试集
        │  │  │  │  └─📃ctb_train.data 训练集
        │  │  │  │  └─📃one_tentn_test_datas.data 十分之一测试集
        │  │  │  │  └─📃one_tentn_test_datas.data 十分之一测试集对应id
        │  │  │  │  └─📃。。。。。。。 下同
        │  │  │  └─📃totaldata.txt CTB8.0所有的数据
        │  └─📁jw ❌这个是任务一开始的时候王老师发过来的数据集，暂时没有用这边的
        │  └─📁split_data 这个是《中医切诊描述数据集》 这个才是蒋文的数据集，结构同ctb。
        │  └─📁UD数据集 结构同上
        │  └─📁wechat 王老师微信发给我的数据集，没用上，不知道是什么数据集
        │  └─📃.py data_status.py 统计数据信息的文件
    └─📁model 当前预训练模型存放的位置
        ├─📁bart-large bart模型存放的位置
        │  └─📃。。。 正常预训练模型的配置
        │  └─📃vocab.txt 预训练模型当前需要加载的词表，根据不同的数据集要进行一定的切换
        │  └─📃vocab_ctb.txt CTB数据集的词表，当要跑CTB数据集时，只需要执行命令 `cp vocab_ctb.txt vocab.txt`,将数据集切换过来就行
        │  └─📃vocab_ud.txt  UD数据集词表，同上
        │  └─📃vocab_jw.txt 中医切诊描述数据集词表，同上
        ├─📁bert_large_chinese bert模型存放的位置
        │  └─📃。。。 正常预训练模型的配置
        │  └─📃vocab.txt 预训练模型当前需要加载的词表，根据不同的数据集要进行一定的切换
        │  └─📃vocab_ctb.txt CTB数据集的词表，当要跑CTB数据集时，只需要执行命令 `cp vocab_ctb.txt vocab.txt`,将数据集切换过来就行
        │  └─📃vocab_ud.txt  UD数据集词表，同上
        │  └─📃vocab_jw.txt 中医切诊描述数据集词表，同上
        ├─📁medbert Mc-BERT模型存放的位置(王老师叫的是medbert，是李凯伦使用医学信息预训练的预训练模型，实际上叫Mc-BERT)
        │  └─📃。。。 正常预训练模型的配置
        │  └─📃vocab.txt 预训练模型当前需要加载的词表，根据不同的数据集要进行一定的切换
        │  └─📃vocab_ctb.txt CTB数据集的词表，当要跑CTB数据集时，只需要执行命令 `cp vocab_ctb.txt vocab.txt`,将数据集切换过来就行
        │  └─📃vocab_ud.txt  UD数据集词表，同上
        │  └─📃vocab_jw.txt 中医切诊描述数据集词表，同上
    └─📁src ✅源代码位置
        ├─📁bilstm_crf 对比模型bilstm_crf对比实验文件(❗️未工程化)
        │  └─📃bilstm_crf.py 模型训练预测的启动类 run的时候直接 python bilstm_crf.py一把梭哈
        │  └─📃models.py bilstm_crf模型代码 写好了，不用看
        │  └─📃model_params.py 模型run的时候的参数，需要改数据集，改预训练模型都在这儿，直接改        
        ├─📁crf 对比模型crf对比实验文件(❗️未工程化)
        │  └─📃crf.py 模型训练预测的启动类 run的时候直接 python crf.py一把梭哈
        │  └─📃models.py crf模型代码 写好了，不用看
        │  └─📃model_params.py 模型run的时候的参数，需要改数据集，改预训练模型都在这儿，直接改        
        ├─📁prompt 当前模型代码
        │  └─📃main.py 模型训练预测的启动类 run的时候直接 python main.py一把梭哈
        │  └─📃models.py HMM-Prompt模型代码 
        │  └─📃model_params.py 模型run的时候的参数，需要改数据集，改预训练模型都在这儿，直接改
        ├─📁data_process 不同数据集有不同的格式，这个包是根据不同数据集加载数据，然后统一成一个格式 [中国/经济/简讯，‘NR/NN/NN’].(因为数据已经统一好格式，划分好了，所以不会被调用，但是后续需要加数据集可以这样写)
        │  └─📃data_processing.py 提供上层数据加载调用，提示模板也写在这个里面，在这里面进行模板填充
        │  └─📃utils.py 计算loss、划分batch
        │  └─📃xxx_2_standard.py xxx数据转换为标准数据格式的代码
        ├─📁nohmm 消融实验，不加HMM
        │  └─📃main.py 同上
        │  └─📃models.py 同上
        │  └─📃model_params.py 同上
        ├─📁split_data 数据划分代码
        │  └─📃split_data.py 数据划分代码
        ├─📁test 测试代码
        ├─📃predict.py 预测时的代码，对外暴露test_model()， nohmm\prompt中用
        ├─📃start_tensorboard.sh 开启tensorboard
        
        
        
         
        
        
        
        
         
        
        
    
        
        
        
        

    └─original_dataset  数据集的原始形式（会包含原始数据集的处理，比如原始数据集的合并，️❌目前没用）
    └─pull.sh 从GitHub同步代码下来的脚本
    └─submit.sh 向GitHub提交代码脚本     
```
