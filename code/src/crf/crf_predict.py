import logddd
import torch
from code.src.crf.crf import load_model, load_instance_data
from code.src.crf.model_params import Config


def status_muti():
    # 返回数据集中的兼类词
    """
     统计的兼类词问题
    """
    with open("/Users/dailinfeng/Desktop/prompt/code/data/jw/after_pos_seg.txt", "r") as f:
        readlines = f.readlines()
        labels = set()
        words_dict = dict({})
        for line in readlines:
            label = line.split("    ")[1].split(" ")
            words = line.split("    ")[0].split(" ")

            for index, word in enumerate(words):
                if word in words_dict:
                    if label[index] not in words_dict[word]["label"]:
                        words_dict[word]["label"].add(label[index])
                        words_dict[word]["sentence"].append(line)

                else:
                    words_dict[word] = {
                        "label": set([label[index]]),
                        "sentence": [line]
                    }

        count = 0
        res = []
        for key in words_dict:
            if len(words_dict[key]["sentence"]) > 1:
                count = count + len(words_dict[key]["sentence"])
                print(key, end=" ")
                print(words_dict[key]["sentence"])
                res.append([x.replace("\n", "") for x in words_dict[key]["sentence"]])
                print()

        print(count)
        return res


model_checkpoint = ""
if __name__ == '__main__':
    _, tokenizer = load_model(model_checkpoint)
    model = torch.load(Config.continue_plm_file)
    # 获取所有兼类词列表
    all_multi_category = status_muti()
    for words in all_multi_category:
        instance_data = load_instance_data(words, tokenizer, Config, False)
        # 加载模型
        model = torch.load(Config.continue_plm_file)
        loss, paths = model(instance_data)
        logddd.log(paths)
