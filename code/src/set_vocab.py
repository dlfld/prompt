if __name__ == "__main__":
    with open("/home/dlf/prompt/code/model/bert_large_chinese/vocab.txt","r", encoding='utf-8') as f:
        lines = f.readlines()
    res = set()
    for line in lines:
        res.add(line.strip()+"\n")
    with open("/home/dlf/prompt/vocab.txt","w", encoding='utf-8') as f:
        f.writelines(list(res))
    