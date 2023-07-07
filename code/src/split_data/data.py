if __name__ == '__main__':
    import joblib
    traindata = joblib.load("/home/dlf/prompt/code/data/ctb/split_data/few_shot/ctb_train.data")
    max_len = 0
    sentence = []
    for item in traindata:
        if len(item[0]) > max_len:
            max_len = len(item[0])
            sentence = item
    import logddd
    logddd.log(sentence)
    logddd.log(len(sentence[0].split("/")))