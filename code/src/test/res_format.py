import re

if __name__ == '__main__':

    data = """ main.py:228	line:228 -> logddd.log(prf) :  ({'recall': 0.7321873657069188, 'f1': 0.7446919844384317, 'precision': 0.7643336452517406},) Sat Oct 28 20:54:47 2023
 main.py:228	line:228 -> logddd.log(prf) :  ({'recall': 0.7735281478298238, 'f1': 0.7760462165693364, 'precision': 0.7897660984034257},) Sun Oct 29 00:27:24 2023
 main.py:228	line:228 -> logddd.log(prf) :  ({'recall': 0.7580575848732274, 'f1': 0.7587008102126477, 'precision': 0.7641576827472449},) Sun Oct 29 05:35:43 2023
 main.py:228	line:228 -> logddd.log(prf) :  ({'recall': 0.7390631714654061, 'f1': 0.7351150385507215, 'precision': 0.7498420268374988},) Sun Oct 29 09:01:52 2023
 main.py:228	line:228 -> logddd.log(prf) :  ({'recall': 0.7471422432316287, 'f1': 0.7529165060050949, 'precision': 0.7737806621002012},) Sun Oct 29 12:23:28 2023
 main.py:248	line:248 -> logddd.log(prf) :  ('当前train数量为:25',) Sun Oct 29 12:23:28 2023
    """
    items = data.split("\n")
    total_res = []
    for item in items:
        if item.strip() == "":
            continue
        pattern = r"当前train数量为:(\d+)"
        match = re.search(pattern, item)
        # 匹配 bert_bilstm_crf.py:360	line:360 -> logddd.log(prf) :  ('当前train数量为:5',) Wed Jul 26 14:13:28 2023 这个句子，

        if match is not None:
            matched_str = match.group(1)
            train_count = int(matched_str)
            # 在下面记录上当前的train_count
            total_res.append([train_count, 0, 0])
            total_res.append([])
            continue

        pattern = r"'recall': ([0-9.]*)"
        match = re.search(pattern, item)
        recall = match.group(1)

        pattern = r"'f1': ([0-9.]*)"
        match = re.search(pattern, item)
        f1 = match.group(1)

        pattern = r"'precision': ([0-9.]*)"
        match = re.search(pattern, item)
        precision = match.group(1)
        total_res.append([precision, recall, f1])

    import csv

    with open("resres.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(total_res)

    #     item = item.split(":  (")[1].split(",)")[0]
    #     lines = item.replace("{", "").replace("}", "").split(",")
    #     lines[0], lines[1], lines[2] = lines[2], lines[0], lines[1]
    #     res = []
    #     for line in lines:
    #         res.append(str(line.split(": ")[1]))
    #     total_res.append(res)
    #
    # total_res.append([])
    #
    # import csv
    # with open("resres.csv","a") as f:
    #         writer = csv.writer(f)
    #         writer.writerows(total_res)
