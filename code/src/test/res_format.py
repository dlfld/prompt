import re

if __name__ == '__main__':

    data = """  main.py:221	line:221 -> logddd.log(prf) :  ({'recall': 0.6470216796463902, 'f1': 0.6026219411698335, 'precision': 0.6274852240154333, 'acc': 0.6470216796463902},) Wed Jan 10 16:37:47 2024
 main.py:221	line:221 -> logddd.log(prf) :  ({'recall': 0.7259524310671438, 'f1': 0.700024219282642, 'precision': 0.7147368676747863, 'acc': 0.7259524310671438},) Wed Jan 10 18:03:42 2024
 main.py:221	line:221 -> logddd.log(prf) :  ({'recall': 0.5990317827825721, 'f1': 0.5415783141092675, 'precision': 0.5448441803707785, 'acc': 0.5990317827825721},) Wed Jan 10 19:24:06 2024
 main.py:221	line:221 -> logddd.log(prf) :  ({'recall': 0.650389391707009, 'f1': 0.6259118427144486, 'precision': 0.6389194497324423, 'acc': 0.650389391707009},) Wed Jan 10 20:48:14 2024
 main.py:221	line:221 -> logddd.log(prf) :  ({'recall': 0.620500947169017, 'f1': 0.5634064524856787, 'precision': 0.6350102308041902, 'acc': 0.620500947169017},) Wed Jan 10 22:08:52 2024
 main.py:233	line:233 -> logddd.log(prf) :  ('当前train数量为:5',) Wed Jan 10 22:08:52 2024
 main.py:221	line:221 -> logddd.log(prf) :  ({'recall': 0.7848873921279731, 'f1': 0.7703665514185749, 'precision': 0.7764673084453012, 'acc': 0.7848873921279731},) Wed Jan 10 23:38:03 2024
 main.py:221	line:221 -> logddd.log(prf) :  ({'recall': 0.7253209850557777, 'f1': 0.70146550274277, 'precision': 0.7022484389877534, 'acc': 0.7253209850557777},) Thu Jan 11 01:03:38 2024
 main.py:221	line:221 -> logddd.log(prf) :  ({'recall': 0.7219532729951589, 'f1': 0.6991139052264508, 'precision': 0.7418325304950368, 'acc': 0.7219532729951589},) Thu Jan 11 02:31:09 2024
 main.py:221	line:221 -> logddd.log(prf) :  ({'recall': 0.7255314670595664, 'f1': 0.6894078759283213, 'precision': 0.7288865556546403, 'acc': 0.7255314670595664},) Thu Jan 11 03:57:52 2024
 main.py:221	line:221 -> logddd.log(prf) :  ({'recall': 0.7192170069459062, 'f1': 0.6834660123107948, 'precision': 0.6945086201539743, 'acc': 0.7192170069459062},) Thu Jan 11 05:24:03 2024
 main.py:233	line:233 -> logddd.log(prf) :  ('当前train数量为:10',) Thu Jan 11 05:24:03 2024
 main.py:221	line:221 -> logddd.log(prf) :  ({'recall': 0.7762576299726374, 'f1': 0.7732546023767957, 'precision': 0.7831416412475074, 'acc': 0.7762576299726374},) Thu Jan 11 07:02:29 2024
 main.py:221	line:221 -> logddd.log(prf) :  ({'recall': 0.7718375078930751, 'f1': 0.7659392030865767, 'precision': 0.7810358454216663, 'acc': 0.7718375078930751},) Thu Jan 11 08:34:22 2024
 main.py:221	line:221 -> logddd.log(prf) :  ({'recall': 0.7764681119764261, 'f1': 0.7571817690686502, 'precision': 0.7764804570161664, 'acc': 0.7764681119764261},) Thu Jan 11 10:10:16 2024
 main.py:221	line:221 -> logddd.log(prf) :  ({'recall': 0.7754157019574827, 'f1': 0.760541357532654, 'precision': 0.7728931295825703, 'acc': 0.7754157019574827},) Thu Jan 11 11:45:38 2024
 main.py:221	line:221 -> logddd.log(prf) :  ({'recall': 0.7880446221848032, 'f1': 0.7815933200782977, 'precision': 0.7862040683420511, 'acc': 0.7880446221848032},) Thu Jan 11 13:20:27 2024
 main.py:233	line:233 -> logddd.log(prf) :  ('当前train数量为:15',) Thu Jan 11 13:20:27 2024
 main.py:221	line:221 -> logddd.log(prf) :  ({'recall': 0.8227741528099347, 'f1': 0.8184842385714455, 'precision': 0.8245131637169548, 'acc': 0.8227741528099347},) Thu Jan 11 14:59:31 2024
 main.py:221	line:221 -> logddd.log(prf) :  ({'recall': 0.8086718585560935, 'f1': 0.7931634274277021, 'precision': 0.785641396272808, 'acc': 0.8086718585560935},) Thu Jan 11 16:38:13 2024
 main.py:221	line:221 -> logddd.log(prf) :  ({'recall': 0.7943590822984635, 'f1': 0.788550527397879, 'precision': 0.8070967982984443, 'acc': 0.7943590822984635},) Thu Jan 11 18:21:40 2024
 main.py:221	line:221 -> logddd.log(prf) :  ({'recall': 0.8065670385182067, 'f1': 0.7945715791553025, 'precision': 0.7950684205809946, 'acc': 0.8065670385182067},) Thu Jan 11 19:58:42 2024
 main.py:221	line:221 -> logddd.log(prf) :  ({'recall': 0.7859398021469164, 'f1': 0.7731354049335883, 'precision': 0.7936735879148756, 'acc': 0.7859398021469164},) Thu Jan 11 21:40:27 2024
 main.py:233	line:233 -> logddd.log(prf) :  ('当前train数量为:20',) Thu Jan 11 21:40:27 2024
 main.py:221	line:221 -> logddd.log(prf) :  ({'recall': 0.814144390654599, 'f1': 0.8109808053691081, 'precision': 0.8124875316656599, 'acc': 0.814144390654599},) Thu Jan 11 23:29:19 2024
 main.py:221	line:221 -> logddd.log(prf) :  ({'recall': 0.8358240370448327, 'f1': 0.8294003720658268, 'precision': 0.8296330757557451, 'acc': 0.8358240370448327},) Fri Jan 12 01:24:17 2024
 main.py:221	line:221 -> logddd.log(prf) :  ({'recall': 0.8242475268364555, 'f1': 0.8091460055719852, 'precision': 0.8013888894921857, 'acc': 0.8242475268364555},) Fri Jan 12 03:14:39 2024
 main.py:221	line:221 -> logddd.log(prf) :  ({'recall': 0.8154072826773311, 'f1': 0.8064860945033411, 'precision': 0.8184615165798959, 'acc': 0.8154072826773311},) Fri Jan 12 05:01:27 2024
 main.py:221	line:221 -> logddd.log(prf) :  ({'recall': 0.8217217427909914, 'f1': 0.8156179473993097, 'precision': 0.8213690090331093, 'acc': 0.8217217427909914},) Fri Jan 12 06:40:21 2024
 main.py:233	line:233 -> logddd.log(prf) :  ('当前train数量为:25',) Fri Jan 12 06:40:21 2024
    """
    items = data.split("\n")
    total_res = []
    for item in items:
        if item.strip() == "":
            continue
        pattern = r"当前train数量为:(\d+)"
        match = re.search(pattern, item)
        # 匹配 bilstm_crf.py:360	line:360 -> logddd.log(prf) :  ('当前train数量为:5',) Wed Jul 26 14:13:28 2023 这个句子，

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
