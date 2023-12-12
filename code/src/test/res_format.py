import re

if __name__ == '__main__':

    data = """ main.py:234	line:234 -> logddd.log(prf) :  ({'recall': 0.5034095467308464, 'f1': 0.5185255027989225, 'precision': 0.5856154175897651},) Fri Dec  8 20:51:04 2023
 main.py:234	line:234 -> logddd.log(prf) :  ({'recall': 0.4239871640593662, 'f1': 0.42419396217177935, 'precision': 0.46501639521325044},) Fri Dec  8 21:01:41 2023
 main.py:234	line:234 -> logddd.log(prf) :  ({'recall': 0.5082230244685119, 'f1': 0.5037949585729129, 'precision': 0.5373228513290232},) Fri Dec  8 21:12:20 2023
 main.py:234	line:234 -> logddd.log(prf) :  ({'recall': 0.49097472924187724, 'f1': 0.48130916134087576, 'precision': 0.5143662841561194},) Fri Dec  8 21:22:49 2023
 main.py:234	line:234 -> logddd.log(prf) :  ({'recall': 0.46770958684316083, 'f1': 0.4286861024337976, 'precision': 0.4421155164576519},) Fri Dec  8 21:33:10 2023
 main.py:254	line:254 -> logddd.log(prf) :  ('当前train数量为:5',) Fri Dec  8 21:33:10 2023
 main.py:234	line:234 -> logddd.log(prf) :  ({'recall': 0.5499398315282792, 'f1': 0.5819697625277368, 'precision': 0.6761284811528561},) Fri Dec  8 21:44:45 2023
 main.py:234	line:234 -> logddd.log(prf) :  ({'recall': 0.5298836742880064, 'f1': 0.5531758002254482, 'precision': 0.5920700889475118},) Fri Dec  8 21:56:00 2023
 main.py:234	line:234 -> logddd.log(prf) :  ({'recall': 0.5499398315282792, 'f1': 0.525594348901462, 'precision': 0.5704898145321975},) Fri Dec  8 22:07:24 2023
 main.py:234	line:234 -> logddd.log(prf) :  ({'recall': 0.5531488166867228, 'f1': 0.5341263831078908, 'precision': 0.5596516141083535},) Fri Dec  8 22:19:11 2023
 main.py:234	line:234 -> logddd.log(prf) :  ({'recall': 0.5639791415964701, 'f1': 0.5632685353008064, 'precision': 0.6132376158539522},) Fri Dec  8 22:30:28 2023
 main.py:254	line:254 -> logddd.log(prf) :  ('当前train数量为:10',) Fri Dec  8 22:30:28 2023
 main.py:234	line:234 -> logddd.log(prf) :  ({'recall': 0.5655836341756919, 'f1': 0.6184725930190751, 'precision': 0.700871588517676},) Fri Dec  8 22:42:51 2023
 main.py:234	line:234 -> logddd.log(prf) :  ({'recall': 0.592057761732852, 'f1': 0.6164890363038749, 'precision': 0.6807249140473453},) Fri Dec  8 22:54:54 2023
 main.py:234	line:234 -> logddd.log(prf) :  ({'recall': 0.5812274368231047, 'f1': 0.6023478886800245, 'precision': 0.6426514802782204},) Fri Dec  8 23:07:28 2023
 main.py:234	line:234 -> logddd.log(prf) :  ({'recall': 0.5888487765744084, 'f1': 0.593172632057811, 'precision': 0.6133546505964452},) Fri Dec  8 23:19:17 2023
 main.py:234	line:234 -> logddd.log(prf) :  ({'recall': 0.5804251905334937, 'f1': 0.5930403381431816, 'precision': 0.6442932755078911},) Fri Dec  8 23:31:36 2023
 main.py:254	line:254 -> logddd.log(prf) :  ('当前train数量为:15',) Fri Dec  8 23:31:36 2023
 main.py:234	line:234 -> logddd.log(prf) :  ({'recall': 0.5844364219815483, 'f1': 0.6274423547260348, 'precision': 0.6848989230998378},) Fri Dec  8 23:44:35 2023
 main.py:234	line:234 -> logddd.log(prf) :  ({'recall': 0.6526273565984757, 'f1': 0.6441597332053722, 'precision': 0.6501890156971899},) Fri Dec  8 23:57:55 2023
 main.py:234	line:234 -> logddd.log(prf) :  ({'recall': 0.5912555154432411, 'f1': 0.6019125685892124, 'precision': 0.6419491720204736},) Sat Dec  9 00:10:33 2023
 main.py:234	line:234 -> logddd.log(prf) :  ({'recall': 0.586843160850381, 'f1': 0.59651749878501, 'precision': 0.6561386259678273},) Sat Dec  9 00:23:28 2023
 main.py:234	line:234 -> logddd.log(prf) :  ({'recall': 0.5820296831127156, 'f1': 0.6110683815238772, 'precision': 0.6573389418145554},) Sat Dec  9 00:36:47 2023
 main.py:254	line:254 -> logddd.log(prf) :  ('当前train数量为:20',) Sat Dec  9 00:36:47 2023
 main.py:234	line:234 -> logddd.log(prf) :  ({'recall': 0.6205375050140393, 'f1': 0.6283245057807609, 'precision': 0.6657825299168254},) Sat Dec  9 00:50:18 2023
 main.py:234	line:234 -> logddd.log(prf) :  ({'recall': 0.5864420377055756, 'f1': 0.5724703845537957, 'precision': 0.5782049123980716},) Sat Dec  9 01:04:28 2023
 main.py:234	line:234 -> logddd.log(prf) :  ({'recall': 0.6133172884075411, 'f1': 0.6433007270883723, 'precision': 0.7118279434281629},) Sat Dec  9 01:18:11 2023
 main.py:234	line:234 -> logddd.log(prf) :  ({'recall': 0.6446048937023666, 'f1': 0.6544035992579837, 'precision': 0.7100911541486051},) Sat Dec  9 01:32:06 2023
 main.py:234	line:234 -> logddd.log(prf) :  ({'recall': 0.5848375451263538, 'f1': 0.6277757665587488, 'precision': 0.7114684135808478},) Sat Dec  9 01:45:46 2023
 main.py:254	line:254 -> logddd.log(prf) :  ('当前train数量为:25',) Sat Dec  9 01:45:46 2023
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
