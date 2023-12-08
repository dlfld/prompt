import re

if __name__ == '__main__':

    data = """ crf.py:307	line:307 -> logddd.log(prf) :  ({'recall': 0.22136197136197136, 'f1': 0.1687022333773607, 'precision': 0.1494103232188675},) Thu Nov 30 16:23:45 2023
 crf.py:307	line:307 -> logddd.log(prf) :  ({'recall': 0.43997668997669, 'f1': 0.3988054491442897, 'precision': 0.4815849143935775},) Thu Nov 30 16:26:08 2023
 crf.py:307	line:307 -> logddd.log(prf) :  ({'recall': 0.44755244755244755, 'f1': 0.41221102669159265, 'precision': 0.44673750575660964},) Thu Nov 30 16:28:31 2023
 crf.py:307	line:307 -> logddd.log(prf) :  ({'recall': 0.4244089244089244, 'f1': 0.39565208328677853, 'precision': 0.4438961409755963},) Thu Nov 30 16:30:53 2023
 crf.py:307	line:307 -> logddd.log(prf) :  ({'recall': 0.46703296703296704, 'f1': 0.41783451465889915, 'precision': 0.4594083923050894},) Thu Nov 30 16:33:14 2023
 crf.py:327	line:327 -> logddd.log(prf) :  ('当前train数量为:5',) Thu Nov 30 16:33:14 2023
 crf.py:307	line:307 -> logddd.log(prf) :  ({'recall': 0.5234765234765235, 'f1': 0.49395711497566674, 'precision': 0.517038313363241},) Thu Nov 30 16:35:36 2023
 crf.py:307	line:307 -> logddd.log(prf) :  ({'recall': 0.5452047952047953, 'f1': 0.5332301112669902, 'precision': 0.5645572852184875},) Thu Nov 30 16:37:58 2023
 crf.py:307	line:307 -> logddd.log(prf) :  ({'recall': 0.5626873126873126, 'f1': 0.5331998444919079, 'precision': 0.5345887052720454},) Thu Nov 30 16:40:21 2023
 crf.py:307	line:307 -> logddd.log(prf) :  ({'recall': 0.5145687645687645, 'f1': 0.4896977839452225, 'precision': 0.5201716630324363},) Thu Nov 30 16:42:43 2023
 crf.py:307	line:307 -> logddd.log(prf) :  ({'recall': 0.543040293040293, 'f1': 0.5128477966874041, 'precision': 0.5141886722424264},) Thu Nov 30 16:45:05 2023
 crf.py:327	line:327 -> logddd.log(prf) :  ('当前train数量为:10',) Thu Nov 30 16:45:05 2023
 crf.py:307	line:307 -> logddd.log(prf) :  ({'recall': 0.582001332001332, 'f1': 0.5542087702261205, 'precision': 0.573904930095394},) Thu Nov 30 16:47:29 2023
 crf.py:307	line:307 -> logddd.log(prf) :  ({'recall': 0.5876623376623377, 'f1': 0.5553272160889181, 'precision': 0.5579536504404379},) Thu Nov 30 16:49:52 2023
 crf.py:307	line:307 -> logddd.log(prf) :  ({'recall': 0.5454545454545454, 'f1': 0.5195301550165526, 'precision': 0.5270361359582862},) Thu Nov 30 16:52:16 2023
 crf.py:307	line:307 -> logddd.log(prf) :  ({'recall': 0.5699300699300699, 'f1': 0.550273717912645, 'precision': 0.5653702809668647},) Thu Nov 30 16:54:40 2023
 crf.py:307	line:307 -> logddd.log(prf) :  ({'recall': 0.5778388278388278, 'f1': 0.5687435878871394, 'precision': 0.5827810116856414},) Thu Nov 30 16:57:04 2023
 crf.py:327	line:327 -> logddd.log(prf) :  ('当前train数量为:15',) Thu Nov 30 16:57:04 2023
 crf.py:307	line:307 -> logddd.log(prf) :  ({'recall': 0.6373626373626373, 'f1': 0.6223046148848883, 'precision': 0.6424524116208267},) Thu Nov 30 16:59:32 2023
 crf.py:307	line:307 -> logddd.log(prf) :  ({'recall': 0.5886613386613386, 'f1': 0.5807955391778546, 'precision': 0.5986163649408208},) Thu Nov 30 17:02:00 2023
 crf.py:307	line:307 -> logddd.log(prf) :  ({'recall': 0.5724275724275725, 'f1': 0.554542784625125, 'precision': 0.5636733103995518},) Thu Nov 30 17:04:28 2023
 crf.py:307	line:307 -> logddd.log(prf) :  ({'recall': 0.6226273726273727, 'f1': 0.6189417896883179, 'precision': 0.6404573720967528},) Thu Nov 30 17:06:56 2023
 crf.py:307	line:307 -> logddd.log(prf) :  ({'recall': 0.6411921411921412, 'f1': 0.6221434392467685, 'precision': 0.6443765578765056},) Thu Nov 30 17:09:24 2023
 crf.py:327	line:327 -> logddd.log(prf) :  ('当前train数量为:20',) Thu Nov 30 17:09:24 2023
 crf.py:307	line:307 -> logddd.log(prf) :  ({'recall': 0.6023976023976024, 'f1': 0.5827935625814965, 'precision': 0.5866023181144098},) Thu Nov 30 17:11:56 2023
 crf.py:307	line:307 -> logddd.log(prf) :  ({'recall': 0.5714285714285714, 'f1': 0.5560653422752162, 'precision': 0.5626062202387427},) Thu Nov 30 17:14:27 2023
 crf.py:307	line:307 -> logddd.log(prf) :  ({'recall': 0.632950382950383, 'f1': 0.6230050604556481, 'precision': 0.6367971214856712},) Thu Nov 30 17:16:59 2023
 crf.py:307	line:307 -> logddd.log(prf) :  ({'recall': 0.6214618714618715, 'f1': 0.6122836324535496, 'precision': 0.6278488810933525},) Thu Nov 30 17:19:31 2023
 crf.py:307	line:307 -> logddd.log(prf) :  ({'recall': 0.6295371295371296, 'f1': 0.617455936725274, 'precision': 0.6243047172772652},) Thu Nov 30 17:22:02 2023
 crf.py:327	line:327 -> logddd.log(prf) :  ('当前train数量为:25',) Thu Nov 30 17:22:02 2023
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
