import re

if __name__ == '__main__':

    data = """ main.py:189	line:189 -> logddd.log(prf) :  ({'recall': 0.48586387434554973, 'f1': 0.4436635266704804, 'precision': 0.5080061616028017, 'acc': 0.48586387434554973},) Thu Dec 14 12:12:15 2023
 main.py:189	line:189 -> logddd.log(prf) :  ({'recall': 0.3049214659685864, 'f1': 0.29740483724692574, 'precision': 0.3854058872915051, 'acc': 0.3049214659685864},) Thu Dec 14 13:39:27 2023
 main.py:189	line:189 -> logddd.log(prf) :  ({'recall': 0.4829319371727749, 'f1': 0.4670227316123955, 'precision': 0.49081399623788874, 'acc': 0.4829319371727749},) Thu Dec 14 15:07:28 2023
 main.py:189	line:189 -> logddd.log(prf) :  ({'recall': 0.4192670157068063, 'f1': 0.43091382258419214, 'precision': 0.4726933759135296, 'acc': 0.4192670157068063},) Thu Dec 14 16:35:48 2023
 main.py:189	line:189 -> logddd.log(prf) :  ({'recall': 0.5107853403141361, 'f1': 0.46437529661804416, 'precision': 0.49397894818582594, 'acc': 0.5107853403141361},) Thu Dec 14 18:03:07 2023
 main.py:201	line:201 -> logddd.log(prf) :  ('当前train数量为:5',) Thu Dec 14 18:03:07 2023
 main.py:189	line:189 -> logddd.log(prf) :  ({'recall': 0.5489005235602095, 'f1': 0.5393129232386619, 'precision': 0.5667280463606654, 'acc': 0.5489005235602095},) Thu Dec 14 19:46:56 2023
 main.py:189	line:189 -> logddd.log(prf) :  ({'recall': 0.5750785340314136, 'f1': 0.5528198533695645, 'precision': 0.5872498033898547, 'acc': 0.5750785340314136},) Thu Dec 14 21:32:44 2023
 main.py:189	line:189 -> logddd.log(prf) :  ({'recall': 0.5526701570680628, 'f1': 0.5353836390750403, 'precision': 0.5549983422317802, 'acc': 0.5526701570680628},) Thu Dec 14 23:12:29 2023
 main.py:189	line:189 -> logddd.log(prf) :  ({'recall': 0.5614659685863874, 'f1': 0.5467969037820527, 'precision': 0.5777079481281366, 'acc': 0.5614659685863874},) Fri Dec 15 00:53:33 2023
 main.py:189	line:189 -> logddd.log(prf) :  ({'recall': 0.578848167539267, 'f1': 0.5637968881631308, 'precision': 0.5723111809065456, 'acc': 0.578848167539267},) Fri Dec 15 02:42:14 2023
 main.py:201	line:201 -> logddd.log(prf) :  ('当前train数量为:10',) Fri Dec 15 02:42:14 2023
 main.py:189	line:189 -> logddd.log(prf) :  ({'recall': 0.5842931937172775, 'f1': 0.5802938599223955, 'precision': 0.6124373004311626, 'acc': 0.5842931937172775},) Fri Dec 15 04:38:20 2023
 main.py:189	line:189 -> logddd.log(prf) :  ({'recall': 0.5932984293193717, 'f1': 0.5784715974695993, 'precision': 0.5844569832958001, 'acc': 0.5932984293193717},) Fri Dec 15 06:36:10 2023
 main.py:189	line:189 -> logddd.log(prf) :  ({'recall': 0.6012565445026178, 'f1': 0.5869922799398174, 'precision': 0.6237907340759705, 'acc': 0.6012565445026178},) Fri Dec 15 08:25:27 2023
 main.py:189	line:189 -> logddd.log(prf) :  ({'recall': 0.601675392670157, 'f1': 0.5888145641052425, 'precision': 0.6034061756068902, 'acc': 0.601675392670157},) Fri Dec 15 10:11:36 2023
 main.py:189	line:189 -> logddd.log(prf) :  ({'recall': 0.5953926701570681, 'f1': 0.590393160938312, 'precision': 0.6186614238467266, 'acc': 0.5953926701570681},) Fri Dec 15 12:07:46 2023
 main.py:201	line:201 -> logddd.log(prf) :  ('当前train数量为:15',) Fri Dec 15 12:07:46 2023
 main.py:189	line:189 -> logddd.log(prf) :  ({'recall': 0.6163350785340315, 'f1': 0.5990926213936351, 'precision': 0.6160486088183986, 'acc': 0.6163350785340315},) Fri Dec 15 14:23:36 2023
 main.py:189	line:189 -> logddd.log(prf) :  ({'recall': 0.6067015706806282, 'f1': 0.6014275621181356, 'precision': 0.614321056376638, 'acc': 0.6067015706806282},) Fri Dec 15 16:25:16 2023
 main.py:189	line:189 -> logddd.log(prf) :  ({'recall': 0.6192670157068063, 'f1': 0.6081907512508851, 'precision': 0.612017848075556, 'acc': 0.6192670157068063},) Fri Dec 15 18:31:19 2023
 main.py:189	line:189 -> logddd.log(prf) :  ({'recall': 0.6127748691099476, 'f1': 0.6128707045645562, 'precision': 0.6265869023158886, 'acc': 0.6127748691099476},) Fri Dec 15 20:51:17 2023
 main.py:189	line:189 -> logddd.log(prf) :  ({'recall': 0.660523560209424, 'f1': 0.6483495605969888, 'precision': 0.6559585329723808, 'acc': 0.660523560209424},) Fri Dec 15 23:07:25 2023
 main.py:201	line:201 -> logddd.log(prf) :  ('当前train数量为:20',) Fri Dec 15 23:07:25 2023
 main.py:189	line:189 -> logddd.log(prf) :  ({'recall': 0.6372774869109947, 'f1': 0.6232030305838884, 'precision': 0.6400593901565326, 'acc': 0.6372774869109947},) Sat Dec 16 01:22:20 2023
 main.py:189	line:189 -> logddd.log(prf) :  ({'recall': 0.6372774869109947, 'f1': 0.6188437789743447, 'precision': 0.6301855735772393, 'acc': 0.6372774869109947},) Sat Dec 16 03:38:07 2023
 main.py:189	line:189 -> logddd.log(prf) :  ({'recall': 0.6578010471204189, 'f1': 0.6526609042632343, 'precision': 0.6616993701494075, 'acc': 0.6578010471204189},) Sat Dec 16 06:12:42 2023
 main.py:189	line:189 -> logddd.log(prf) :  ({'recall': 0.6479581151832461, 'f1': 0.636740019115434, 'precision': 0.6514756193016883, 'acc': 0.6479581151832461},) Sat Dec 16 08:44:28 2023
 main.py:189	line:189 -> logddd.log(prf) :  ({'recall': 0.6328795811518325, 'f1': 0.6274584079183149, 'precision': 0.6405634489487795, 'acc': 0.6328795811518325},) Sat Dec 16 11:04:14 2023
 main.py:201	line:201 -> logddd.log(prf) :  ('当前train数量为:25',) Sat Dec 16 11:04:14 2023
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
