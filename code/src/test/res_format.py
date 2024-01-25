import re

if __name__ == '__main__':

    data = """
     main.py:191	line:191 -> logddd.log(prf) :  ({'recall': 0.7553148816686723, 'f1': 0.7121493058280418, 'precision': 0.7414095474869167, 'acc': 0.7553148816686723},) Thu Dec 14 17:11:02 2023
 main.py:191	line:191 -> logddd.log(prf) :  ({'recall': 0.6177296430004011, 'f1': 0.5578626694893317, 'precision': 0.5384255986894385, 'acc': 0.6177296430004011},) Thu Dec 14 17:26:30 2023
 main.py:191	line:191 -> logddd.log(prf) :  ({'recall': 0.7007621339751303, 'f1': 0.642593718420281, 'precision': 0.6318156678814341, 'acc': 0.7007621339751303},) Thu Dec 14 17:41:45 2023
 main.py:191	line:191 -> logddd.log(prf) :  ({'recall': 0.6566385880465303, 'f1': 0.6540342221814857, 'precision': 0.6817674567254062, 'acc': 0.6566385880465303},) Thu Dec 14 17:56:46 2023
 main.py:191	line:191 -> logddd.log(prf) :  ({'recall': 0.5363016446048937, 'f1': 0.43273203944489547, 'precision': 0.5565123842693147, 'acc': 0.5363016446048937},) Thu Dec 14 18:11:31 2023
 main.py:203	line:203 -> logddd.log(prf) :  ('当前train数量为:5',) Thu Dec 14 18:11:31 2023
 main.py:191	line:191 -> logddd.log(prf) :  ({'recall': 0.8042519053349378, 'f1': 0.7658055653298972, 'precision': 0.7709636388876395, 'acc': 0.8042519053349378},) Thu Dec 14 18:28:07 2023
 main.py:191	line:191 -> logddd.log(prf) :  ({'recall': 0.7336542318491777, 'f1': 0.7058040068586496, 'precision': 0.7057018355447177, 'acc': 0.7336542318491777},) Thu Dec 14 18:44:10 2023
 main.py:191	line:191 -> logddd.log(prf) :  ({'recall': 0.753309265944645, 'f1': 0.6992683904450512, 'precision': 0.7359801731211999, 'acc': 0.753309265944645},) Thu Dec 14 19:00:27 2023
 main.py:191	line:191 -> logddd.log(prf) :  ({'recall': 0.766546329723225, 'f1': 0.7399069885110012, 'precision': 0.7730143918946455, 'acc': 0.766546329723225},) Thu Dec 14 19:16:50 2023
 main.py:191	line:191 -> logddd.log(prf) :  ({'recall': 0.6875250701965503, 'f1': 0.6456095142668019, 'precision': 0.6884922701104308, 'acc': 0.6875250701965503},) Thu Dec 14 19:32:28 2023
 main.py:203	line:203 -> logddd.log(prf) :  ('当前train数量为:10',) Thu Dec 14 19:32:28 2023
 main.py:191	line:191 -> logddd.log(prf) :  ({'recall': 0.802647412755716, 'f1': 0.7826181243202426, 'precision': 0.8046399016226959, 'acc': 0.802647412755716},) Thu Dec 14 19:50:18 2023
 main.py:191	line:191 -> logddd.log(prf) :  ({'recall': 0.7862013638186923, 'f1': 0.7491272264273975, 'precision': 0.7348879723478927, 'acc': 0.7862013638186923},) Thu Dec 14 20:07:32 2023
 main.py:191	line:191 -> logddd.log(prf) :  ({'recall': 0.8014440433212996, 'f1': 0.7746669277746332, 'precision': 0.782273052710561, 'acc': 0.8014440433212996},) Thu Dec 14 20:25:47 2023
 main.py:191	line:191 -> logddd.log(prf) :  ({'recall': 0.7777777777777778, 'f1': 0.7533865278646166, 'precision': 0.7574255972439989, 'acc': 0.7777777777777778},) Thu Dec 14 20:42:54 2023
 main.py:191	line:191 -> logddd.log(prf) :  ({'recall': 0.7970316887284397, 'f1': 0.7767730739470605, 'precision': 0.8036009768350708, 'acc': 0.7970316887284397},) Thu Dec 14 21:26:23 2023
 main.py:203	line:203 -> logddd.log(prf) :  ('当前train数量为:15',) Thu Dec 14 21:26:23 2023
 main.py:191	line:191 -> logddd.log(prf) :  ({'recall': 0.8363417569193743, 'f1': 0.8309960339733394, 'precision': 0.833100981414519, 'acc': 0.8363417569193743},) Thu Dec 14 23:07:53 2023
 main.py:191	line:191 -> logddd.log(prf) :  ({'recall': 0.8138788608102687, 'f1': 0.7921550836076585, 'precision': 0.792060254247083, 'acc': 0.8138788608102687},) Fri Dec 15 00:54:08 2023
 main.py:191	line:191 -> logddd.log(prf) :  ({'recall': 0.8058563979141596, 'f1': 0.7822894514619422, 'precision': 0.7855151476160404, 'acc': 0.8058563979141596},) Fri Dec 15 02:37:11 2023
 main.py:191	line:191 -> logddd.log(prf) :  ({'recall': 0.8303249097472925, 'f1': 0.8135908356947535, 'precision': 0.8283358184892144, 'acc': 0.8303249097472925},) Fri Dec 15 04:23:01 2023
 main.py:191	line:191 -> logddd.log(prf) :  ({'recall': 0.8291215403128761, 'f1': 0.8065364273151182, 'precision': 0.8078261539664433, 'acc': 0.8291215403128761},) Fri Dec 15 06:09:52 2023
 main.py:203	line:203 -> logddd.log(prf) :  ('当前train数量为:20',) Fri Dec 15 06:09:52 2023
 main.py:191	line:191 -> logddd.log(prf) :  ({'recall': 0.8223024468511834, 'f1': 0.8210765881446647, 'precision': 0.8394192953233675, 'acc': 0.8223024468511834},) Fri Dec 15 07:59:01 2023
 main.py:191	line:191 -> logddd.log(prf) :  ({'recall': 0.8142799839550742, 'f1': 0.792640283367184, 'precision': 0.8086689416570393, 'acc': 0.8142799839550742},) Fri Dec 15 09:48:47 2023
 main.py:191	line:191 -> logddd.log(prf) :  ({'recall': 0.8355395106297633, 'f1': 0.8228907637299908, 'precision': 0.826340779872953, 'acc': 0.8355395106297633},) Fri Dec 15 11:39:08 2023
 main.py:191	line:191 -> logddd.log(prf) :  ({'recall': 0.8407541115122342, 'f1': 0.8200639482622879, 'precision': 0.832003630020599, 'acc': 0.8407541115122342},) Fri Dec 15 12:18:18 2023
 main.py:191	line:191 -> logddd.log(prf) :  ({'recall': 0.8435619735258725, 'f1': 0.8185700637969082, 'precision': 0.8164182273477948, 'acc': 0.8435619735258725},) Fri Dec 15 12:37:49 2023
 main.py:203	line:203 -> logddd.log(prf) :  ('当前train数量为:25',) Fri Dec 15 12:37:49 2023
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
