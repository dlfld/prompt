import re

if __name__ == '__main__':

    data = """ bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.27763902763902765, 'f1': 0.26282963412959714, 'precision': 0.2637953463758238},) Tue Aug 15 05:06:52 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.27464202464202464, 'f1': 0.24843732833713356, 'precision': 0.23109497370638435},) Tue Aug 15 05:13:44 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.28363303363303366, 'f1': 0.21981368896128134, 'precision': 0.18536211301998617},) Tue Aug 15 05:20:36 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.2405094905094905, 'f1': 0.18739956291313678, 'precision': 0.1821241737142062},) Tue Aug 15 05:27:27 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.2977855477855478, 'f1': 0.25117299652407016, 'precision': 0.24566098823150978},) Tue Aug 15 05:34:19 2023
 bert_bilstm_crf.py:360	line:360 -> logddd.log(prf) :  ('当前train数量为:5',) Tue Aug 15 05:34:19 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.2863802863802864, 'f1': 0.1712141171508237, 'precision': 0.1221611198174713},) Tue Aug 15 05:41:15 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.2686480186480186, 'f1': 0.2155052632706653, 'precision': 0.22475990531575593},) Tue Aug 15 05:48:11 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.28954378954378956, 'f1': 0.2412050112836819, 'precision': 0.22376697624815914},) Tue Aug 15 05:55:07 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.3006160506160506, 'f1': 0.24008127260297074, 'precision': 0.2257013565241058},) Tue Aug 15 06:02:04 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.2957042957042957, 'f1': 0.2290250548784116, 'precision': 0.1957657834382855},) Tue Aug 15 06:09:00 2023
 bert_bilstm_crf.py:360	line:360 -> logddd.log(prf) :  ('当前train数量为:10',) Tue Aug 15 06:09:01 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.27697302697302695, 'f1': 0.21964271537053043, 'precision': 0.2271049353584786},) Tue Aug 15 06:16:05 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.2520812520812521, 'f1': 0.1733053948924356, 'precision': 0.2557682149802512},) Tue Aug 15 06:23:09 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.3061105561105561, 'f1': 0.26479846448271893, 'precision': 0.2743997390278009},) Tue Aug 15 06:30:14 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.30902430902430905, 'f1': 0.25558151621024205, 'precision': 0.2509895856085619},) Tue Aug 15 06:37:18 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.27697302697302695, 'f1': 0.22677518407125416, 'precision': 0.2267250811760583},) Tue Aug 15 06:44:22 2023
 bert_bilstm_crf.py:360	line:360 -> logddd.log(prf) :  ('当前train数量为:15',) Tue Aug 15 06:44:22 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.2559107559107559, 'f1': 0.23919064051528593, 'precision': 0.24856714174530228},) Tue Aug 15 06:51:51 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.28471528471528473, 'f1': 0.2607595091634085, 'precision': 0.26269812316667923},) Tue Aug 15 06:59:19 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.27322677322677325, 'f1': 0.2481277251632816, 'precision': 0.2565096947218406},) Tue Aug 15 07:06:48 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.2619047619047619, 'f1': 0.23025695132345442, 'precision': 0.2160711401298433},) Tue Aug 15 07:14:17 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.2982017982017982, 'f1': 0.25529945230527734, 'precision': 0.24245376735496396},) Tue Aug 15 07:21:45 2023
 bert_bilstm_crf.py:360	line:360 -> logddd.log(prf) :  ('当前train数量为:20',) Tue Aug 15 07:21:45 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.31385281385281383, 'f1': 0.2791816778421472, 'precision': 0.2768403162156544},) Tue Aug 15 07:29:18 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.30128205128205127, 'f1': 0.2711708118726071, 'precision': 0.25411350338895417},) Tue Aug 15 07:36:52 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.2991175491175491, 'f1': 0.26930940991983604, 'precision': 0.28309970620340286},) Tue Aug 15 07:44:25 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.2824675324675325, 'f1': 0.25800886874599943, 'precision': 0.27583038503099466},) Tue Aug 15 07:51:58 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.2804695304695305, 'f1': 0.2571986436119814, 'precision': 0.243867997971385},) Tue Aug 15 07:59:31 2023
 bert_bilstm_crf.py:360	line:360 -> logddd.log(prf) :  ('当前train数量为:25',) Tue Aug 15 07:59:31 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.29087579087579085, 'f1': 0.2792124009437698, 'precision': 0.2979274032296591},) Tue Aug 15 08:08:14 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.289044289044289, 'f1': 0.27397112595337697, 'precision': 0.2837788231491136},) Tue Aug 15 08:17:00 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.30486180486180486, 'f1': 0.273667264245712, 'precision': 0.26083829690738247},) Tue Aug 15 08:25:44 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.2862970362970363, 'f1': 0.2815210129584612, 'precision': 0.2845777847954094},) Tue Aug 15 08:34:27 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.29978354978354976, 'f1': 0.2790956519957923, 'precision': 0.2813452425586882},) Tue Aug 15 08:43:10 2023
 bert_bilstm_crf.py:360	line:360 -> logddd.log(prf) :  ('当前train数量为:50',) Tue Aug 15 08:43:10 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.31185481185481184, 'f1': 0.27660852593502877, 'precision': 0.27842252549533897},) Tue Aug 15 08:52:39 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.30594405594405594, 'f1': 0.28814151957604056, 'precision': 0.29028056329974317},) Tue Aug 15 09:02:09 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.3225940725940726, 'f1': 0.3032759332897551, 'precision': 0.30113437150950223},) Tue Aug 15 09:11:39 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.3041958041958042, 'f1': 0.2823316455864038, 'precision': 0.2787140182164156},) Tue Aug 15 09:21:09 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.3082750582750583, 'f1': 0.289427860124964, 'precision': 0.28398978393902286},) Tue Aug 15 09:30:42 2023
 bert_bilstm_crf.py:360	line:360 -> logddd.log(prf) :  ('当前train数量为:75',) Tue Aug 15 09:30:42 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.34973359973359974, 'f1': 0.34666610293755773, 'precision': 0.3489088253593573},) Tue Aug 15 09:41:20 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.3041125541125541, 'f1': 0.29683041418115413, 'precision': 0.29800159579970276},) Tue Aug 15 09:52:00 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.3244255744255744, 'f1': 0.3197916377402843, 'precision': 0.33016914823270793},) Tue Aug 15 10:02:39 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.31393606393606394, 'f1': 0.30043406816439466, 'precision': 0.29627389868581405},) Tue Aug 15 10:13:15 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.33166833166833165, 'f1': 0.3172357439707304, 'precision': 0.32675726818883216},) Tue Aug 15 10:23:51 2023
 bert_bilstm_crf.py:360	line:360 -> logddd.log(prf) :  ('当前train数量为:100',) Tue Aug 15 10:23:51 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.3856976356976357, 'f1': 0.3839236793030239, 'precision': 0.3908590451789223},) Tue Aug 15 10:38:18 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.38836163836163834, 'f1': 0.3815527814489804, 'precision': 0.38134327327466383},) Tue Aug 15 10:52:48 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.30536130536130535, 'f1': 0.2896801884514887, 'precision': 0.2848539268506818},) Tue Aug 15 11:07:14 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.3170995670995671, 'f1': 0.30075382571322107, 'precision': 0.29658302058398356},) Tue Aug 15 11:21:40 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.3822843822843823, 'f1': 0.3788492177749951, 'precision': 0.37937085306535223},) Tue Aug 15 11:36:06 2023
 bert_bilstm_crf.py:360	line:360 -> logddd.log(prf) :  ('当前train数量为:200',) Tue Aug 15 11:36:06 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.5920745920745921, 'f1': 0.5898406357031658, 'precision': 0.5929321350181876},) Tue Aug 15 12:02:31 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.5502830502830502, 'f1': 0.5525288325645238, 'precision': 0.5609621788861786},) Tue Aug 15 12:28:57 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.5185647685647685, 'f1': 0.5207146638690544, 'precision': 0.5272693882862305},) Tue Aug 15 12:55:19 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.5868298368298368, 'f1': 0.5866163644347264, 'precision': 0.5893222144868907},) Tue Aug 15 13:21:35 2023
 bert_bilstm_crf.py:340	line:340 -> logddd.log(prf) :  ({'recall': 0.5407092907092907, 'f1': 0.5380848304460606, 'precision': 0.5435546562024653},) Tue Aug 15 13:47:59 2023
 bert_bilstm_crf.py:360	line:360 -> logddd.log(prf) :  ('当前train数量为:500',) Tue Aug 15 13:47:59 2023"""
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
