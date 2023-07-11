
import json
import csv
if __name__ == '__main__':
    data = """ crf.py:287	line:287 -> logddd.log(prf) :  ({'recall': 0.5133809725210832, 'f1': 0.4947030898538106, 'precision': 0.49242282122186204},) Thu Jul  6 13:29:52 2023
 crf.py:287	line:287 -> logddd.log(prf) :  ({'recall': 0.5440332844462143, 'f1': 0.5261566327479665, 'precision': 0.5254332046021585},) Thu Jul  6 14:46:33 2023
 crf.py:287	line:287 -> logddd.log(prf) :  ({'recall': 0.5445206840193865, 'f1': 0.5331301057533986, 'precision': 0.5378595440030683},) Thu Jul  6 16:09:25 2023
 crf.py:287	line:287 -> logddd.log(prf) :  ({'recall': 0.5250873919796281, 'f1': 0.5090662127103831, 'precision': 0.5175470556337521},) Thu Jul  6 17:23:04 2023
 crf.py:287	line:287 -> logddd.log(prf) :  ({'recall': 0.5220658678140571, 'f1': 0.49862895276230945, 'precision': 0.4992336586472969},) Thu Jul  6 18:47:18 2023"""
    items = data.split("\n")
    total_res = []
    for item in items:
        item = item.split(":  (")[1].split(",)")[0]
        lines = item.replace("{","").replace("}","").split(",")
        lines[0],lines[1],lines[2] = lines[2],lines[0],lines[1]
        res = []
        for line in lines:
            res.append(str(line.split(": ")[1]))
        total_res.append(res)
    total_res.append([])
    with open("res.csv","a") as f:
            writer = csv.writer(f)
            writer.writerows(total_res)