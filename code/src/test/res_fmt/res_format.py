import re

# if __name__ == '__main__':
#
#     data = """
#
#     """
with open("/Users/dailinfeng/Desktop/prompt/code/src/test/res_fmt/res.txt", "r") as f:
    data = f.read()
    items = data.split("\n")
    total_res = []
    for item in items:
        if item.strip() == "":
            continue
        pattern = r"当前train数量为:(\d+)"
        match = re.search(pattern, item)
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

    with open("./resres.csv", "w") as f:
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
