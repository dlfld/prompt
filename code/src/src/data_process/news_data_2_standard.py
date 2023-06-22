from typing import List


def format_data_type_people_daily(datas:List[str]) -> List[List[str]]:
    """
        对人民日报数据进行处理。
        :param datas 19980101-01-001-002/m  中共中央/nt  总书记/n  、/w  国家/n  主席/n  江/nr  泽民/nr
        :return: 更改好的数据
            [
                ['脉/数', 'NR/VA'],
                ...
            ]
    """
    res = []
    for data in datas:
        if data == "\n":
            continue
        data = data.replace("\n","").split("  ")[1:-1]
        data_str = ""
        label_str = ""
        for index,item in enumerate(data):
            item = item.split("/")
            if index == 0:
                data_str += item[0]
                label_str += item[1]
            else:
                data_str += "/" + item[0]
                label_str += "/" + item[1]

        if data_str[-1] == '/':
            data_str = data_str[:-1]
        res.append([data_str,label_str])

    return res















if __name__ == '__main__':
    with open("/home/dlf/prompt/code/data/jw/PeopleDaily199801.txt","r") as f:
        datas = f.readlines()
        res = format_data_type_people_daily(datas)

        labels = []
        for item in res:
            label = item[1]
            print(item[1])
            labels.extend(label.split("/"))
        print(list(set(labels)))