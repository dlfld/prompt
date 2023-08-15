datas = """(Loss=3.74678):
(Loss=3.5637):
(Loss=3.60734):
(Loss=3.5585):
(Loss=3.64523):
(Loss=3.33424):
(Loss=3.44937):
(Loss=3.21226):
(Loss=3.28779):
(Loss=3.02186):
(Loss=3.17414):
(Loss=2.91781):
(Loss=3.13319):
(Loss=2.88952):
(Loss=3.04575):
(Loss=2.89138):
(Loss=2.90101):
(Loss=2.85725):
(Loss=2.88041):
(Loss=2.82843):
(Loss=2.85148):
(Loss=2.84713):
(Loss=2.83664):
(Loss=2.85521):
(Loss=2.86669):
(Loss=2.82473):"""
if __name__ == '__main__':
    for data in datas.split("\n"):
        data = data.replace("(Loss=","").replace("):","")
        print(data)