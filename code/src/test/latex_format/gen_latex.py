import pandas as pd


def gen_latex():
    data = pd.read_csv('data.csv', header=None, sep='\t')
    res = "&\makecell{" + f"{data[0][5].round(3)}\\\\" + f"{data[0].min().round(2)}-" + f"{data[0].max().round(2)}" + "}&\makecell{" + f"{data[1][5].round(3)}\\\\" + f"{data[1].min().round(2)}-" + f"{data[1].max().round(2)}" + "}&\makecell{" + f"{data[2][5].round(3)}\\\\" + f"{data[2].min().round(2)}-" + f"{data[2].max().round(2)}" + "}"
    print(res)


gen_latex()
