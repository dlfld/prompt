def get_labels():
    return ["[PLB]", "NR", "NN", "AD", "PN", "OD", "CC", "DEG",
            "SP", "VV", "M", "PU", "CD", "BP", "JJ", "LC", "VC",
            "VA", "VE",
            "NT-SHORT", "AS-1", "PN", "MSP-2", "NR-SHORT", "DER",
            "URL", "DEC", "FW", "IJ", "NN-SHORT", "BA", "NT", "MSP", "LB",
            "P", "NOI", "VV-2", "ON", "SB", "CS", "ETC", "DT", "AS", "M", "X",
            "DEV"
            ]


if __name__ == '__main__':
    print(len(get_labels()))