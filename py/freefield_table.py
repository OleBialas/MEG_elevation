import csv
import numpy as np
import os

def freefield_table(**kwargs):

    """
    :param path (str): full path to the .csv file containing the table
    :param kwargs (list of str): attributes for which the table can be filtered. Possible arguments:
    Arch: I-VII. I is the center arch, even numbers are arches to the right, odd to the left
    Channel: 1-24, number of the analog output channel the speaker is connected to. Since we use two processors every number appears twice
    RX8: 1 or 2, number of the processor .In the TDT software speakers are addressed unambiguously by their channel + processor number
    Ongoing: 1 - 48 ongoing numbering going from bottom to top, left to right
    Azimuth: horizontal angle in degree. 0 represents the mid arch, a negative sign arches to the left, a positive to the right
    Elevation: vertical angle in degree. 0 represents the interaural line, a negative sign speakers below, a positive above
    :return (dict): filtered table. if no keyword arguments are provided, the whole table is returned.
    """

    handle = open(os.environ["EXPDIR"]+"freefield.csv", encoding="utf8")
    reader = csv.reader(handle)
    headers = reader.__next__()
    table = {}
    for h in headers:
        table[h] = []
    for row in reader:
        for h, v in zip(headers, row):
            table[h].append(v)

    for title, values in kwargs.items():
        tmp = {}
        for key in table:
            tmp[key] = []
        for value in values:
            pos = np.where(np.asanyarray(table[title]) == value)[0]
            for j in pos:
                for key in table.keys():
                    tmp[key].append(table[key][j])
        table = tmp
    return table

if __name__ == "__main__":
    import os
    os.environ["EXPDIR"] = "C:/Projects/Ole_Elevation/"
    table = freefield_table(Ongoing=["23"])
    print(table)