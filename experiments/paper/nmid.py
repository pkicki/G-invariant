def pm(d):
    return "%.1f $\pm$ %.1f" % d

with open("./area_nmid.csv", 'r') as fh:
#with open("./poly_nmid.csv", 'r') as fh:
    lines = fh.read().split("\n")[:-1]
    data = [line.split("\t") for line in lines]
    names = [x[0] for x in data]
    mae = [(float(x[2])*1e3, float(x[3])*1e3) for x in data if "test" in x[1]]
    #mae = [(float(x[2])*1e2, float(x[3])*1e2) for x in data if "test" in x[1]]
    rt = [(float(x[4])*1e3, float(x[5])*1e3) for x in data if "test" if x[1]]
    print(mae)
    print(len(mae))
    print(rt)
    print(len(rt))
    for i in range(8):
        text = " & ".join([str(2 ** i), pm(mae[i]), pm(rt[i]), pm(mae[8 + i]), pm(rt[8 + i])]) + " \\\\"
        print(text)
