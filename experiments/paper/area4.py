def pm(d):
    return "%.1f $\pm$ %.1f" % d

with open("./area4.csv", 'r') as fh:
    lines = fh.read().split("\n")[:-1]
    data = [line.split("\t") for line in lines]
    names = [x[0] for x in data]
    data = [(float(x[2])*1e3, float(x[3])*1e3) for x in data]
    print(data)
    print(names)
    for i in range(int(len(data) / 3)):
        p = 3*i
        #print(names[p])
        #print(data[p], data[p+1], data[p+2])
        t = " & ".join([names[p], pm(data[p]), pm(data[p+1]), pm(data[p+2])])
        #t = " & ".join([pm(data[p]), pm(data[p+1]), pm(data[p+2])]) + " & "
        print(t)

