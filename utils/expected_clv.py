def e_clv(alpha, beta, d, net_cf):
    t = list(range(0, 200))
    r = []
    s = []
    disc = []
    for i in t:
        if i == 0: 
            r.append(0)
            s.append(1)
            disc.append(1)
        else:
            r.append((beta+i-1)/(alpha+beta+i-1))
            s.append(r[i]*s[i-1])
            disc.append((1/(1+d)**i))
    clv = net_cf * sum([x * y for x, y in zip(s, disc)])
    return clv
