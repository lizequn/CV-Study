def conv_filter(F,H):
    k = len(H)
    if k < 3:
        return
    if k!=len(H[0]):
        return
    off = (k-1)/2
    mi = len(F)
    mj = len(F[0])
    G = np.zeros((mi,mj))
    for i in xrange(0,mi):
        for j in xrange(0,mj):
            sum = 0
            for u in xrange(0,k+1):
                for v in xrange(0,k+1):
                    ru = u-off
                    rv = v-off
                    if i-ru < 0 or i-ru >=mi:
                        continue
                    if j-rv < 0 or j-rv >=mj:
                        continue
                    sum += F[i-ru][j-rv] * H[u][v]
            G[i][j] = sum
    G = np.array(G,np.float32)
    return G
