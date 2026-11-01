import sys; sys.path.insert(0,'.')
import mem
C=mem.C
bad=[]
for L in ((1,2),(2,1)):
    for s in (32,35,38,41,80,85,90,95):
        for o in mem.RETRY:
            try:
                b=C.CBody(s+o,'body','lean',L); b.g.close()
            except Exception as e:
                bad.append((L,s,o,s+o))
print(bad)
