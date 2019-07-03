import sys

#Okumura-Hata path loss model.
def hataUrban(sensi):
    pathLoss = 17.5 - sensi
    d = 10**((pathLoss-124.76)/35.22)
    print(d)


if (len(sys.argv) == 2):
    sf_sensi = int(sys.argv[1])

hataUrban(sf_sensi)
exit()