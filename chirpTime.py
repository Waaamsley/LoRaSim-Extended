import math
import sys

def chirpTime(sf, bw):
    chirpyTime = (2**sf)/bw
    return chirpyTime


if (len(sys.argv) == 3):
    spreadingFactor = float(sys.argv[1])
    bandWidth = float(sys.argv[2])
else:
    print("Incorrect input")
    exit()

print(chirpTime(spreadingFactor, bandWidth))
exit()