"""
Testing LoRaSims path loss function to find the maximum distances for each SF.
needed functions and supporting information:

Lpl = Lpld0 + 10*gamma*math.log10(distance/d0)
Can rearrange above to get:
distance = d0 * 10**((Lpl-Lpld0)/10*2.08)
Above equation can give maximum distance for a given receiver sensitivity + Tx Power.
"""

#import numpy as np

class maxDistFinder():

    """
    Initialisation method
    """
    def __init__(self, sensitivities):
        self.sensi = sensitivities
        return

    """
    This methods finds whether a given nodes packets can reach the base-station.
    This method also returns the minimum viable spreading factor.
    """
    def maxDist(self, rssi):
        distance = 40 * 10**((rssi-127.41)/20.8)

        return distance

mdf = maxDistFinder([150, 147, 146, 143, 140, 137])

for sens in mdf.sensi:
    print(mdf.maxDist(sens))

