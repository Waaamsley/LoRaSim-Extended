"""
Testing LoRaSims path loss function to find the maximum distances for each SF.
needed functions and supporting information:
Lpl = Lpld0 + 10*gamma*math.log10(distance/d0)
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
    def maxDist(self, sf, bw, txPower, distance):

        return

