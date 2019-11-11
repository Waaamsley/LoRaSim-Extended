import random
import math
import numpy as np
import time


class nodePlacer():

    def __init__(self, nodes):
        self.nodes = nodes
        return

    def placeNodes(self, maxDist, bsx, bsy, experiment):
        found = 0
        rounds = 0
        x = 0.0
        y = 0.0

        while (found == 0 and rounds < 100 and experiment != 7):
            a = random.random()
            b = random.random()
            if b<a:
                a,b = b,a
            posx = b*maxDist*math.cos(2*math.pi*a/b)+bsx
            posy = b*maxDist*math.sin(2*math.pi*a/b)+bsy
            if len(self.nodes) > 0:
                for index, n in enumerate(self.nodes):
                    dist = np.sqrt(((abs(n.x-posx))**2)+((abs(n.y-posy))**2))
                    if dist >= 10:
                        found = 1
                        x = posx
                        y = posy
                    else:
                        rounds = rounds + 1
                        if rounds == 100:
                            print "could not place new node, giving up"
                            exit(-1)
            else:
                print "first node"
                x = posx
                y = posy
                found = 1
        if (experiment == 7):
            x = 10
            y = 10
            dist = 10
        else:
            dist = np.sqrt((x-bsx)*(x-bsx)+(y-bsy)*(y-bsy))

        return x, y, dist


class channelUsage(object):
    def __init__(self):
        #self.noTraffic = 0.0
        self._traffic = 0
        self.empty = False
        self.f_flag = 0.0
        self.e_flag = 0.0
        self.accum_e = 0.0
        self.accum_f = 0.0

    @property
    def traffic(self):
        return self._traffic

    @traffic.setter
    def traffic(self, value):
        self._traffic = value

        if self.traffic == 0.0 and not self.empty:
            self.empty = True
            self.e_flag = time.time()
            if (self.f_flag > 0.0):
                self.accum_f += (time.time()) - self.f_flag

        if self.traffic > 0.0 and self.empty:
            self.empty = False
            self.f_flag = time.time()
            self.accum_e += (time.time()) - self.e_flag


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
    def __init__(self):
        return

    """
    This methods finds whether a given nodes packets can reach the base-station.
    This method also returns the minimum viable spreading factor.
    """
    def maxDistance(self, rssi):
        distance = 40 * 10**((rssi-127.41)/20.8)

        return distance


class fairSF():

    def __init__(self, nodeCount, sfList):
        self.node_count = nodeCount
        self.sf_list = sfList
        return


    def base_function(self):
        sum_result = 0.0

        for sf in self.sf_list:
            sum_result += sf/(2**sf)

        return sum_result


    def get_percentages(self):
        sf_percentages = []

        for sf in self.sf_list:
            sf_percentages.append(self.get_percentage(sf))

        return sf_percentages


    def get_percentage(self, sf):
        sf_percentage = 0.0

        sum_result = self.base_function()
        sf_percentage =  (sf/(2**sf)) * sum_result

        return sf_percentage


class experiments():

    def __init__(self, experiment):
        self.experiment = experiments
        self.sf = 0
        self.cr = 0
        self.bw = 0
        self.ch = 0

        if experiment in [0, 1]:
            self.experimentOne()
        if experiment == 2:
            self.experimentTwo()
        if experiment in [3, 5]:
            self.experimentThive()
        if experiment == 4:
            self.experimentFour()
        if experiment == 6:
            self.experimentSix()
        if experiment == 7:
            self.experimentSeven()
        if experiment == 8:
            self.experimentEight()

    def experimentOne(self):
        self.sf = 0

    def experimentTwo(self):
        self.sf = 0

    def experimentThive(self):
        self.sf = 0

    def experimentFour(self):
        self.sf = 0

    def experimentSix(self):
        self.sf = 0

    def experimentSeven(self):
        self.sf = 0

    def experimentEight(self):
        self.sf = 0


class estimator:

    def airtime(self, sf, cr, pl, bw):
        H = 0  # implicit header disabled (H=0) or not (H=1)
        DE = 0  # low data rate optimization enabled (=1) or not (=0)
        Npream = 8  # number of preamble symbol (12.25  from Utz paper)

        if bw == 125 and sf in [11, 12]:
            # low data rate optimization mandated for BW125 with SF11 and SF12
            DE = 1
        if sf == 6:
            # can only have implicit header with SF6
            H = 1

        Tsym = (2.0 ** sf) / bw
        Tpream = (Npream + 4.25) * Tsym
        print "sf", sf, " cr", cr, "pl", pl, "bw", bw
        payloadSymbNB = 8 + max(math.ceil((8.0 * pl - 4.0 * sf + 28 + 16 - 20 * H) / (4.0 * (sf - 2 * DE))) * (cr + 4),
                                0)
        Tpayload = payloadSymbNB * Tsym
        return Tpream + Tpayload

    def chirpTime(self, sf, bw):
        chirpyTime = (2 ** sf) / bw
        return chirpyTime

    # Okumura-Hata path loss model.
    def hataUrban(self, sensi):
        pathLoss = 17.5 - sensi
        d = 10 ** ((pathLoss - 124.76) / 35.22)
        print(d)