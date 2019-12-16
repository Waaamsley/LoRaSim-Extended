import random
import math
import numpy as np
import time
import operator


class nodePlacer():

    def __init__(self, nodes, nrNodes, distributionType, sensi):
        self.nodes = nodes
        self.nrNodes = nrNodes
        self.distributionType = distributionType
        self.sensi = sensi

        if self.distributionType == "ideal":
            fairSFGetter = fairSF(nrNodes, [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
            self.sfCounts = fairSFGetter.getSFCounts()
            self.distanceFinder = maxDistFinder()
        return

    def logic(self, maxDist, bsx, bsy, nodeid):
        x = 0
        y = 0
        dist = 0

        if(self.distributionType == "uniform"):
            x, y, dist = self.uniformPlace(maxDist, bsx, bsy,)
        elif(self.distributionType == "uniform basic"):
            x, y, dist = self.uniformPlaceBasic(maxDist, bsx, bsy)
        elif(self.distributionType == "ideal"):
            x, y, dist = self.idealPlace(bsx, bsy, nodeid)

        return x, y, dist

    def idealPlace(self, bsx, bsy, nodeid):
        x = 0
        y = 0
        dist = -1
        region = 0
        sum = 0
        for i, sfCount in enumerate(self.sfCounts):
            sum += sfCount
            if nodeid < sum:
                region = i
                break

        # currently assuming static txPower of 14dB
        rssi = 14 + (-1 * self.sensi[region, 1])
        regionMaxDistance = self.distanceFinder.maxDistance(rssi)
        if region > 0:
            minRssi = 14 + (-1 * self.sensi[region-1, 1])
            regionMinDistance = self.distanceFinder.maxDistance(minRssi)
        else:
            regionMinDistance = 15 #0 number 15 introduces ma deadzone which is 12.64 metres

        # Very bad way to account for minimum allowed distance.
        while dist < regionMinDistance or dist > regionMaxDistance:
            a = random.random()
            b = random.random()
            if b < a:
                a, b = b, a
            x = b * regionMaxDistance * math.cos(2 * math.pi * a / b) + bsx
            y = b * regionMaxDistance * math.sin(2 * math.pi * a / b) + bsy
            dist = np.sqrt((x - bsx) * (x - bsx) + (y - bsy) * (y - bsy))

        return x, y, dist

    def uniformPlaceBasic(self, maxDist, bsx, bsy):
        x = 0
        y = 0
        dist = 0
        a = random.random()
        b = random.random()

        if b < a:
            a, b = b, a
        x = b * maxDist * math.cos(2 * math.pi * a / b) + bsx
        y = b * maxDist * math.sin(2 * math.pi * a / b) + bsy
        dist = np.sqrt((x-bsx)*(x-bsx)+(y-bsy)*(y-bsy))

        return x, y, dist

    def uniformPlace(self, maxDist, bsx, bsy):
        found = 0
        rounds = 0
        x = 0.0
        y = 0.0
        dist = 0

        while (found == 0 and rounds < 100):
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
                            print("could not place new node, giving up")
                            exit(-1)
            else:
                x = posx
                y = posy
                found = 1
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
    def maxDistance(self, maxLoss):
        distance = 40 * 10**((maxLoss-127.41)/20.8)

        return distance


class fairSF():

    def __init__(self, nrNodes, sfList):
        self.nrNodes = nrNodes
        self.sfList = sfList
        self.baseResult = self.baseFunction()
        return


    def baseFunction(self):
        sumResult = 0.0

        for sf in self.sfList:
            sumResult += sf/(2**sf)

        return sumResult

    def getSFCounts(self):
        sfCounts = []
        total = 0

        sfPercentages = self.getPercentages()
        beforeRound = []
        for sfP in sfPercentages:
            tempCount = int(round(sfP * self.nrNodes))
            beforeRound.append(sfP * self.nrNodes)
            sfCounts.append(tempCount)
            total += tempCount

        difference = total - self.nrNodes
        if difference != 0:
            print("Round off error!!!!! total - nrNodes Difference : ", difference)
            print("before Round: ", beforeRound, "sfCounts: ", sfCounts, "\nnrNodes", self.nrNodes)
            quit()
        #if difference > 0:
            #subtract nodes from regions
        #elif difference < 0:
            #add nodes to region

        return sfCounts

    def getPercentages(self):
        sfPercentages = []

        for sf in self.sfList:
            sfPercentages.append(self.getPercentage(sf))

        return sfPercentages


    def getPercentage(self, sf):
        sfPercentage = 0.0

        sfPercentage =  (sf/(2**sf)) / self.baseResult

        return sfPercentage


class experiments():

    def __init__(self, eXperiment, nrChannels, sensi, plen, GL):
        self.experiment = eXperiment
        self.esti = estimator()
        self.nrChannels = nrChannels
        self.sensi = sensi
        self.plen = plen
        self.GL = GL
        self.sfCounts = [0, 0, 0, 0, 0, 0]
        self.powerControl = [7, 8, 9, 10, 11, 12]

    def logic(self, prx):
        sf = 0
        cr = 0
        bw = 0
        ch = random.randint(0, self.nrChannels - 1)
        rectime = 0
        Prx = prx

        if self.experiment == 1:
            sf, cr, bw = self.experimentOne()
        elif self.experiment == 2:
            sf, cr, bw = self.experimentTwo()
        elif self.experiment == 3:
            sf, cr, bw = self.experimentThree()
        elif self.experiment in [4]:
            sf, cr, bw = self.experimentFour(Prx)
        elif self.experiment == 5:
            sf, cr, bw = self.experimentFive()
        elif self.experiment == 6:
            self.experimentSix()
        else:
            print("Invalid experiment!\nQuitting!")
            quit()

        rectime = self.esti.airtime(sf, 1, self.plen, bw)
        if self.experiment == 1:
            rectime = self.esti.airtime(sf, 4, self.plen, bw)

        self.sfCounts[sf-7] += 1
        return sf, cr, bw, ch, rectime

    def experimentOne(self):
        return 12, 4, 125

    def experimentTwo(self):
        return 7, 1, 125

    def experimentThree(self):
        return 12, 1, 125

    def experimentFour(self, prx):
        minairtime = 9999
        sf = 0
        bw = 125
        cr = 1
        Prx = prx

        for i in range(0, 6):
            if (self.sensi[i, 1] <= Prx):
                sf = int(self.sensi[i, 0])
                minairtime = self.esti.airtime(sf, 1, self.plen, bw)
                break
        if (minairtime == 9999):
            print "does not reach base station"
            exit(-1)

        return sf, cr, bw

    # Original inspiration for Divide & Conquer
    def experimentFive(self):
        sf = 0
        bw = 125
        cr = 1

        # SF, CR, BW
        return 12, 1, 125

    #FADR - Fair Allocation Data Rate Algorithm
    def experimentSix(self):

        return 0, 0, 0

class powerControl():

    def __init__(self, powerScheme, sensi, sensiDiff, GL):
        self.powerScheme = powerScheme
        self.sensi = sensi
        self.sensiDiff = sensiDiff
        self.GL = GL
        self.atrGet = operator.attrgetter

    def logic(self, nodes):
        if self.powerScheme == 1:
            self.powerOne(nodes)
        elif self.powerScheme == 2:
            self.powerTwo(nodes)
        elif self.powerScheme == 3:
            self.powerThree(nodes)
        else:
            return

    #will have to reset txpow, Prx, rssi
    def powerOne(self, nodes):
        for node in nodes:
            minsensi = self.sensi[node.packet.sf - 7, 1]
            Lpl = node.packet.Lpl
            txpow = node.packet.txpow
            Prx = node.packet.Prx
            txpow = max(2, txpow - math.floor(Prx - minsensi))
            Prx = txpow - self.GL - Lpl
            node.packet.txpow = txpow
            node.packet.Prx = Prx
            node.packet.rssi = Prx

    # FADR - Fair Adaptive Data Rate
    # I have implemented their power control system
    def powerTwo(self, nodes):
        # First sort nodes by RSSI, done with __lt__ method on node class.
        nodesSorted = nodes
        nodesSorted.sort()

        # get max/min RSSI and min CIR (inter SF collision?)
        minRSSI = min(nodesSorted, key=self.atrGet('packet.rssi')).packet.rssi
        maxRSSI = max(nodesSorted, key=self.atrGet('packet.rssi')).packet.rssi
        minCIR = 8

        # Find range of power levels to use
        powerLevels = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        minPower = powerLevels.pop(0)
        print minRSSI, maxRSSI
        for i, powerLevel in enumerate(powerLevels):
            maxPower = powerLevel
            if (maxRSSI + minPower - minRSSI - maxPower) <= minCIR:
                powerLevels = powerLevels[0, i]
                break
            elif powerLevel == max(powerLevels):
                maxPower = powerLevels.pop()

        # Recalc MinRSSI, MaxRSSI
        minRSSI = min(minRSSI + maxPower, maxRSSI + minPower)
        maxRSSI = max(minRSSI + maxPower, maxRSSI + minPower)

        # Assign minimum power and save minPowerIndex
        for n in nodesSorted:
            if n.packet.rssi + minPower > minRSSI:
                minPowerIndex = i - 1
                break
            else:
                n.packet.txpow = minPower
                Prx = n.packet.txpow - self.GL - n.packet.Lpl
                n.packet.Prx = Prx
                n.packet.rssi = Prx

        # Assign maximum power and save maxPowerIndex
        for n in reversed(nodesSorted):
            if n.packet.rssi + maxPower - minRSSI > minCIR:
                maxPowerIndex = i - 1
                break
            else:
                n.packet.txpow = maxPower
                Prx = n.packet.txpow - self.GL - n.packet.Lpl
                n.packet.Prx = Prx
                n.packet.rssi = Prx

        #Assign the reaming power levels to the inbetween nodes
        tempIndex = minPowerIndex
        maxNodeRssi = nodesSorted[maxPowerIndex].packet.rssi
        for powerLevel in powerLevels:
            tempNodeRssi = nodesSorted[tempIndex].packet.rssi
            if tempNodeRssi + powerLevel - minRSSI <= minCIR \
                and tempNodeRssi + powerLevel - maxNodeRssi - maxPower <= minCIR:
                for i in range(tempIndex, maxPowerIndex):
                    currNodeRssi = nodesSorted[i].packet.rssi
                    if currNodeRssi + powerLevel - maxNodeRssi - maxPower > minCIR:
                        tempIndex = i - 1
                        break
                    else:
                        nodesSorted[i].packet.txpow = powerLevel
                        Prx = n.packet.txpow - self.GL - n.packet.Lpl
                        nodesSorted[i].packet.Prx = Prx
                        nodesSorted[i].packet.rssi = Prx
        return

    def powerThree(self, nodes):
        return


class estimator():

    # this function computes the airtime of a packet
    # according to LoraDesignGuide_STD.pdf
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
        #print "sf", sf, " cr", cr, "pl", pl, "bw", bw
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