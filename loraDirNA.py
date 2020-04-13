#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
 LoRaSim 0.2.1: simulate collisions in LoRa
 Copyright © 2016 Thiemo Voigt <thiemo@sics.se> and Martin Bor <m.bor@lancaster.ac.uk>

 This work is licensed under the Creative Commons Attribution 4.0
 International License. To view a copy of this license,
 visit http://creativecommons.org/licenses/by/4.0/.

 Do LoRa Low-Power Wide-Area Networks Scale? Martin Bor, Utz Roedig, Thiemo Voigt
 and Juan Alonso, MSWiM '16, http://dx.doi.org/10.1145/2988287.2989163

 $Date: 2017-05-12 19:16:16 +0100 (Fri, 12 May 2017) $
 $Revision: 334 $
"""

import simpy
import random
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import networkSupport

"""
 SYNOPSIS:
   ./loraDirNA.py <nodes> <avgsend> <experiment> <powerScheme> <simtime> <channels> <full_collision>
 DESCRIPTION:
    nodes
        number of nodes to simulate
    avgsend
        average sending interval in milliseconds
    experiment
        experiment is an integer that determines with what radio settings the
        simulation is run. All nodes are configured with a fixed transmit power
        and a single transmit frequency, unless stated otherwise.
        1   use the settings with the the slowest datarate (SF12, BW125, CR4/8).
        2   use the settings with the fastest data rate (SF7, BW500, CR4/5).
        3   use the default specified settings.
        4   optimise the setting per node based on the distance to the gateway.
        5   similair to experiment 3, but also optimises the transmit power.
        6   Divide and Conquer
    powerScheme
        Which power control scheme to use
        1 minimise all TP power as much as possible
        2 FADR TP control
        3 Mine?
    simtime
        total running time in milliseconds
    Channels
        How many channels for nodes to be distributed across. Assumes even distribution.
    collision
        set to 1 to enable the full collision check, 0 to use a simplified check.
        With the simplified check, two messages collide when they arrive at the
        same time, on the same frequency and spreading factor. The full collision
        check considers the 'capture effect', whereby a collision of one or the
 OUTPUT
    The result of every simulation run will be appended to a file named expX.dat,
    whereby X is the experiment number. The file contains a space separated table
    of values for nodes, collisions, transmissions and total energy spent. The
    data file can be easily plotted using e.g. gnuplot.
"""

# this is an array with measured values for sensitivity
# see paper, Table 3
sf7 = np.array([7, -123, -120, -116])
sf8 = np.array([8, -126, -123, -119])
sf9 = np.array([9, -129, -125, -122])
sf10 = np.array([10, -132, -128, -125])
sf11 = np.array([11, -133, -130, -128])
sf12 = np.array([12, -136, -133, -130])
sensi = np.array([sf7, sf8, sf9, sf10, sf11, sf12])

# Arrays with dB differences for inter-SF interference
sf7diff = np.array([6, -8, -9, -9, -9, -9])
sf8diff = np.array([-11, 6, -11, -12, -13, -13])
sf9diff = np.array([-15, -13, 6, -13, -14, -15])
sf10diff = np.array([-19, -18, -17, 6, -17, -18])
sf11diff = np.array([-22, -22, -21, -20, 6, -20])
sf12diff = np.array([-25, -25, -25, -24, -23, 6])
sensiDiff = np.array([sf7diff, sf8diff, sf9diff, sf10diff, sf11diff, sf12diff])

TxPowers = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]


#
# check for collisions at base station
# Note: called before a packet (or rather node) is inserted into the list
def checkcollision(packet):
    global fullCollision
    col = 0  # flag needed since there might be several collisions for packet
    processing = 0
    for i in range(0, len(packetsAtBS)):
        if packetsAtBS[i].packet.processed == 1:
            processing = processing + 1
    if processing > maxBSReceives:
        packet.processed = 0
    else:
        packet.processed = 1

    if packetsAtBS:
        for other in packetsAtBS:
            if other.nodeid != packet.nodeid:
                if fullCollision:
                    if timing_collision(packet, other.packet):
                        collided_packets = power_collision(packet, other.packet)
                        for p in collided_packets:
                            p.collided = 1
                            if p == packet:
                                col = 1
                else:
                    if timing_collision(packet, other.packet) and other.packet.sf == packet.sf:
                        other.collided = 1
                        packet.collided = 1
                        col = 1

        return col
    return 0


def power_collision(p1, p2):
    global interferCount
    power_threshold = sensiDiff[p1.sf - 7][p2.sf - 7]  # dB

    if (abs(p1.rssi - p2.rssi) <= power_threshold) and p1.sf == p2.sf:
        return p1, p2
    elif p1.rssi - p2.rssi <= power_threshold:
        interferCount[p2.sf - 7] += 1
        return p1,

    power_threshold = sensiDiff[p2.sf - 7][p1.sf - 7]  # dB

    if p2.rssi - p1.rssi <= power_threshold:
        interferCount[p1.sf - 7] += 1
        return p2,

    return []


def timing_collision(p1, p2):
    # The only way we can win is by being late enough (only the first n - 5 preamble symbols overlap)

    # assuming 8 preamble symbols
    npream = 8

    # we can lose at most (Npream - 5) * Tsym of our preamble
    tpreamb = 2 ** p1.sf / (1.0 * p1.bw) * (npream - 5)

    # check whether p2 ends in p1's critical section
    p2_end = p2.addTime + p2.rectime
    p1_cs = env.now + tpreamb
    if p1_cs < p2_end:
        return True
    return False


# Creates a list of nodes.
def create_nodes():
    for i in range(0, nrNodes):
        node = myNode(i, bsId, avgSend)
        nodes.append(node)
        env.process(transmitter.transmit(node))


#
# this function creates a node
#
class myNode:
    def __init__(self, nodeid, bs, duty):
        global experiment
        global plen
        global nodePlacer
        global Ptx
        self.nodeid = nodeid
        self.period = 999999
        self.bs = bs
        self.x = 0
        self.y = 0
        self.dist = 0

        self.x, self.y, self.dist = nodePlacer.logic(maxDist, bsx, bsy, nodeid)

        self.packet = myPacket(self.nodeid, self.dist)
        self.sent = 0
        # self.period = (self.packet.rectime * (100 / duty))
        self.period = duty

        # graphics for node
        global graphics
        if graphics == 1:
            global ax
            ax.add_artist(plt.Circle((self.x, self.y), 2, fill=True, color='blue'))

    def __lt__(self, other):
        return self.packet.Lpl > other.packet.rssi
        #return self.packet.rssi > other.packet.rssi


#
# this function creates a packet (associated with a node)
# it also sets all parameters, currently random
#
class myPacket:
    def __init__(self, nodeid, distance):
        global Ptx
        global gamma
        global d0
        global var
        global Lpld0
        global GL
        global nrChannels
        global plen

        # Phase One.
        self.Lpl = Lpld0 + 10 * gamma * math.log10(distance / d0)
        self.nodeid = nodeid
        self.pl = plen
        self.addTime = 0.0
        self.collided = 0
        self.processed = 0

        # Phase Two.
        self.sf = 0
        self.cr = 0
        self.bw = 0
        self.ch = 0
        self.rectime = 0
        self.symTime = 0

        # Phase Three.
        self.txpow = 14  # Default.
        self.rssi = 0

    def phase_two(self, sf, cr, bw, ch, rectime):
        self.sf = sf
        self.cr = cr
        self.bw = bw
        self.ch = ch
        self.rectime = rectime
        self.symTime = (2.0 ** self.sf) / self.bw

    def phase_three(self, txpow):
        self.txpow = txpow
        self.rssi = self.txpow - GL - self.Lpl


#
# main discrete event loop, runs for each node
# a global list of packet being processed at the gateway
# is maintained
#
class myTransmitter:
    def __init__(self, environment, obzerver):
        self.env = environment
        self.observer = obzerver

    def transmit(self, node):
        while True:
            yield self.env.timeout(random.expovariate(1.0 / float(node.period)))

            # time sending and receiving
            # packet arrives -> add to base station

            node.sent = node.sent + 1
            global sfSent
            sfSent[node.packet.sf - 7] += 1
            if node in packetsAtBS:
                print "ERROR: packet already in"
            else:
                sensitivity = sensi[node.packet.sf - 7, [125, 250, 500].index(node.packet.bw) + 1]
                if node.packet.rssi < sensitivity:
                    node.packet.lost = True
                else:
                    node.packet.lost = False
                    # adding packet if no collision
                    if checkcollision(node.packet) == 1:
                        node.packet.collided = 1
                    else:
                        node.packet.collided = 0
                    packetsAtBS.append(node)
                    self.observer.traffic = len(packetsAtBS)
                    node.packet.addTime = env.now

            yield env.timeout(node.packet.rectime)

            if node.packet.lost:
                global nrLost
                global sfLost
                nrLost += 1
                sfLost[node.packet.sf - 7] += 1
            if node.packet.collided == 1:
                global nrCollisions
                global sfCollided
                nrCollisions = nrCollisions + 1
                sfCollided[node.packet.sf - 7] += 1
            if node.packet.collided == 0 and not node.packet.lost:
                global nrReceived
                global sfReceived
                nrReceived = nrReceived + 1
                sfReceived[node.packet.sf - 7] += 1
            if node.packet.processed == 1:
                global nrProcessed
                nrProcessed = nrProcessed + 1

            # complete packet has been received by base station
            # can remove it
            if node in packetsAtBS:
                packetsAtBS.remove(node)
                self.observer.traffic = len(packetsAtBS)
                # reset the packet
            node.packet.collided = 0
            node.packet.processed = 0
            node.packet.lost = False


#
# "main" program
#

# get arguments
inputs = []
nrNodes_list = []
if len(sys.argv) == 3:
    inputFile = open(sys.argv[1], "r")
    nodeFile = open(sys.argv[2], 'r')
    for line in inputFile:
        inputs.append(line.strip())
    for line in nodeFile:
        nrNodes_list.append(line.strip())
else:
    print ("usage: submit the name of the inputs file for them to be read in.\n"
           "Consult the inputs_README for information")
    exit(-1)


avgSend = float(inputs[1])
experiment = int(inputs[2])
powerScheme = int(inputs[3])
simtime = int(inputs[4])
nrChannels = int(inputs[5])
fullCollision = int(inputs[6])
graphics = int(inputs[7])
distributionType = inputs[8]
Ptx = int(inputs[9])
print ("Nodes:", nrNodes_list)
print ("Average Send Time / Inter Packet Arrival Time:", avgSend)
print ("Experiment: ", experiment)
print ("Power Control Scheme: ", powerScheme)
print ("Simtime: ", simtime)
print ("Channels: ", nrChannels)
print ("Full Collision: ", fullCollision)
print ("Graphics: ", graphics)
print ("Distribution Type: ", distributionType)
print ("Base Tx Power: ", Ptx)

# global stuff
# Can do a while loop from here to end to repeat simulations.
sfs = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
results = open("results.txt", "a")
figure_count = 0
# For loop for how different nrNodes.
for nrNodes in nrNodes_list:
    results.write("----------------------------------------------------------------\n")
    results.write("----------------------------------------------------------------\n")
    fair_sf_getter = networkSupport.fairSF(nrNodes, sfs)
    sf_counts = fair_sf_getter.get_sf_counts()
    placementGenerator = networkSupport.placementGenerator(nrNodes, sf_counts)
    configurations = []
    placementGenerator.full_placement(configurations)

    configurations = configurations[17:]

    repetition = 0  # Going to do 5 repititions
    config_rep = 0  # max configurations of 20
    while config_rep < len(configurations):
        sfSent = [0, 0, 0, 0, 0, 0]
        sfReceived = [0, 0, 0, 0, 0, 0]
        sfLost = [0, 0, 0, 0, 0, 0]
        sfCollided = [0, 0, 0, 0, 0, 0]
        interferCount = [0, 0, 0, 0, 0, 0]
        curr_config = configurations[config_rep]
        results.write("Configuration: " + str(config_rep + 18) + ". Repetition: " + str(repetition + 1)
                      + ". Region Counts: " + str(curr_config) + "\n")

        nodes = []
        packetsAtBS = []
        env = simpy.Environment()
        # max distance: 300m in city, 3000 m outside (5 km Utz experiment)
        # also more unit-disc like according to Utz
        bsId = 1
        nrCollisions = 0
        nrReceived = 0
        nrProcessed = 0
        nrLost = 0

        # maximum number of packets the BS can receive at the same time
        maxBSReceives = 999

        gamma = 2.08
        d0 = 40.0
        var = 0  # variance ignored for nows
        Lpld0 = 127.41
        GL = 0
        plen = 20
        distFinder = networkSupport.maxDistFinder()
        observer = networkSupport.channelUsage()
        nodePlacer = networkSupport.nodePlacer(nodes, nrNodes, distributionType, sensi, Ptx, curr_config)
        experiLogic = networkSupport.experiments(experiment, nrChannels, sensi, plen, GL, Ptx)
        powerLogic = networkSupport.powerControl(powerScheme, sensi, sensiDiff, GL, Ptx)
        transmitter = myTransmitter(env, observer)

        if experiment == 2:
            minsensi = sensi[0, 1]
        else:
            minsensi = np.amin(sensi)
        Lpl = Ptx - minsensi
        maxDist = distFinder.max_distance((minsensi * -1) + max(TxPowers))
        results.write("maxDist: " + str(maxDist) + "\n")

        # base station placement
        bsx = maxDist + 10
        bsy = maxDist + 10
        xmax = bsx + maxDist + 20
        ymax = bsy + maxDist + 20

        # prepare graphics and add sink
        if graphics == 1:
            plt.ion()
            figure_count += 1
            plt.figure(figure_count)
            ax = plt.gcf().gca()
            # XXX should be base station position
            ax.add_artist(plt.Circle((bsx, bsy), 3, fill=True, color='green'))
            ax.add_artist(plt.Circle((bsx, bsy), maxDist, fill=False, color='green'))

        # Creates a list of nodes and their packets
        create_nodes()

        sorted_nodes = list(nodes)
        sorted_nodes.sort()
        for node in sorted_nodes:
            print node.packet.Lpl

        quit()

        experiLogic.logic(nodes, sf_counts, curr_config)

        # prepare show
        if graphics == 1:
            plt.xlim([0, xmax])
            plt.ylim([0, ymax])
            plt.draw()

        # start simulation
        nodesSorted = nodes
        nodesSorted.sort()
        powerLogic.logic(nodes)
        env.run(until=simtime)

        # print stats and save into file
        results.write("nrCollisions: " + str(nrCollisions) + "\n")

        # compute energy
        # Transmit consumption in mA from -2 to +17 dBm
        TX = [22, 22, 22, 23,  # RFO/PA0: -2..1
              24, 24, 24, 25, 25, 25, 25, 26, 31, 32, 34, 35, 44,  # PA_BOOST/PA1: 2..14
              82, 85, 90,  # PA_BOOST/PA1: 15..17
              105, 115, 125]  # PA_BOOST/PA1+PA2: 18..20
        V = 3.0  # voltage XXX
        sent = sum(n.sent for n in nodes)
        energy = sum(node.packet.rectime * TX[int(node.packet.txpow) + 2] * V * node.sent for node in nodes) / 1e6

        results.write("energy (in J): " + str(energy) + "\n")
        results.write("sent packets: " + str(sent) + "\n")
        results.write("collisions: " + str(nrCollisions) + "\n")
        results.write("received packets: " + str(nrReceived) + "\n")
        results.write("processed packets: " + str(nrProcessed) + "\n")
        results.write("lost packets: " + str(nrLost) + "\n")

        # data extraction rate
        der = (sent - nrCollisions) / float(sent)
        results.write("sent - nrCollisions. DER: " + str(der) + "\n")
        der = nrReceived / float(sent)
        results.write("nrReceived / sent. DER method 2: " + str(der) + "\n")

        counter = 7
        for receivedStat, sentStat, lostStat, collideStat, interferStat in zip(sfReceived, sfSent, sfLost, sfCollided,
                                                                               interferCount):
            if float(receivedStat) > 0 and float(sentStat) > 0:
                results.write("SF" + str(counter) + " DER: " + str(float(receivedStat) / float(sentStat)) +
                              " Received/Sent/Lost/Collided/Interfered Packets: " + str(float(receivedStat)) + "/"
                              + str(float(sentStat)) + "/" + str(float(lostStat)) + "/"
                              + str(float(collideStat)) + "/" + str(float(interferStat)) + "\n")
            else:
                results.write("SF" + str(counter) + "Exception: Received/Sent/Lost/Collided/Interefered Packets : " +
                              str(float(receivedStat)) + "/" + str(float(sentStat)) + "/" + str(float(lostStat)) + "/"
                              + str(float(collideStat)) + "/" + str(float(interferStat)) + "\n")
            counter += 1
        results.write("SF Counts: " + str(experiLogic.sfCounts) + "\n")
        if observer.accum_f > 0 and observer.accum_e > 0:
            totalTime = observer.accum_f + observer.accum_e
            results.write("Accumulted full time: " + str(observer.accum_f) + ", " + str(observer.accum_f / totalTime)
                          + "%" + "\n")
            results.write("Accumulted empty time: " + str(observer.accum_e) + ", " + str(observer.accum_e / totalTime)
                          + "%" + "\n")
        results.write("----------------------------------------------------------------\n")

        repetition += 1
        if repetition == 5:
            repetition = 0
            config_rep += 1

results.close()
# this can be done to keep graphics visible
if graphics == 1:
    plt.show()
    raw_input('Press Enter to continue ...')

# save experiment data into a dat file that can be read by e.g. gnuplot
# name of file would be:  exp0.dat for experiment 0
# fname = "exp" + str(experiment) + ".dat"
# print fname
# if os.path.isfile(fname):
#    res = "\n" + str(nrNodes) + " " + str(nrCollisions) + " "  + str(sent) + " " + str(energy)
# else:
#    res = "#nrNodes nrCollisions nrTransmissions OverallEnergy\n" + str(nrNodes) \
#    + " " + str(nrCollisions) + " "  + str(sent) + " " + str(energy)
# with open(fname, "a") as myfile:
#    myfile.write(res)
# myfile.close()

# with open('nodes.txt','w') as nfile:
#     for n in nodes:
#         nfile.write("{} {} {}\n".format(n.x, n.y, n.nodeid))
# with open('basestation.txt', 'w') as bfile:
#     bfile.write("{} {} {}\n".format(bsx, bsy, 0))
