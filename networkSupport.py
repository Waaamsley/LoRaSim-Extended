"""
    Written by James Walmsley.
    This file contains all of the classes/methods that support the main simulation file.
    Therefore, this can be thought of as my supporting library to simplify the main file loraDirNA.py.
"""

import random
import math
import numpy as np
import time
import operator

"""
    Generates node distributions for simulation.
"""
class placementGenerator:
    """
        Initialisation method for distribution generation.
        @:param nr_nodes: Number of nodes for experiment.
    """
    def __init__(self, nr_nodes, region_counts):
        self.nr_nodes = nr_nodes
        self.region_counts = region_counts

    """
        Generates a large set of node distributions for experiments.
        @:param configurations: List of output configurations to append node distributions to.
    """
    def full_placement(self, configurations):
        start = [self.nr_nodes, 0, 0, 0, 0, 0]
        goal = list(self.region_counts)
        configurations.append(list(start))
        self.wave(configurations, start, goal, 5)

        start = list(goal)
        avg = float(self.nr_nodes) / float(len(self.region_counts))
        goal = [avg, avg, avg, avg, avg, avg]
        self.wave(configurations, start, goal, 5)

        start = list(goal)
        goal = list(self.region_counts)
        goal.reverse()
        self.wave(configurations, start, goal, 5)

        start = list(goal)
        goal = [0, 0, 0, 0, 0, self.nr_nodes]
        self.wave(configurations, start, goal, 5)

    """
        Generates a wave effect of node distributions.
        Returns several distributions that when visualised, node placement change looks like a wave.
        This is generated when given a starting and goal configuration, inbetween configs are generated.
        @:param configurations: List of output configurations to append node distributions to.
        @:param start: the starting configuration to generate from.
        @:param goal: The final configuration.
        @:param steps: How many frames of the wave to generate as configurations.
    """
    def wave(self, configurations, start, goal, steps):
        differences = []
        total = 0
        actual_total = 0

        for i in range(len(start) - 1):
            total += goal[i]
            actual_total += start[i]
            differences.append(actual_total - total)

        temp = list(start)
        mods = [0, 0, 0, 0, 0]
        for i in range(steps):
            total = 0
            for j in range(len(differences)):
                if start[j] != 0:
                    if mods[j] == 0:
                        mods[j] = 1.0 / (float(steps) - i)
                    change = differences[j] * mods[j]
                    start[j] -= change
                    start[j + 1] += change
                    temp[j] = int(round(start[j]))
                    temp[j + 1] = int(round(start[j + 1]))
                    total += temp[j]
                else:
                    break
            total += temp[-1]
            temp[0] += (self.nr_nodes - total)
            configurations.append(list(temp))


"""
    Provides the metrics that would be given as if the node is physically placed.
"""
class nodePlacer:

    """
        Initialisation method.
        @:param nodes: List of node instances.
        @:param nrdnodes: List of the number of nodes for each experiment.
        @:param distributiontype: The type of distribution to follow when placing nodes.
        @:param sensi: The minimum sensitivity required for packet transmissions with specific parameters to be received.
        @:param ptx: The default transmission power value to assign.
        @:param placement: The ideal placement based on my and base paper core philosophy.
    """
    def __init__(self, nodes, nrnodes, distributiontype, sensi, ptx, placement):
        self.nodes = nodes
        self.nrNodes = nrnodes
        self.distributionType = distributiontype
        self.sensi = sensi
        self.Ptx = ptx
        self.sfCounts = placement
        self.distanceFinder = maxDistFinder()

    """
        Calculates actual distance between base-station and node given a x, y co-ordinate.
        @:param dist: Node distance from base station.
        @:param bsx: x coord of base station.
        @:param bsy: y coord of base station.
        @:returns x, y, dist: x is node x coord, y is node y coord, dist is node distance to base station.
    """
    @staticmethod
    def base_math(dist, bsx, bsy):
        a = random.random()
        b = random.random()
        if b < a:
            a, b = b, a
        x = b * dist * math.cos(2 * math.pi * a / b) + bsx
        y = b * dist * math.sin(2 * math.pi * a / b) + bsy
        dist = np.sqrt((x - bsx) * (x - bsx) + (y - bsy) * (y - bsy))

        return x, y, dist

    """
        Orchestrates the class logic on node placement based on experiment input.
        @:param maxdist: The max distance a node could possibly transmit.
        @:param bsx: x coord of base station.
        @:param bsy: y coord of base station.
        @:nodeid: The nodes unique identifier.
        @:returns x, y, dist: x is node x coord, y is node y coord, dist is node distance to base station.
    """
    def logic(self, maxdist, bsx, bsy, nodeid):
        x = 0
        y = 0
        dist = 0

        if self.distributionType == "uniform":
            x, y, dist = self.uniform_place(maxdist, bsx, bsy)
        elif self.distributionType == "uniform basic":
            x, y, dist = self.uniform_place_basic(maxdist, bsx, bsy)
        elif self.distributionType == "controlled":
            x, y, dist = self.controlled_place(bsx, bsy, nodeid)

        return x, y, dist

    """
        One form of node placement.
        This form follows a set number of nodes per spreading factor region.
        @:param bsx: x coord of base station.
        @:param bsy: y coord of base station.
        @:nodeid: The nodes unique identifier.
        @:returns x, y, dist: x is node x coord, y is node y coord, dist is node distance to base station.
    """
    def controlled_place(self, bsx, bsy, nodeid):
        x = 0
        y = 0
        dist = -1
        region = -1
        sum = 0
        for i, sfCount in enumerate(self.sfCounts):
            sum += sfCount
            if nodeid < sum:
                region = i
                break
        if region == -1:
            region = len(self.sfCounts)-1

        # currently assuming static txPower of 14dB
        rssi = self.Ptx + (-1 * self.sensi[region, 1])
        region_max_distance = self.distanceFinder.max_distance(rssi)
        if region > 0:
            min_rssi = self.Ptx + (-1 * self.sensi[region - 1, 1])
            region_min_distance = self.distanceFinder.max_distance(min_rssi)
        else:
            region_min_distance = 0  # 0 number 15 introduces ma deadzone which is 12.64 metres

        # Very bad way to account for minimum allowed distance.
        while dist < region_min_distance or dist > region_max_distance:
            x, y, dist = self.base_math(region_max_distance, bsx, bsy)

        return x, y, dist

    """
        Places the node as they currently are (random place).
        @:param maxdist: The max distance a node could possibly transmit.
        @:param bsx: x coord of base station.
        @:param bsy: y coord of base station.
        @:returns x, y, dist: x is node x coord, y is node y coord, dist is node distance to base station.
    """
    def uniform_place_basic(self, max_dist, bsx, bsy):
        x, y, dist = self.base_math(max_dist, bsx, bsy)

        return x, y, dist

    """
            Places node similar to above but ensures nodes are spaced apart.
            @:param maxdist: The max distance a node could possibly transmit.
            @:param bsx: x coord of base station.
            @:param bsy: y coord of base station.
            @:returns x, y, dist: x is node x coord, y is node y coord, dist is node distance to base station.
    """
    def uniform_place(self, max_dist, bsx, bsy):
        found = 0
        rounds = 0
        x = 0.0
        y = 0.0

        while found == 0 and rounds < 100:
            posx, posy, dist = self.base_math(max_dist, bsx, bsy)
            if len(self.nodes) > 0:
                for index, n in enumerate(self.nodes):
                    dist = np.sqrt(((abs(n.x - posx)) ** 2) + ((abs(n.y - posy)) ** 2))
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
        dist = np.sqrt((x - bsx) * (x - bsx) + (y - bsy) * (y - bsy))

        return x, y, dist

"""
    Provides experiment logic.
"""
class experiments:

    """
        Initialisation method.
        @:param xperiment: The experiment that is being simulated.
        @:param nr_channels: How many channels are available for the network to use.
        @:param sensi: Minimum sensitivities for node transmissions to be considered received/viable.
        @:param plen: Packet length.
        @:param gl: Overall transmissin gains/losses.
        @:param ptx: Default transmission power value for nodes.
    """
    def __init__(self, xperiment, nr_channels, sensi, plen, gl, ptx):
        self.experiment = xperiment
        self.esti = estimator()
        self.nrChannels = nr_channels
        self.sensi = sensi
        self.plen = plen
        self.GL = gl
        self.ptx = ptx
        self.sfCounts = [0, 0, 0, 0, 0, 0]
        self.sfs = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

    """
        Logic on which experiments to run depending on user input.
        @:param nodes: List of node instances.
        @:param ideal: The ideal node distribution to achieve based on core philosophy.
        @:param truth: The true node distribution.
        @:param start: follow through parameter used for a specific experiment where this method gets recalled.
    """
    def logic(self, nodes, ideal, truth, start):
        if self.experiment == 1:
            self.basic_experiment(nodes, 12, 4, 125)
        elif self.experiment == 2:
            self.basic_experiment(nodes, 7, 1, 125)
        elif self.experiment == 3:
            self.basic_experiment(nodes, 12, 1, 125)
        elif self.experiment == 4:
            self.experiment_four(nodes)
        elif self.experiment == 5:
            self.experiment_five(nodes, ideal, truth, [], len(nodes))
        elif self.experiment ==  6:
            sf_assigns = self.experiment_six(nodes, len(nodes), start)
            return sf_assigns
        elif self.experiment == 7:
            self.experiment_seven(nodes)
        elif self.experiment == 8:
            self.experiment_eight(nodes)
        else:
            print("Invalid experiment!\nQuitting!")
            quit()

    """
        Basic experiment supplied by original LoRaSim.
        @:param nodes: List of node instances.
        @:param sf: Spreading factor value.
        @:param cr: Coding rate value.
        @:param bw: Bandwidth value.
    """
    def basic_experiment(self, nodes, sf, cr, bw):
        for node in nodes:
            ch = random.randint(0, self.nrChannels - 1)
            rectime = self.esti.airtime(sf, cr, self.plen, bw)
            node.packet.phase_two(sf, cr, bw, ch, rectime)
            self.sfCounts[sf - 7] += 1

    """
        RSSI based method of assignin node parameters.
        Assigns parameters so that node transmissions have shortest viable airtime.
        I have implemented a code representaion of the RSSI solution found in several papers.
        @:param nodes: List of node instances.
    """
    def experiment_four(self, nodes):
        for node in nodes:
            ch = random.randint(0, self.nrChannels - 1)
            minairtime = 9999
            sf = 0
            for i in range(0, 6):
                if self.sensi[i, 1] <= (self.ptx - self.GL - node.packet.Lpl):
                    sf = int(self.sensi[i, 0])
                    minairtime = self.esti.airtime(sf, 1, self.plen, 125)
                    break
            if minairtime == 9999:
                print("does not reach base station")
                exit(-1)

            rectime = self.esti.airtime(sf, 1, self.plen, 125)
            node.packet.phase_two(sf, 1, 125, ch, rectime)
            self.sfCounts[sf - 7] += 1

    """
        My Solution
        @:param nodes: List of node instances.
        @:param ideal: The desired number of nodes per spreading factor group.
        @:param truth: Number of nodes available to be assigned to each spreading factor group.
        @:param actual: A list for the method to assign values to, this is fedback to method call.
        @:param nr_nodes: How many nodes are in experiment.
    """
    def experiment_five(self, nodes, ideal, truth, actual, nr_nodes):
        sf_possible = [0, 0, 0, 0, 0, 0]
        temp_total = 0
        for i, amt in enumerate(truth):
            temp_total += amt
            sf_possible[i] = temp_total

        used_total = 0
        for i, total in enumerate(sf_possible):
            difference = total - used_total
            if ideal[i] <= difference:
                actual.append(ideal[i])
                used_total += ideal[i]
            else:
                actual.append(difference)
                used_total += difference
                fair_sf_getter = fairSF(nr_nodes - used_total, self.sfs[i + 1:])
                ideal = ideal[:i + 1] + fair_sf_getter.get_sf_counts()

                if i > 0:
                    ratio = float(actual[i]) / float(ideal[i])
                    equivalent = ratio * ideal[i - 1]
                    if equivalent < actual[i - 1]:
                        split_total = actual[i] + actual[i - 1]
                        fair_sf_getter = fairSF(split_total, self.sfs[i - 1:i + 1])
                        split_ideal = fair_sf_getter.get_sf_counts()
                        actual[i - 1] = split_ideal[0]
                        actual[i] = split_ideal[1]

        # did I delete a sort line somewhere?
        nodes.sort()

        counter = 0
        for i, count in enumerate(actual):
            for j in range(count):
                sf = i + 7
                ch = random.randint(0, self.nrChannels - 1)
                rectime = self.esti.airtime(sf, 1, self.plen, 125)
                nodes[counter].packet.phase_two(sf, 1, 125, ch, rectime)
                self.sfCounts[sf - 7] += 1
                counter += 1

    """
        OG solution or FADR!
        Code implementation of one of the papers I based my work on.
        @:param nodes: List of node instances.
        @:param nr_nodes: Number of nodes in the experiment.
        @:param start: Extra parameter for specific experiment where this method gets recalled.
    """
    def experiment_six(self, nodes, nr_nodes, start):
        nodes.sort()
        validate2 = False
        fair_sf_getter = fairSF(nr_nodes - start, self.sfs)
        sf_assigns = fair_sf_getter.get_sf_counts()

        for i in range(0, start):
            ch = random.randint(0, self.nrChannels - 1)
            rectime = self.esti.airtime(7, 1, self.plen, 125)
            self.sfCounts[nodes[i].packet.sf-7] -= 1
            nodes[i].packet.phase_two(7, 1, 125, ch, rectime)
            self.sfCounts[nodes[i].packet.sf-7] += 1
            nodes[i].packet.phase_three(2.0)

        total = 0
        for i, item in enumerate(sf_assigns):
            for number in range(item):
                sf = i+7
                ch = random.randint(0, self.nrChannels - 1)
                rectime = self.esti.airtime(i+7, 1, self.plen, 125)
                nodes[total+start].packet.phase_two(sf, 1, 125, ch, rectime)
                self.sfCounts[sf - 7] += 1
                total += 1

        # reassigns sf to nodes to validate transmissions. Not going to use.
        # This is an extension I made to their solution to improve it for real world use.
        if validate2:
            for i, node in enumerate(nodes):
                minsensi = self.sensi[node.packet.sf - 7, 1]
                if 14 - node.Lpl < minsensi:
                    for j in range(node.packet.sf - 6, len(self.sensi)):
                        threshold = self.sensi[j, 1]
                        if 14 - node.Lpl > threshold:
                            sf = j + 7
                            ch = random.randint(0, self.nrChannels - 1)
                            rectime = self.esti.airtime(j + 7, 1, self.plen, 125)
                            self.sfCounts[node.packet.sf-7] -= 1
                            node.packet.phase_two(sf, 1, 125, ch, rectime)
                            self.sfCounts[node.packet.sf-7] += 1
                            if i < start:
                                txpow = max(2, self.ptx - math.floor((self.ptx - node.packet.Lpl) - minsensi))
                                node.packet.phase_three(txpow)

        return sf_assigns

    """
    Randomn assign parameters.
    @:param nodes: List of node instances.
    """
    def experiment_seven(self, nodes):
        sf_list = [7, 8, 9, 10, 11, 12]
        for node in nodes:
            sf = random.choice(sf_list)
            ch = random.randint(0, self.nrChannels - 1)
            rectime = self.esti.airtime(sf, 1, self.plen, 125)
            node.packet.phase_two(sf, 1, 125, ch, rectime)
            self.sfCounts[node.packet.sf - 7] += 1

        return

    """
        Random assign parameters with viability check.
        @:param nodes: List of node instances.
    """
    def experiment_eight(self, nodes):
        sf_list = [7, 8, 9, 10, 11, 12]
        for node in nodes:
            for j, sf_thresholds in enumerate(self.sensi):
                threshold = sf_thresholds[1]
                if 14 + node.packet.Lpl > threshold:
                    sf = random.choice(sf_list[j:])
                    ch = random.randint(0, self.nrChannels - 1)
                    rectime = self.esti.airtime(sf, 1, self.plen, 125)
                    node.packet.phase_two(sf, 1, 125, ch, rectime)
                    self.sfCounts[node.packet.sf-7] += 1
                    break

"""
    Logic for assigning transmission powers to nodes.
    Controls node power consumption.
"""
class powerControl:

    """
        Initialisation method.
        @:param power_scheme: The power scheme to follow.
        @:param sensi: List of sensitivity values for a transmission to be considered viable/received.
        @:param sensi_diff: List of sensitivity combination differences required for the capture effect.
        @:param gl: Overall transmission gains/losses.
        @:param ptx: Default transmission power value for nodes.
    """
    def __init__(self, power_scheme, sensi, sensi_diff, gl, ptx):
        self.powerScheme = power_scheme
        self.sensi = sensi
        self.sensiDiff = sensi_diff
        self.GL = gl
        self.ptx = ptx
        self.atrGet = operator.attrgetter

    """
        Provides logic for which power schemes to use depending on user experiment input.
        @:param nodes: List of node instances.
        @:param experi_logic: Which experiment is being simulation.
    """
    def logic(self, nodes, experi_logic):
        if self.powerScheme == 1:
            self.power_one(nodes)
        elif self.powerScheme == 2:
            self.power_two(nodes)
        elif self.powerScheme == 3:
            self.power_three(nodes, experi_logic)
        else:
            for node in nodes:
                node.packet.phase_three(self.ptx)

    """
        My transmission power scheme (TPS) method.
        @:param nodes: List of node instances.
    """
    def power_one(self, nodes):
        for node in nodes:
            minsensi = self.sensi[node.packet.sf - 7, 1]
            txpow = max(2, self.ptx - math.floor((self.ptx - node.packet.Lpl) - minsensi))
            if txpow > 14:
                txpow = 14
            node.packet.phase_three(txpow)

    """
        FADR - Fair Adaptive Data Rate
        @:param nodes: List of node instances.
    """
    def power_two(self, nodes):
        # First sort nodes by RSSI, done with __lt__ method on node class.
        nodes.sort()
        for n in nodes:
            n.packet.phase_three(14)

        # get max/min RSSI and min CIR (inter SF collision?)
        min_rssi_node = min(nodes, key=self.atrGet('packet.rssi'))#.packet.rssi
        max_rssi_node = max(nodes, key=self.atrGet('packet.rssi'))#.packet.rssi
        min_cir = 8

        # Find range of power levels to use
        # need to rewrite this !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        power_levels = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        min_power = power_levels.pop(0)
        max_power = None
        for i, power_level in enumerate(power_levels):
            max_power = power_level
            if abs(((-1*max_rssi_node.packet.Lpl) + min_power) - ((-1*min_rssi_node.packet.Lpl) + max_power)) <= min_cir:
                power_levels = power_levels[0: i]
                break
            elif power_level == max(power_levels):
                max_power = max(power_levels)

        # Recalc min_rssi, max_rssi
        nodes[-1].packet.phase_three(max_power) # min node
        nodes[0].packet.phase_three(min_power) # max node
        temp_node = None
        if nodes[-1].packet.rssi > nodes[0].packet.rssi:
            min_rssi_node = nodes[0]
            max_rssi_node = nodes[-1]

        # Assign minimum power and save minPowerIndex
        min_power_index = None
        for i, n in enumerate(nodes):
            if abs(min_power - n.packet.Lpl) > abs(min_rssi_node.packet.rssi):
                min_power_index = i - 1
                break
            else:
                n.packet.phase_three(min_power)

        # Assign maximum power and save maxPowerIndex
        max_power_index = None
        for i in range(len(nodes)-1, min_power_index, -1):
            if abs(abs(max_power - nodes[i].packet.Lpl) - abs(min_rssi_node.packet.rssi)) > min_cir:
                max_power_index = i - 1
                break
            else:
                nodes[i].packet.phase_three(max_power)

        # Assign the remaining power levels to the inbetween node
        if (max_power_index < min_power_index):
            print ("hit weird error")
            quit()
        start_rssi = min_rssi_node.packet.rssi
        temp_index = min_power_index
        assign_loop = True
        index = 0
        while assign_loop and index < len(power_levels):
            power_level = power_levels[index]
            for i in range(temp_index, max_power_index):
                if (abs(power_level - nodes[i].packet.Lpl + abs(start_rssi)) > min_cir):
                    temp_index = i
                    start_rssi = nodes[temp_index].packet.rssi
                    break
                else:
                    nodes[i].packet.phase_three(power_level)
                if (i == max_power_index-1):
                    assign_loop = False
            index += 1

        return

    """
        Original transmission power scheme solution from paper where I got my core insights.
        @:param nodes: List of node instances.
        @:param experi_logic: Which experiment is being simulated.
    """
    def power_three(self, nodes, experi_logic):
        validate = False
        nodes.sort()

        start = 0
        looping = True
        while looping:
            sf_assigns = experi_logic.logic(nodes, [], [], start)
            last_sf8 = start + sf_assigns[0] + sf_assigns[1] - 1

            node_a = nodes[last_sf8]
            node_b = nodes[start]
            cir = self.sensiDiff[node_a.packet.sf - 7][node_b.packet.sf - 7]

            if 14 - node_a.packet.Lpl - cir > 2 - node_b.packet.Lpl:
                looping = False
                # Validate here, if is for error check. else is standard.
                if validate:
                    minsensi = self.sensi[nodes[start].packet.sf - 7, 1]
                    new_txpow = max(2, self.ptx - math.floor((self.ptx - nodes[start].packet.Lpl) - minsensi))
                    if new_txpow > 14.0:
                        new_txpow = 14
                    nodes[start].packet.phase_three(new_txpow)
                else:
                    nodes[start].packet.phase_three(2)
                nodes[last_sf8].packet.phase_three(14)
            else:
                experi_logic.logic(nodes, [], [], start)
            start += 1

        # Assign power levels, Need to add a viable check.
        nodes.reverse()
        first_sf8 = (((start-1) + sf_assigns[0])*-1)-1
        for i in range(0, (len(nodes) - start)):
            if nodes[i].packet.sf == 7:
                # Make sure first sf8 node does not interfer with sf7 node.
                node_a = nodes[i]
                node_b = nodes[first_sf8]
            else:
                # Make sure nearest node will not interfere with current node.
                node_a = nodes[i]
                node_b = nodes[start*-1]

            cir = self.sensiDiff[node_a.packet.sf - 7][node_b.packet.sf - 7]
            diff = (14 - node_a.packet.Lpl) - (node_b.packet.txpow - node_b.packet.Lpl)
            if diff > cir:
                diff2 = cir - diff
                # Validate here.
                if validate:
                    minsensi = self.sensi[node_a.packet.sf - 7, 1]
                    new_txpow = max(14 - abs(diff2), \
                                    self.ptx - math.floor((self.ptx - node_a.packet.Lpl) - minsensi))
                    if new_txpow > 14.0:
                        new_txpow = 14
                else:
                    new_txpow = max(2, 14 - math.floor(abs(diff2)))
                    if new_txpow > 14.0:
                        new_txpow = 14
                node_a.packet.phase_three(new_txpow)
            else:
                nodes[i].packet.phase_three(14)
        """
            minsensi = self.sensi[node.packet.sf - 7, 1]
            txpow = max(2, self.ptx - math.floor((self.ptx - node.packet.Lpl) - minsensi))
            node.packet.phase_three(txpow)
        """

    """
        Random assign power levels.
        @:param nodes: List of node instances.
    """
    def power_four(self, nodes):
        power_levels = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        for node in nodes:
            new_txpow = random.choice(power_levels)
            node.packet.phase_three(new_txpow)

    """
        random assign power levels with viability check.
        @:param nodes: List of node instances.
    """
    def power_five(self, nodes):
        power_levels = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        for node in nodes:
            for j, txpow in enumerate(power_levels):
                threshold = self.sensi[node.packet.sf-7][1]
                if txpow - node.packet.Lpl > threshold:
                    new_txpow = random.choice(power_levels[j:])
                    node.packet.phase_three(new_txpow)

    # Use input file Tp
    def power_six(self, nodes):
        for node in nodes:
            node.packet.phase_three(self.ptx)

"""
    An Observer class to get insights on channel usage.
    A very efficient way to observer the channel so that this code is only ran when needed.
    Allows for detailed information on % of channel time used.
    @:param Object: Object instance for auto updates so that this class can monitor the simulation.
"""
class channelUsage(object):
    """
        Initialisation method.
    """
    def __init__(self):
        # self.noTraffic = 0.0
        self._traffic = 0
        self.empty = False
        self.f_flag = 0.0
        self.e_flag = 0.0
        self.accum_e = 0.0
        self.accum_f = 0.0

    """
        Property method for channelUsage instances.
    """
    @property
    def traffic(self):
        return self._traffic

    """
        When correct event occurs, traffic value is updated.
        Includes logic to determine when packets are starting/finishing to get correct stats.
        @:param value: The time value to use to update channel usage statistics.
    """
    @traffic.setter
    def traffic(self, value):
        self._traffic = value

        if self.traffic == 0.0 and not self.empty:
            self.empty = True
            self.e_flag = time.time()
            if self.f_flag > 0.0:
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

Finds maximum viable distance for LoRaWAN node.
"""
class maxDistFinder:


    """
    Initialisation method.
    Was not needed.
    """
    def __init__(self):
        return

    """
        This methods finds whether a given nodes packets can reach the base-station.
        This method also returns the minimum viable spreading factor.
        @:param max_loss: The maximum loss that can be had before packet is lost.
        @:return distance: Maximum distance a node can transmit with its current parameters.
    """
    @staticmethod
    def max_distance(max_loss):
        distance = 40 * 10 ** ((max_loss - 127.41) / 20.8)

        return distance

"""
    Code implementation of core math logic from core paper that I based much of my research on.
"""
class fairSF:

    """
        Initialisation method.
        @:param nr_nodes: Number of nodes in the experiment.
        @:param sf_list: List of spreading factors being used in the experiment.
    """
    def __init__(self, nr_nodes, sf_list):
        self.nrNodes = nr_nodes
        self.sfList = sf_list
        self.baseResult = self.base_function
        return

    """
        Provides final calculation.
        @:return sum_result: Base function for other functions for core philosophy.
    """
    @property
    def base_function(self):
        sum_result = 0.0

        for sf in self.sfList:
            sum_result += sf / (2 ** sf)

        return sum_result

    """
        Gets the number of nodes to assign to each spreading factor region.
        This is based on the logic of the fairness equation.
        @:return sf_counts: number of nodes to ideally assign to each spreading factor group.
    """
    def get_sf_counts(self):
        sf_counts = []
        total = 0

        sf_percentages = self.get_percentages()
        before_round = []
        for sfP in sf_percentages:
            temp_count = int(round(sfP * self.nrNodes))
            before_round.append(sfP * self.nrNodes)
            sf_counts.append(temp_count)
            total += temp_count

        difference = self.nrNodes - total
        sf_counts[0] += difference
        return sf_counts

    """
        Gets above as percentages.
        @:return sf_percentages: Percentage of total node pool to ideally assign to each spreading factor group.
    """
    def get_percentages(self):
        sf_percentages = []

        for sf in self.sfList:
            sf_percentages.append(self.get_percentage(sf))

        return sf_percentages

    """
        Gets a single percentage.
        @:return sf_percentage: Percentage of nodes to ideally assign to current spreading factor group.
    """
    def get_percentage(self, sf):
        sf_percentage = (sf / (2 ** sf)) / self.baseResult

        return sf_percentage

"""
    Seperate class to estimate LoRaWAN packet airtimes.
    Mostly implemented this to firm my own understanding of this area.
"""
class estimator:

    """
        This function computes the airtime of a packet
        According to LoraDesignGuide_STD.pdf
    """
    def __init__(self):
        pass

    """
        Calulates airtime of LoRaWAN packet given inputs.
        @:param sf: Spreading Factor value.
        @:param cr: Coding rate value.
        @:param pl: Packet length value.
        @:param bw: Bandwidth parameter.
        @:return t_pream + t_payload: Total aitime for packet.
    """
    @staticmethod
    def airtime(sf, cr, pl, bw):
        h = 0  # implicit header disabled (H=0) or not (H=1)
        de = 0  # low data rate optimization enabled (=1) or not (=0)
        n_pream = 8  # number of preamble symbol (12.25  from Utz paper)

        if bw == 125 and sf in [11, 12]:
            # low data rate optimization mandated for BW125 with SF11 and SF12
            de = 1
        if sf == 6:
            # can only have implicit header with SF6
            h = 1

        t_sym = (2.0 ** sf) / bw
        t_pream = (n_pream + 4.25) * t_sym
        # print "sf", sf, " cr", cr, "pl", pl, "bw", bw
        payload_symb_nb = 8 + max(math.ceil((8.0 * pl - 4.0 * sf + 28 + 16 - 20 * h)
                                            / (4.0 * (sf - 2 * de))) * (cr + 4), 0)
        t_payload = payload_symb_nb * t_sym
        return t_pream + t_payload

    """
        Calculates chirp time for a given LoRaWAN packet.
        @:param sf: Spreading Factor value.
        @:param bw: Bandwidth parameter.
        @:return chirpy_time: Time to send a chirp.
    """
    @staticmethod
    def chirp_time(sf, bw):
        chirpy_time = (2 ** sf) / bw
        return chirpy_time

    """
        Okumura-Hata path loss model.
        @:param sensi: List of sensitivity values for a transmission to be considered viable/received.
    """
    @staticmethod
    def hata_urban(sensi):
        path_loss = 17.5 - sensi
        d = 10 ** ((path_loss - 124.76) / 35.22)
        print(d)
