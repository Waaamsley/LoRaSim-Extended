import random
import math
import numpy as np
import time
import operator


class placementGenerator:

    def __init__(self, nr_nodes, region_counts):
        self.nr_nodes = nr_nodes
        self.region_counts = region_counts
        self.modifier = 0.1

    def wave_placement(self):
        configurations = []
        avg = self.nr_nodes/len(self.region_counts)

        differences = []
        for j in range(0, 6):
            difference = self.region_counts[j] - avg
            differences.append(difference)

        changes = []
        for item in differences:
            change = item * self.modifier
            changes.append(change)

        configurations.append(self.region_counts)
        for i in range(0, 10):
            temp = list(configurations[-1])
            for j, item in enumerate(changes):
                temp[j] -= item
                #temp[j] = round(temp[j] - item)
            configurations.append(temp)

        changes.reverse()
        for i in range(0, 10):
            temp = list(configurations[-1])
            for j, item in enumerate(changes):
                temp[j] += item
                #temp[j] = round(temp[j] + item)
            configurations.append(temp)

        return configurations

class nodePlacer:

    def __init__(self, nodes, nrnodes, distributiontype, sensi, ptx, placement):
        self.nodes = nodes
        self.nrNodes = nrnodes
        self.distributionType = distributiontype
        self.sensi = sensi
        self.Ptx = ptx
        self.sfCounts = placement
        self.distanceFinder = maxDistFinder()
        # fair_sf_getter = fairSF(self.nrNodes, [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        # self.sfCounts = fair_sf_getter.get_sf_counts()

        return

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
            region = len(self.sfCounts-1)

        # currently assuming static txPower of 14dB
        rssi = self.Ptx + (-1 * self.sensi[region, 1])
        region_max_distance = self.distanceFinder.max_distance(rssi)
        if region > 0:
            min_rssi = self.Ptx + (-1 * self.sensi[region - 1, 1])
            region_min_distance = self.distanceFinder.max_distance(min_rssi)
        else:
            region_min_distance = 15  # 0 number 15 introduces ma deadzone which is 12.64 metres

        # Very bad way to account for minimum allowed distance.
        while dist < region_min_distance or dist > region_max_distance:
            x, y, dist = self.base_math(region_max_distance, bsx, bsy)

        return x, y, dist

    def uniform_place_basic(self, max_dist, bsx, bsy):
        x, y, dist = self.base_math(max_dist, bsx, bsy)

        return x, y, dist

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


class experiments:

    def __init__(self, xperiment, nr_channels, sensi, plen, gl):
        self.experiment = xperiment
        self.esti = estimator()
        self.nrChannels = nr_channels
        self.sensi = sensi
        self.plen = plen
        self.GL = gl
        self.sfCounts = [0, 0, 0, 0, 0, 0]
        self.powerControl = [7, 8, 9, 10, 11, 12]

    def logic(self, prx):
        sf = 0
        cr = 0
        bw = 0
        ch = random.randint(0, self.nrChannels - 1)

        if self.experiment == 1:
            sf, cr, bw = self.experiment_one()
        elif self.experiment == 2:
            sf, cr, bw = self.experiment_two()
        elif self.experiment == 3:
            sf, cr, bw = self.experiment_three()
        elif self.experiment == 4:
            sf, cr, bw = self.experiment_four(prx)
        else:
            print("Invalid experiment!\nQuitting!")
            quit()

        rectime = self.esti.airtime(sf, 1, self.plen, bw)
        if self.experiment == 1:
            rectime = self.esti.airtime(sf, 4, self.plen, bw)

        self.sfCounts[sf - 7] += 1
        return sf, cr, bw, ch, rectime

    @staticmethod
    def experiment_one():
        return 12, 4, 125

    @staticmethod
    def experiment_two():
        return 7, 1, 125

    @staticmethod
    def experiment_three():
        return 12, 1, 125

    def experiment_four(self, prx):
        minairtime = 9999
        sf = 0
        bw = 125
        cr = 1

        for i in range(0, 6):
            if self.sensi[i, 1] <= prx:
                sf = int(self.sensi[i, 0])
                minairtime = self.esti.airtime(sf, 1, self.plen, bw)
                break
        if minairtime == 9999:
            print "does not reach base station"
            exit(-1)

        return sf, cr, bw


class powerControl:

    def __init__(self, power_scheme, sensi, sensi_diff, gl):
        self.powerScheme = power_scheme
        self.sensi = sensi
        self.sensiDiff = sensi_diff
        self.GL = gl
        self.atrGet = operator.attrgetter

    def logic(self, nodes):
        if self.powerScheme == 1:
            self.power_one(nodes)
        elif self.powerScheme == 2:
            self.power_two(nodes)
        elif self.powerScheme == 3:
            self.power_three(nodes)
        else:
            return

    # will have to reset txpow, Prx, rssi
    def power_one(self, nodes):
        for node in nodes:
            minsensi = self.sensi[node.packet.sf - 7, 1]
            lpl = node.packet.Lpl
            txpow = node.packet.txpow
            prx = node.packet.prx
            txpow = max(2, txpow - math.floor(prx - minsensi))
            prx = txpow - self.GL - lpl
            node.packet.txpow = txpow
            node.packet.prx = prx
            node.packet.rssi = prx

    # FADR - Fair Adaptive Data Rate
    # I have implemented their power control system
    def power_two(self, nodes):
        # First sort nodes by RSSI, done with __lt__ method on node class.
        nodes_sorted = nodes
        nodes_sorted.sort()

        # get max/min RSSI and min CIR (inter SF collision?)
        min_rssi = min(nodes_sorted, key=self.atrGet('packet.rssi')).packet.rssi
        max_rssi = max(nodes_sorted, key=self.atrGet('packet.rssi')).packet.rssi
        min_cir = 8

        # Find range of power levels to use
        power_levels = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        min_power = power_levels.pop(0)
        max_power = None
        print min_rssi, max_rssi
        for i, power_level in enumerate(power_levels):
            max_power = power_level
            if (max_rssi + min_power - min_rssi - max_power) <= min_cir:
                power_levels = power_levels[0: i]
                break
            elif power_level == max(power_levels):
                max_power = power_levels.pop()

        # Recalc min_rssi, max_rssi
        min_rssi = min(min_rssi + max_power, max_rssi + min_power)
        # max_rssi = max(min_rssi + max_power, max_rssi + min_power). Need to revisit why this is calced.

        # Assign minimum power and save minPowerIndex
        min_power_index = None
        for i, n in enumerate(nodes_sorted):
            if n.packet.rssi + min_power > min_rssi:
                min_power_index = i - 1
                print ("here", i)
                break
            else:
                n.packet.txpow = min_power
                prx = n.packet.txpow - self.GL - n.packet.Lpl
                n.packet.prx = prx
                n.packet.rssi = prx

        # Assign maximum power and save maxPowerIndex
        max_power_index = None
        for i, n in enumerate(reversed(nodes_sorted)):
            if n.packet.rssi + max_power - min_rssi > min_cir:
                max_power_index = i - 1
                break
            else:
                n.packet.txpow = max_power
                prx = n.packet.txpow - self.GL - n.packet.Lpl
                n.packet.prx = prx
                n.packet.rssi = prx

        # Assign the reaming power levels to the inbetween nodes
        temp_index = min_power_index
        max_node_rssi = nodes_sorted[max_power_index].packet.rssi
        for power_level in power_levels:
            temp_node_rssi = nodes_sorted[temp_index].packet.rssi
            if temp_node_rssi + power_level - min_rssi <= min_cir \
                    and temp_node_rssi + power_level - max_node_rssi - max_power <= min_cir:
                for i in range(temp_index, max_power_index):
                    curr_node_rssi = nodes_sorted[i].packet.rssi
                    if curr_node_rssi + power_level - max_node_rssi - max_power > min_cir:
                        temp_index = i - 1
                        break
                    else:
                        nodes_sorted[i].packet.txpow = power_level
                        prx = math.ceil(nodes_sorted[i].packet.txpow - self.GL - nodes_sorted[i].packet.Lpl)
                        nodes_sorted[i].packet.prx = prx
                        nodes_sorted[i].packet.rssi = prx
        return

    def power_three(self, nodes):
        # First sort nodes by RSSI, done with __lt__ method on node class.
        nodes_sorted = nodes
        nodes_sorted.sort()
        nodes_sorted.reverse()

        start = 0
        while True:
            first_sf8 = 0
            last_sf8 = 0
            for i, n in enumerate(nodes_sorted, start):
                # Get first sf8 node for later.
                if n.packet.sf == 7:  # and nodesSorted[-1].packet.sf == 8
                    first_sf8 = i - 1
                    break
                # Main point of this for loop is to get last sf8 node.
                if n.packet.sf == 8 and nodes_sorted[i - 1].packet.sf == 9:
                    last_sf8 = i
            start += 1

            node_a = nodes_sorted[start * -1]
            node_b = nodes_sorted[last_sf8]
            cir = self.sensiDiff[node_a.packet.sf - 7][node_a.packet.sf - 7]
            if 2 - node_a.packet.Lpl - (14 - node_b.packet.Lpl) < abs(cir):
                break
            print node_a.packet.Lpl, node_b.packet.Lpl, cir
            print ("HEREEEEE, power allocation was not viable.")
            print (2 - node_a.packet.Lpl, 14 - node_b.packet.Lpl, cir)
            quit()
            # Need to reapply spreading factors
            # Will have to do the replacement phase (or do i?)

        # Assign power levels
        for i, n in enumerate(nodes_sorted, start):
            txpow = n.packet.txpow
            if n.packet.sf == 7:
                node_a = nodes_sorted[first_sf8]
                cir = self.sensiDiff[n.packet.sf - 7][node_a.packet.sf - 7]
                if n.packet.rssi > node_a.packet.rssi:
                    txpow = 2
                else:
                    difference = node_a.packet.rssi - n.packet.rssi
                    # cir is negative value.
                    cirdiff = difference + cir
                    txpow = max(2, txpow + cirdiff)
            else:
                node_a = nodes_sorted[start * -1]
                if n.packet.rssi > node_a.packet.rssi:
                    txpow = 2
                else:
                    difference = node_a.packet.rssi - n.packet.rssi
                    # cir is negative value.
                    cirdiff = difference + cir
                    txpow = max(2, txpow + cirdiff)
            n.packet.txpow = txpow
            prx = n.packet.txpow - self.GL - n.packet.Lpl
            n.packet.prx = prx
            n.packet.rssi = prx
            # print "testing:", n.packet.sf, cir, txpow, Prx
        return


class channelUsage(object):
    def __init__(self):
        # self.noTraffic = 0.0
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
"""


class maxDistFinder:
    """
    Initialisation method
    """

    def __init__(self):
        return

    """
    This methods finds whether a given nodes packets can reach the base-station.
    This method also returns the minimum viable spreading factor.
    """

    @staticmethod
    def max_distance(max_loss):
        distance = 40 * 10 ** ((max_loss - 127.41) / 20.8)

        return distance


class fairSF:

    def __init__(self, nr_nodes, sf_list):
        self.nrNodes = nr_nodes
        self.sfList = sf_list
        self.baseResult = self.base_function
        return

    @property
    def base_function(self):
        sum_result = 0.0

        for sf in self.sfList:
            sum_result += sf / (2 ** sf)

        return sum_result

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

        difference = total - self.nrNodes
        if difference != 0:
            print("Round off error!!!!! total - nrNodes Difference : ", difference)
            print("before Round: ", before_round, "sfCounts: ", sf_counts, "\nnrNodes", self.nrNodes)
            quit()
        # if difference > 0:
        # subtract nodes from regions
        # elif difference < 0:
        # add nodes to region

        return sf_counts

    def get_percentages(self):
        sf_percentages = []

        for sf in self.sfList:
            sf_percentages.append(self.get_percentage(sf))

        return sf_percentages

    def get_percentage(self, sf):
        sf_percentage = (sf / (2 ** sf)) / self.baseResult

        return sf_percentage


class estimator:

    # this function computes the airtime of a packet
    # according to LoraDesignGuide_STD.pdf
    def __init__(self):
        pass

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

    @staticmethod
    def chirp_time(sf, bw):
        chirpy_time = (2 ** sf) / bw
        return chirpy_time

    # Okumura-Hata path loss model.
    @staticmethod
    def hata_urban(sensi):
        path_loss = 17.5 - sensi
        d = 10 ** ((path_loss - 124.76) / 35.22)
        print(d)
