import networkSupport
import math

nrNodes = 600
sfs = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
fair_sf_getter = networkSupport.fairSF(nrNodes, sfs)
sf_counts = fair_sf_getter.get_sf_counts()
average_nodes_region = nrNodes / len(sfs)
mod = 0.1


def option_one(region_counts, modifier):
    temp = region_counts
    differences = []
    for j in range(0, 3):
        difference = temp[j] - temp[-1 * (j + 1)]
        differences.append(difference)

    changes = []
    for item in differences:
        change = item * modifier
        changes.append(change)

    for i in range(0, 11):
        print i, temp
        for j, item in enumerate(changes):
            temp[j] -= item
            temp[-1 * (j + 1)] += item


def option_two(region_counts, modifier, avg):
    temp = region_counts
    differences = []
    for j in range(0, 6):
        difference = temp[j] - avg
        differences.append(difference)

    changes = []
    for item in differences:
        change = item * modifier
        changes.append(change)

    for i in range(0, 6):
        print i, temp
        for j, item in enumerate(changes):
            temp[j] += -1 * item

    for i in range(0, 6):
        print i, temp
        for j, item in enumerate(changes):
            temp[j] += item


#option_one(sf_counts, mod)
print("-----------------------------------")
option_two(sf_counts, mod, average_nodes_region)
