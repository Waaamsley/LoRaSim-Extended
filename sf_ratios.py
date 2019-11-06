import sys
SFS = [7, 8, 9, 10, 11, 12]
SF_PERCENTAGES = [44.9, 25.69, 14.45, 8.03, 4.41, 2.36]


def sf_ratio_group(sf):
    node_eqv = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    my_percentage = SF_PERCENTAGES[sf-7]

    for i in range(6):
        node_eqv[i] = (SF_PERCENTAGES[i] / my_percentage)

    return node_eqv

def sf_ratio_groups():
    ratio_groups = []

    for sf in SFS:
        sf_group = sf_ratio_group(sf)
        ratio_groups.append(sf_group)

    return ratio_groups

def normalise(ratio_group):
    normalised = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    for i in range(6):
        normalised[i] = None # was coding here last, what was I doing???!

    return normalised

def to_normalise(ratio_groups):
    normalised_groups = []

    for group in ratio_groups:
        normalised_group = normalise(group)
        normalised_groups.append(normalised_group)

    return normalised_groups
"""
if (len(sys.argv) == 2):
    sfs = sys.argv[1]
else:
    print("Incorrect input")
    exit()
"""

result = sf_ratio_groups()
result = to_normalise(result)
for index, group in enumerate(result):
    print("SF" + str(index+7) + ": ", group)

exit()