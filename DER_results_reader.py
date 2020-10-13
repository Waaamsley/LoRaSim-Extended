import sys

inputFile = open(sys.argv[1], "r")
outputFile = open("distillied_" + sys.argv[1][12:], 'a')

index = 0
magic_number = 22 #rssi is 24
sub_number = 20 #rssi is 23
for line in inputFile.readlines():
    if (((index - sub_number) % magic_number == 0 and index >= magic_number) or index == sub_number):
        #print (index, line)
        outputFile.write(line)
    index += 1