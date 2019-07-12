# LoRaSim-Extended
Backups of My adapted version of LoRaSim

code snippet:

def channel_usage():
    s_flag = 0.0
    accum = 0.0
    empty = False
    while True:
        if (len(packetsAtBS) == 0 and not empty):
            empty = True
            s_flag = time.time()

        if (len(packetsAtBS) > 0 and empty):
            empty = False
            accum += (time.time() - s_flag)
