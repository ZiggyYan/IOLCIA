from queue import *


# Initialization
num_symbol = 10
num_time = 20

symbol_queue = queue.Queue(maxsize=num_symbol)
time_queue = queue.Queue(maxsize=num_time)

confidence_threhold = 0.7
similarity_time_threhold = xxx
similarity_symbol_threhold = xxx


# DURING INFERENCE
prediction = ....
# symbolic part
for i in range(symbol_queue.qsize()):
    item = symbol_queue.get()
    symbol_queue.put(item)
    # PROCESS
    if [similarity] > similarity_symbol_threhold:
        prediction&item....
# time part
for i in range(time_queue.qsize()):
    item = time_queue.get()
    time_queue.put(item)
    time_gap = time_queue.qsize() - i
    # PROCESS
    if [similarity] > similarity_time_threhold:s
        
# AFTER INFERENCE
# ADD
# Calculate Confidence
confidence = ___
if confidence>symbolic_confidence_threhold:
    if symbol_queue.full():
        itm = symbol_queue.get()
    symbol_queue.put(prediction)
if time_queue.full():
    itm = time_queue.get()
time_queue.put(prediction)