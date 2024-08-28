import pycuda.driver as cuda

cuda.init()
device_count = cuda.Device.count()

for i in range(device_count):
    device = cuda.Device(i)
    print("Device {}: {}".format(i, device.name()))
    print("Memory: {} MB".format(device.total_memory() // (1024 * 1024)))
    print("Compute Capability: {}.{}".format(*device.compute_capability()))