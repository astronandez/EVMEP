# Make sure you have loaded quantized_model.tflite onto device in finder

import sensor
import image
import tf
import gc

# Camera setup
sensor.reset()
sensor.set_pixformat(sensor.RGB565)  # Use grayscale to save memory
sensor.set_framesize(sensor.QQVGA)      # Lower resolution
sensor.skip_frames(time=2000)

# Disable auto gain and white balance to save computation resources
sensor.set_auto_gain(False)
sensor.set_auto_whitebal(False)



gc.collect()

# Get current memory usage statistics
mem_alloc = gc.mem_alloc()  # Memory currently allocated
mem_free = gc.mem_free()    # Memory currently free

# Print memory usage
print("Memory Allocated: {} bytes".format(mem_alloc))
print("Memory Free: {} bytes".format(mem_free))
print("Total Memory: {} bytes".format(mem_alloc + mem_free))




# Load your TensorFlow Lite model
# Make sure to provide the correct filename
model_path = 'quantized_model.tflite'
net = tf.load(model_path, load_to_fb=False)  # Load model into heap instead of frame buffer

# Main loop
while True:
    print('hello3')
    img = sensor.snapshot()

    # Run object detection
    objs = net.classify(img)

    # Print results
    for obj in objs:
        print('Label: %s, Confidence: %f' % (obj.label(), obj.output()))

    # Garbage collection to free up unused memory
    gc.collect()

