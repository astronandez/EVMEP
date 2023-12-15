
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

print('hello1')

# Load your TensorFlow Lite model
# Make sure to provide the correct filename
model_path = 'quantized_model.tflite'
net = tf.load(model_path, load_to_fb=True)  # Load model into heap instead of frame buffer

print('hello2')

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

