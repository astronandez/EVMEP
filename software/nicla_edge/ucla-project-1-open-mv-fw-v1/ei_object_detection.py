# Edge Impulse - OpenMV Object Detection

import sensor, image, time, os, tf, math, uos, gc

sensor.reset()                         # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565)    # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)      # Set frame size to QVGA (320x240)
sensor.set_windowing((240, 240))       # Set 240x240 window.
sensor.skip_frames(time=2000)          # Let the camera adjust.

net = None
labels = None
min_confidence = 0.3

try:
    # Load built in model
    labels, net = tf.load_builtin_model('trained')
except Exception as e:
    raise Exception(e)

colors = [ # Add more colors if you are detecting more than 7 types of classes at once.
    (255,   0,   0),
    (255, 128,   0),
    (255, 255,   0),
    (128, 255,   0),
    (  0, 255,   0),
    (  0, 255, 128),
    (  0, 255, 255),
    (  0, 128, 255),
    (  0,   0, 255),
    (128,   0, 255),
    (255,   0, 255),
    (255,   0, 128),
    (255,   0, 128),
]

def estimate_vechicle_weight(car_type_label):
    estimated_mass = 0
    if car_type_label == labels[0]:
        estimated_mass = 3092
    elif car_type_label == labels[1]:
        estimated_mass = 4037
    elif car_type_label == labels[2]:
        estimated_mass = 3448
    elif car_type_label == labels[3]:
        estimated_mass = 3466
    elif car_type_label == labels[4]:
        estimated_mass = 2992
    elif car_type_label == labels[5]:
        estimated_mass = 4462
    elif car_type_label == labels[6]:
        estimated_mass = 4451
    elif car_type_label == labels[7]:
        estimated_mass = 4433
    elif car_type_label == labels[8]:
        estimated_mass = 4331
    elif car_type_label == labels[9]:
        estimated_mass = 3743
    elif car_type_label == labels[10]:
        estimated_mass = 3494
    elif car_type_label == labels[11]:
        estimated_mass = 3695
    else:
        estimated_mass = 0

    return estimated_mass

clock = time.clock()
while(True):
    clock.tick()

    img = sensor.snapshot()

    # detect() returns all objects found in the image (splitted out per class already)
    # we skip class index 0, as that is the background, and then draw circles of the center
    # of our objects

    for i, detection_list in enumerate(net.detect(img, thresholds=[(math.ceil(min_confidence * 255), 255)])):
        if (i == 0): continue # background class
        if (len(detection_list) == 0): continue # no detections for this class?

        weight = estimate_vechicle_weight(labels[i])
        #print("********** %s **********" % labels[i])

        print("Detected %s, weight: %s" % (labels[i], weight))

        for d in detection_list:
            [x, y, w, h] = d.rect()
            center_x = math.floor(x + (w / 2))
            center_y = math.floor(y + (h / 2))
            print('x %d\ty %d' % (center_x, center_y))
            img.draw_circle((center_x, center_y, 12), color=colors[i], thickness=2)

    #print(clock.fps(), "fps", end="\n\n")
