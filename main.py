import os
import glob
import cv2
import nd2
from multiprocessing import Pool
from src.common import Pipeline
from src.loader import ImageContainer 
from src.preprocessing import *
from src.segmentation import SegmentAnything



def process_image(filename, pipeline):

    print('===== FUNCTION process_image =====')

    # load image from file
    img = ImageContainer(filename)

    print(type(img.image))
    cv2.imshow(img.filename, img.image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    
    # process image through pipeline
    processed_image = pipeline.process(img)
    
    # save processed image to file or something
    pass



def get_min_and_max(img, saturated=0.35):
    print('===== FUNCTION get_min_and_max =====')


    histogram, bins = np.histogram(img.flatten(), bins=256, range=(img.min(), img.max()))

    hmin, hmax = 0, 255
    threshold = int(img.size * saturated / 200.0)
    print('threshold: ', threshold)
    count = 0
    for i in range(len(histogram)):
        count += histogram[i]
        if count > threshold:
            hmin = i
            break
    count = 0
    for i in range(len(histogram)-1, -1, -1):
        count += histogram[i]
        if count > threshold:
            hmax = i
            break

    bin_size = bins[1] - bins[0]

    low_value = bins[hmin + 1] + hmin * bin_size
    high_value = bins[hmax + 1] + hmax * bin_size

    return low_value, high_value


def auto_contrast(img):
    """
    Apply auto-contrast adjustment to an image.
    """

    # low_value = 1278.0
    # high_value = 5931.0

    # low_value = 1254.84375
    # high_value = 5970.1875

    low_value, high_value = get_min_and_max(img)

    print('low_value: ', low_value)
    print('high_value: ', high_value)

    # low_value = 1200
    # high_value = 9600

    # Scale the image to 0-255 range
    scaled_image = (img - low_value) * (255.0 / (high_value - low_value))
    scaled_image[scaled_image < 0] = 0
    scaled_image[scaled_image > 255] = 255
    scaled_image = scaled_image.astype(np.uint8)

    return scaled_image

  
if __name__ == '__main__':
    folder = r'C:\Users\rlpri\Downloads\input'
    filenames = glob.glob(os.path.join(folder, '*.nd2'))

    print((os.path.join(folder, '*.nd2')))

    print(filenames)

    nodes = [
        # set channel as
        # SelectChannel(channel=1),
        SegmentAnything(channel=1),
        # CLAHEContrast(),
        # ShowImage()
    ]
    pipeline = Pipeline(nodes)
    
    with Pool() as p:
        p.starmap(process_image, [(filename, pipeline) for filename in filenames])



# TODO: Build Pipeline from Config

# def load_config(config_file):

#     # print('===== FUNCTION load_config =====')

#     with open(config_file) as f:
#         config = json.load(f)
    
#     print(config)

#     steps = []
#     for step_config in config["steps"]:
#         # print(step_config)

#         module_name = importlib.import_module(step_config["module"])
#         # print(module_name)

#         class_name = getattr(module_name, step_config["class"])
#         # print(class_name)

#         parameters = step_config.get("parameters")
#         # print(parameters)

#         instance = class_name(**parameters)
#         # print(instance)

#         steps.append({"name": step_config["name"], "instance": instance})
    
#     config['steps'] = steps

#     print(config)

#     return config

# # Parse command-line arguments
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--config", "-c", 
#     default="config.json", 
#     help="configuration file"
# )
# args = parser.parse_args()

# # Load the steps from the configuration file
# steps = load_config(args.config)

# # Build the pipeline
# pipeline = Pipeline()
# for step in steps:
#     print(step)
#     print(step.get('name'))
#     # print(f'Building Step {step.get("name")}')
#     pipeline.add_step(step)