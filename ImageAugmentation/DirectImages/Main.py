from Augment import Augment

params = {
    'SRC_PATH': './SampleImgs',            # Which folder to collect images from
    'DST_PATH': './GeneratedImgs',         # Where to store the generated images
    'AUGMENTATIONS_PER_IMAGE': 5,          # How many augmentations per image
    'OUTPUT_IMAGE_SIZE': (224, 224)        # What should be the output image size
}
a = Augment(params)
a.augment()
