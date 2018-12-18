from Infer import Infer

params = {
    'TOTAL_IMAGES': 100,            # Number of images to generate
    'BATCH_SIZE': 64,
    'SAVE_FOLDER': './Images'        # Path where images are to be saved.
}
i = Infer(params)
i.infer()
