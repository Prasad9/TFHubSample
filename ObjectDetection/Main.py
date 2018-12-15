from Infer import Infer

params = {
    'BATCH_SIZE': 1,                              # Batch size has to be 1
    'PLOT_TOP_K': 3,                              # How many top predictions to plot
    'INFER_PATH': './Data/Imgs',
}
i = Infer(params)
i.infer()
