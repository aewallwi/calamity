import tqdm
import tqdm.notebook as tqdm_notebook

PBARS = {True: tqdm_notebook.tqdm, False: tqdm.tqdm}


def echo(message, verbose=True):
    if verbose:
        print(message)
