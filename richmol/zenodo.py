import progressbar
import urllib.request
import h5py


class ProgressBar():
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar=progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


def downloadFile(dataset_id, file_name, output_file_name, verbose=True):
    """Downloads file `filename` from a Zenodo dataset, given by an ID number
    `dataset_id`, into file `output_file_name`
    """
    url = "https://zenodo.org/record/" + str(dataset_id) + "/files/" + file_name
    if verbose:
        print(f"download {url} into {output_file_name}")
    urllib.request.urlretrieve(url, output_file_name, ProgressBar())
    if verbose:
        print("download complete")
    if verbose:
        print(f"print available datasets in file {output_file_name}")
        with h5py.File(output_file_name, "r") as fl:
            for key, val in fl.items():
                print(f"'{key}'")                   # dataset name
                print("\t", val.attrs["__doc__"])   # dataset description

