"""
Process physionet. Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/generate_timeseries.py
"""

import errno
import os
import pickle
import random
import tarfile

import jax
import jax.numpy as jnp
import numpy as onp


def makedir_exist_ok(dirpath):
    """
    Python2 support for os.makedirs(.., exist_ok=True)
    """
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise


def download_url(url, root, filename=None):
    """Download a file from a url and place it in root.

    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
    """
    from six.moves import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    # downloads file
    if os.path.isfile(fpath):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(
                url, fpath
            )
        except OSError:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(
                    url, fpath
                )


class PhysioNet:
    """
    PhysioNet Dataset.
    """

    urls = [
        'https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download',
        'https://physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz?download',
    ]

    outcome_urls = ['https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt']

    params = [
        'Age', 'Gender', 'Height', 'ICUType', 'Weight', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN',
        'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg',
        'MAP', 'MechVent', 'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate',
        'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC'
    ]

    params_dict = {k: i for i, k in enumerate(params)}

    def __init__(self,
                 root,
                 download=False,
                 quantization=0.1,
                 n_samples=None):

        self.root = root
        self.reduce = "average"
        self.quantization = quantization

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        self.data = []
        for data_file in [self.training_file, self.test_file]:
            infile = open(os.path.join(self.processed_folder, data_file), 'rb')
            self.data += pickle.load(infile)
            infile.close()

        if n_samples is not None:
            self.data = self.data[:n_samples]

    def download(self):
        """
        Download physionet data to disk.
        """
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        for url in self.urls:
            filename = url.rpartition('/')[2]
            download_url(url, self.raw_folder, filename)
            tar = tarfile.open(os.path.join(self.raw_folder, filename), "r:gz")
            tar.extractall(self.raw_folder)
            tar.close()

            print('Processing {}...'.format(filename))

            dirname = os.path.join(self.raw_folder, filename.split('.')[0])
            patients = []
            total = 0
            for file_num, txtfile in enumerate(os.listdir(dirname)):
                print(file_num, txtfile)
                outfile = open("%s/iter.txt" % self.root, "a")
                outfile.write("%d, %s\n".format(file_num, txtfile))
                outfile.close()
                record_id = txtfile.split('.')[0]
                with open(os.path.join(dirname, txtfile)) as f:
                    lines = f.readlines()
                    prev_time = 0
                    tt = [0.]
                    vals = [onp.zeros(len(self.params))]
                    mask = [onp.zeros(len(self.params))]
                    nobs = [onp.zeros(len(self.params))]
                    for line_num, l in enumerate(lines[1:]):
                        # print(line_num, len(lines[1:]))
                        total += 1
                        time, param, val = l.split(',')
                        # Time in hours
                        time = float(time.split(':')[0]) + float(time.split(':')[1]) / 60.
                        # round up the time stamps (up to 6 min by default)
                        # used for speed -- we actually don't need to quantize it in Latent ODE
                        if self.quantization != 0:
                            time = round(time / self.quantization) * self.quantization

                        if time != prev_time:
                            tt.append(time)
                            vals.append(onp.zeros(len(self.params)))
                            mask.append(onp.zeros(len(self.params)))
                            nobs.append(onp.zeros(len(self.params)))
                            prev_time = time

                        if param in self.params_dict:
                            n_observations = nobs[-1][self.params_dict[param]]
                            if self.reduce == 'average' and n_observations > 0:
                                prev_val = vals[-1][self.params_dict[param]]
                                new_val = (prev_val * n_observations + float(val)) / (n_observations + 1)
                                # vals[-1] = jax.ops.index_update(vals[-1],
                                #                                 jax.ops.index[self.params_dict[param]], new_val)
                                vals[-1][self.params_dict[param]] = new_val
                            else:
                                # vals[-1] = jax.ops.index_update(vals[-1],
                                #                                 jax.ops.index[self.params_dict[param]], float(val))
                                vals[-1][self.params_dict[param]] = float(val)
                            # mask[-1] = jax.ops.index_update(mask[-1], jax.ops.index[self.params_dict[param]], 1)
                            mask[-1][self.params_dict[param]] = 1
                            # nobs[-1] = jax.ops.index_add(nobs[-1], jax.ops.index[self.params_dict[param]], 1)
                            nobs[-1][self.params_dict[param]] += 1
                        else:
                            assert param == 'RecordID', 'Read unexpected param {}'.format(param)
                tt = onp.array(tt)
                vals = onp.stack(vals)
                mask = onp.stack(mask)

                patients.append((record_id, tt, vals, mask))

            outfile = open(os.path.join(self.processed_folder,
                                        filename.split('.')[0] + "_" + str(self.quantization) + '.pt'), 'wb')
            pickle.dump(patients, outfile)
            outfile.close()

        print('Done!')

    def _check_exists(self):
        for url in self.urls:
            filename = url.rpartition('/')[2]

            if not os.path.exists(
                os.path.join(self.processed_folder,
                    filename.split('.')[0] + "_" + str(self.quantization) + '.pt')
            ):
                return False
        return True

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def training_file(self):
        return 'set-a_{}.pt'.format(self.quantization)

    @property
    def test_file(self):
        return 'set-b_{}.pt'.format(self.quantization)

    @property
    def label_file(self):
        return 'Outcomes-a.pt'

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        fmt_str += '    Quantization: {}\n'.format(self.quantization)
        fmt_str += '    Reduce: {}\n'.format(self.reduce)
        return fmt_str


def init_physionet_data(rng, parse_args):
    """
    Initialize physionet data for training and testing.
    """
    if not os.path.exists("PhysioNet/processed/final.pt"):
        n_samples = None
        dataset_obj = PhysioNet(root=parse_args.data_root,
                                download=True,
                                quantization=1.,
                                n_samples=n_samples)
        # remove time-invariant features and Patient ID
        remove_params = ['Age', 'Gender', 'Height', 'ICUType']
        params_inds = [dataset_obj.params_dict[param_name]
                       for ind, param_name in enumerate(dataset_obj.params) if param_name not in remove_params]
        for ind, ex in enumerate(dataset_obj.data):
            record_id, tt, vals, mask = ex
            dataset_obj.data[ind] = (tt, vals[:, params_inds], mask[:, params_inds])
        n_samples = len(dataset_obj)

        def _split_train_test(data, train_frac=0.8):
            data_train = data[:int(n_samples * train_frac)]
            data_test = data[int(n_samples * train_frac):]
            return data_train, data_test

        dataset = onp.array(dataset_obj[:n_samples])

        random.Random(parse_args.seed).shuffle(dataset)
        train_dataset, test_dataset = _split_train_test(dataset)

        data_min, data_max = get_data_min_max(dataset_obj)

        processed_dataset = process_batch(train_dataset, data_min=data_min, data_max=data_max)

        with open(os.path.join(parse_args.data_root, "PhysioNet/processed/final.pt"), 'wb') as processed_file:
            pickle.dump(processed_dataset, processed_file, protocol=4)

    with open(os.path.join(parse_args.data_root, "PhysioNet/processed/final.pt"), 'rb') as processed_file:
        processed_dataset = pickle.load(processed_file)

    for key in ["observed_tp", "tp_to_predict"]:
        processed_dataset[key] = jnp.array(processed_dataset[key], dtype=jnp.float64)

    def get_batch_from_processed(inds):
        """
        Get batch from processed data (i.e. union timepoints beforehand).
        """
        keys_to_ind = ["observed_data", "data_to_predict", "observed_mask", "mask_predicted_data"]
        other_keys = ["observed_tp", "tp_to_predict"]
        batch_dict = {}
        for key in other_keys:
            batch_dict[key] = processed_dataset[key]
        for key in keys_to_ind:
            batch_dict[key] = jnp.array(processed_dataset[key][inds], dtype=jnp.float64)
        return batch_dict

    num_train = len(processed_dataset["observed_mask"])
    assert num_train % parse_args.batch_size == 0
    num_train_batches = num_train // parse_args.batch_size

    assert num_train % parse_args.test_batch_size == 0
    num_test_batches = num_train // parse_args.test_batch_size

    # make sure we always save the model on the last iteration
    assert num_train_batches * parse_args.nepochs % parse_args.save_freq == 0

    def gen_data(batch_size, shuffle=True):
        """
        Generator for train data.
        """
        key = rng
        num_batches = num_train // batch_size
        inds = jnp.arange(num_train)

        while True:
            if shuffle:
                key, = jax.random.split(key, num=1)
                epoch_inds = jax.random.shuffle(key, inds)
            else:
                epoch_inds = inds
            for i in range(num_batches):
                batch_inds = onp.array(epoch_inds[i * batch_size: (i + 1) * batch_size])
                yield get_batch_from_processed(batch_inds)
                # batch_dataset = train_dataset[batch_inds]
                # yield process_batch(batch_dataset, data_min=data_min, data_max=data_max)

    ds_train = gen_data(parse_args.batch_size)
    ds_test = gen_data(parse_args.test_batch_size, shuffle=False)

    meta = {
        "num_batches": num_train_batches,
        "num_test_batches": num_test_batches
    }

    return ds_train, ds_test, meta


def normalize_masked_data(data, mask, att_min, att_max):
    """
    Normalize masked data.
    """
    # we don't want to divide by zero
    att_max[att_max == 0] = 1

    data_norm = (data - att_min) / att_max

    # set masked out elements back to zero
    data_norm[mask == 0] = 0

    return data_norm


def split_data_interp(data_dict):
    """
    Split data into observed and to predict for interpolation task.
    """
    data_ = data_dict["data"]
    time_ = data_dict["time_steps"]
    split_dict = {"observed_data": data_,
                  "observed_tp": time_,
                  "data_to_predict": data_,
                  "tp_to_predict": time_,
                  "observed_mask": None,
                  "mask_predicted_data": None
                  }

    if "mask" in data_dict and data_dict["mask"] is not None:
        mask_ = data_dict["mask"]
        split_dict["observed_mask"] = mask_
        split_dict["mask_predicted_data"] = mask_

    return split_dict


def get_data_min_max(records):
    """
    Get min and max for each feature across the dataset.
    """

    cache_path = os.path.join(records.processed_folder, "minmax_" + str(records.quantization) + '.pt')

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as cache_file:
            data = pickle.load(cache_file)
        data_min, data_max = data
        return data_min, data_max

    data_min, data_max = None, None

    for b, (tt, vals, mask) in enumerate(records):
        if b % 100 == 0:
            print(b, len(records))
        n_features = vals.shape[-1]

        batch_min = []
        batch_max = []
        for i in range(n_features):
            non_missing_vals = vals[:, i][mask[:, i] == 1]
            if len(non_missing_vals) == 0:
                batch_min.append(jnp.inf)
                batch_max.append(-jnp.inf)
            else:
                batch_min.append(jnp.min(non_missing_vals))
                batch_max.append(jnp.max(non_missing_vals))

        batch_min = jnp.stack(batch_min)
        batch_max = jnp.stack(batch_max)

        if (data_min is None) and (data_max is None):
            data_min = batch_min
            data_max = batch_max
        else:
            data_min = jnp.minimum(data_min, batch_min)
            data_max = jnp.maximum(data_max, batch_max)

    with open(cache_path, "wb") as cache_file:
        pickle.dump((data_min, data_max), cache_file)

    return data_min, data_max


def process_batch(batch,
                  data_min=None,
                  data_max=None):
    """
    Expects a batch of time series data in the form of (tt, vals, mask) where
        - tt is a 1-dimensional tensor containing T time values of observations.
        - vals is a (T, D) tensor containing observed values for D variables.
        - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
    Returns:
        combined_tt: The union of all time observations.
        combined_vals: (M, T, D) tensor containing the observed values.
        combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
    """
    D = batch[0][1].shape[1]

    # get union of timepoints
    combined_tt, inverse_indices = onp.unique(onp.concatenate([ex[0] for ex in batch]),
                                              return_inverse=True)

    offset = 0
    combined_vals = onp.zeros([len(batch), len(combined_tt), D])
    combined_mask = onp.zeros([len(batch), len(combined_tt), D])

    for b, (tt, vals, mask) in enumerate(batch):

        indices = inverse_indices[offset:offset + len(tt)]
        offset += len(tt)

        combined_vals[b, indices] = vals
        combined_mask[b, indices] = mask

    combined_vals = normalize_masked_data(combined_vals, combined_mask, att_min=data_min, att_max=data_max)

    # normalize times to be in [0, 1]
    if onp.amax(combined_tt) != 0.:
        combined_tt /= onp.amax(combined_tt)

    data_dict = {
        "data": combined_vals,
        "time_steps": combined_tt,
        "mask": combined_mask
    }

    data_dict = split_data_interp(data_dict)
    return data_dict
