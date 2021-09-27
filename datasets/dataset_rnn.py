from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os, torch, h5py
from torch import nn
from scipy import stats
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.model_rnn import rnn_classify

class Generic_WSI_RNN_Dataset(Dataset):
    def __init__(self, csv_path, data_dir, n=100, print_info=True, label_dicts=[{}, {}, {}],
                 label_cols=['label', 'site', 'sex'], patient_voting='max', ):
        self.custom_test_ids = None
        self.print_info = print_info
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.data_dir = None
        self.label_cols = label_cols
        self.split_gen = None
        self.data_dir = data_dir
        self.n = n
        slide_data = pd.read_csv(csv_path)

        self.label_dicts = label_dicts
        self.num_classes = [len(set(label_dict.values())) for label_dict in self.label_dicts]
        slide_data = self.df_prep(slide_data, self.label_dicts, self.label_cols)
        self.slide_data = slide_data

        self.patient_data_prep(patient_voting)
        self.cls_ids_prep()

        if print_info:
            self.summarize()


    @staticmethod
    def rand_row(array, dim_needed):  # array为需要采样的矩阵，dim_needed为想要抽取的行数
        out = np.zeros((dim_needed, 1024))
        row_total = array.shape[0]
        row_sequence = np.random.permutation(row_total)
        out[:row_total] = array[row_sequence[0:dim_needed], :]
        return out

    def __len__(self):
        return self.slide_data.shape[0]

    def __getitem__(self, idx):
        full_path = os.path.join(self.data_dir, os.path.splitext(self.patient_data['case_id'][idx])[0] + '.h5')
        with h5py.File(full_path, 'r') as hdf5_file:
            feature = hdf5_file['features'][:]
            # feature_len = self.n if feature.shape[0] >= self.n else feature.shape[0]
            feature = self.rand_row(feature, self.n)

        return feature, self.patient_data['label'][idx]#, feature_len

    def cls_ids_prep(self):
        # store ids corresponding each class at the patient or case level
        self.patient_cls_ids = [[] for i in range(self.num_classes[0])]
        for i in range(self.num_classes[0]):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

        # store ids corresponding each class at the slide level
        self.slide_cls_ids = [[] for i in range(self.num_classes[0])]
        for i in range(self.num_classes[0]):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def patient_data_prep(self, patient_voting='max'):
        patients = np.unique(np.array(self.slide_data['case_id']))  # get unique patients
        patient_labels = []

        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data['label'][locations].values
            if patient_voting == 'max':
                label = label.max()  # get patient label (MIL convention)
            elif patient_voting == 'maj':
                label = stats.mode(label)[0]
            else:
                raise NotImplementedError
            patient_labels.append(label)

        self.patient_data = {'case_id': patients, 'label': np.array(patient_labels)}

    def summarize(self):
        for task in range(len(self.label_dicts)):
            print('task: ', task)
            print("label column: {}".format(self.label_cols[task]))
            print("label dictionary: {}".format(self.label_dicts[task]))
            print("number of classes: {}".format(self.num_classes[task]))
            print("slide-level counts: ", '\n', self.slide_data[self.label_cols[task]].value_counts(sort=False))

        for i in range(self.num_classes[0]):
            print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
            print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

    @staticmethod
    def df_prep(data, label_dicts, label_cols):
        if label_cols[0] != 'label':
            data['label'] = data[label_cols[0]].copy()

        data.reset_index(drop=True, inplace=True)
        for i in data.index:
            key = data.loc[i, 'label']
            data.at[i, 'label'] = label_dicts[0][key]

        for idx, (label_dict, label_col) in enumerate(zip(label_dicts[1:], label_cols[1:])):
            print(label_dict, label_col)
            data[label_col] = data[label_col].map(label_dict)

        return data

    def return_splits(self, from_id=True, csv_path=None):
        if from_id:
            if len(self.train_ids) > 0:
                train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
                train_split = Generic_Split(train_data, data_dir=self.data_dir, num_classes=self.num_classes,
                                            label_cols=self.label_cols)

            else:
                train_split = None

            if len(self.val_ids) > 0:
                val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
                val_split = Generic_Split(val_data, data_dir=self.data_dir, num_classes=self.num_classes,
                                          label_cols=self.label_cols)

            else:
                val_split = None

            if len(self.test_ids) > 0:
                test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
                test_split = Generic_Split(test_data, data_dir=self.data_dir, num_classes=self.num_classes,
                                           label_cols=self.label_cols)

            else:
                test_split = None


        else:
            assert csv_path
            all_splits = pd.read_csv(csv_path)
            train_split = self.get_split_from_df(all_splits, 'train')
            val_split = self.get_split_from_df(all_splits, 'val')
            test_split = self.get_split_from_df(all_splits, 'test')

        return train_split, val_split, test_split

    def get_split_from_df(self, all_splits=None, split_key='train', return_ids_only=False, split=None):
        if split is None:
            split = all_splits[split_key]
            split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())
            if return_ids_only:
                ids = np.where(mask)[0]
                return ids

            df_slice = self.slide_data[mask].dropna().reset_index(drop=True)
            split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes,
                                  label_cols=self.label_cols)

        else:
            split = None

        return split


class Generic_Split(Generic_WSI_RNN_Dataset):
    def __init__(self, slide_data, data_dir=None, n=100, num_classes=2, label_cols=None):
        self.use_h5 = True
        self.slide_data = slide_data
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.n = n
        self.slide_cls_ids = [[] for i in range(self.num_classes[0])]
        self.label_cols = label_cols
        self.infer = False
        for i in range(self.num_classes[0]):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]
        self.patient_data_prep('max')
    def __len__(self):
        return len(self.slide_data)


if __name__ == '__main__':

    dataset = Generic_WSI_RNN_Dataset('D:/dataset.csv',
                                      'C:/RESULTS_TUMOR_STAIN_NORM/patches', 100,
                                      print_info=True, label_dicts=[{'MSS': 0, 'MSI-H': 1}], label_cols=['label'], )
    loader = DataLoader(dataset, 64, shuffle=False)

    for feature, label, feature_len in loader:
        order_idx = np.argsort(feature_len.numpy())[::-1]
        order_label = label[order_idx.tolist()]
        net = rnn_classify()
        net.train()
        out = net(feature, feature_len)

        print(out)
