import numpy as np
from glob import glob
from torch.utils.data import Dataset, DataLoader
#from joblib import Parallel, delayed
import collections
from collections import deque
#from itertools import islice

# sphere mesh size at different levels
nv_sphere = [12, 42, 162, 642, 2562, 10242, 40962, 163842]
classes = range(0, 7)
#classes = [0,1,2,3,4,5,6,7,8,9,12,13,14,15,16,17,18,19,20]
feat_type = '*'

class S2D3DSegLoader(Dataset):
    """Data loader for 2D3DS dataset."""

    def __init__(self, data_dir, partition, fold, sp_level, in_ch=3):
        """
        Args:
            data_dir: path to data directory
            partition: train or test
            fold: 1 to 5 (for 5-fold cross-validation)
            sp_level: sphere mesh level. integer between 0 and 7.
            
        """
        #assert(partition in ["train", "test"])
        #assert(fold in [1, 2, 3])
        self.in_ch = in_ch
        self.nv = nv_sphere[sp_level]
        self.partition = partition

        feature_type = [feat.split('/')[-1] for feat in glob(data_dir + feat_type)]
        file_format1 = data_dir + feat_type + '/*/*.*.*.dat'
        flist = []
        flist += sorted(glob(file_format1))

        # dict construction
        data = dict()
        for i in flist:
            key = '.'.join(i.split('.')[0:2]).split('/')[-1]
            cat = i.split('.')[0].split('/')[-3]
            if not key in data:
                data[key] = {'subject': key}
            data[key].setdefault(i.split('.')[2]+cat, []).append(i)

        # subject list
        subj = [entry for entry in data]
        subj = sorted(subj)
        subj = deque(subj)
        subj.rotate(12 * (fold - 1))
        subj = list(subj)
        test = subj[36:60]
        train = [item for item in subj if item not in set(test)]
        val = test[0:12]
        test = test[12:24]
        self.flist = []

        # final list
        if partition == "train":
            flist_train = []
            for i in train:
                for feat in feature_type:
                    for deg in range(0, 16):
                        flist_train.append(data[i]['deg' + str(deg) + feat])
            self.flist = flist_train

        if partition == "val":
            flist_test = []
            for i in val:
                for feat in feature_type:
                    flist_test.append(data[i]['deg0curv'])
            self.flist = flist_test

        if partition == "test":
            flist_test = []
            for i in test:
                for feat in feature_type:
                    flist_test.append(data[i]['deg0curv'])
            self.flist = flist_test

        # label dictionary
        lut = collections.defaultdict(lambda : 0) 
        for i, label in enumerate(classes):
            lut[label] = i
        self.lut = lut

    def __len__(self):
        return len(self.flist)

    def __getitem__(self, idx):
        # load files
        subj = self.flist[idx]
        # multi-thread loader
        #data, labels = self.readin(subj)
        #data = data.astype(np.float32)
        #labels = labels.astype(np.int)

        # single-thread loader
        #data = np.array([])
        #for feat in subj:
        #    #data = np.append(data, [np.loadtxt(feat).T[:self.nv]])
        #    #T = []
        #    #fp = open(feat)
        #    #for i, line in enumerate(fp):
        #    #    if i < self.nv:
        #    #        T.append(float(line.split()[0]))
        #    #    else:
        #    #        break
        #    #fp.close()
        #    with open(feat) as f:
        #        T = islice(f, self.nv)
        #        T = list(map(float, T))
        #        #T = [float(i) for i in T]
        #    data = np.append(data, T)
        data = np.array([])
        for feat in subj[:-1]:
            T = np.fromfile(feat,count=self.nv,dtype=np.double)
            data = np.append(data, T)
        T = np.fromfile(subj[-1],count=self.nv,dtype=np.int16)
        data = np.append(data, T)

        data = np.reshape(data, (-1, self.nv))
        labels = data[self.in_ch, :self.nv]
        data = data[:self.in_ch, :self.nv].astype(np.float32)

        labels = [self.lut[label] for label in labels]
        labels = np.asarray(labels).astype(np.int)
        return data, labels

    def readin(self, feat_filelist):
        H = Parallel(n_jobs=4)(delayed(self.process)(i, name) for i, name in enumerate(feat_filelist))
        H.sort()
        H = [r[1] for r in H]
        H = np.array(H)
        #H = np.reshape(H, (-1, self.nv))
        #H = H[:self.in_ch, :self.nv]
        return H[:self.in_ch, :self.nv], H[self.in_ch, :self.nv]

    def process(self, i, file):
        #return [i, np.loadtxt(fname=file, max_rows=self.nv)]
        data = []
        fp = open(file)
        for i, line in enumerate(fp):
            if i < self.nv:
                data.append(float(line.split()[0]))
            else:
                break
        fp.close()
        #with open(file) as f:
        #    data = f.read().splitlines()
        #data = data[0:self.nv]
        #data = [float(i) for i in data]
        return [i, data]
