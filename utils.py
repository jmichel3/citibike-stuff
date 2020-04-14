import os
import pickle
import random
import csv
import sys

import numpy as np
from matplotlib import pyplot

class DataLoader():
    def __init__(
            self,
            batch_size=50,
            seq_length=300,
            scale_factor=10,
            limit=500):
        self.data_dir = "./data"
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.scale_factor = scale_factor  # divide data by this factor
        self.limit = limit  # removes large noisy gaps in the data

        data_file = os.path.join(self.data_dir, "train/201901-citibike-tripdata.csv")

        if not (os.path.exists(data_file)):
            print("couldn't find " + data_file + " in path")

        self.preprocess(self.data_dir, data_file)
        # self.load_preprocessed(data_file)
        self.reset_batch_pointer()


    columnsDict = {
        "tripduration"              : 0,
        "starttime"                 : 1,
        "stoptime"                  : 2,
        "start station id"          : 3,
        "start station name"        : 4,
        "start station latitude"    : 5,
        "start station longitude"   : 6,
        "end station id"            : 7,
        "end station name"          : 8,
        "end station latitude"      : 9,
        "end station longitude"     : 10,
        "bikeid"                    : 11,
        "usertype"                  : 12,
        "birth year"                : 13,
        "gender"                    : 14,
    }

    def columnLookup(self, cols):
        assert len(cols) >= 1, "len(cols) must be >= 1"

        columnNums = []
        for c in cols:
            if c in self.columnsDict:
                columnNums.append(self.columnsDict[c])
            else:
                sys.exit("Error: Unrecognized column name: {}".format(c))

        return columnNums

    def normalizeByColumn(self, data):
        assert len(data.shape) == 2

        colAvgs = np.mean(data, axis=0)
        colAvgs = np.tile(colAvgs, [data.shape[0],1])
        data = np.divide(data, colAvgs)

        colMaxs = np.max(np.abs(data), axis=0)
        colMaxs = np.tile(colMaxs, [data.shape[0],1])
        data = np.divide(data, colMaxs)
        
        return data


    def preprocess(self, data_dir, data_file):
        # create data file from raw xml files from iam handwriting source.

        # build the list of xml files
        filelist = []
        # Set the directory you want to start from
        rootDir = data_dir
        for dirName, subdirList, fileList in os.walk(rootDir):
            #print('Found directory: %s' % dirName)
            for fname in fileList:
                #print('\t%s' % fname)
                filelist.append(dirName + "/" + fname)

        self.csvdata = np.genfromtxt(data_file, delimiter=',', skip_header=1)

        # Numerize the user type
        # usertypeLookup = {"Customer" : 0, "Subscriber" : 1,}
        # usertypeNums = []
        # for i in self.csvdata[:, self.columnLookup(["usertype"])]:
        #     if i == "Customer":
        #         usertypeNums.append(0)
        #     else:
        #         usertypeNums.append(1)

        # self.csvdata[:,self.columnsDict["usertype"]] = np.asarray(usertypeNums)

        # Purge the following columns from data

        column_delete_list = ["starttime", "stoptime", "start station name", "start station latitude", "start station longitude", "end station name", "end station latitude", "end station longitude","usertype"]
        self.csvdata = np.delete(self.csvdata, self.columnLookup(column_delete_list), 1)

        # Purge rows with invalid data ()
        nanRows = np.where(np.any(np.isnan(self.csvdata), axis=1)==1)
        self.csvdata = np.delete(self.csvdata, nanRows, axis=0)

        # Normalize each column to +/- 1


        # Debug
        # print("self.csvdata.shape={}".format(self.csvdata.shape))
        print("self.csvdata[0]={}".format(self.csvdata[0]))
        print("self.csvdata[1]={}".format(self.csvdata[1]))
        print("self.csvdata[1]={}".format(self.csvdata[2]))
        # print("np.any(np.isnan(self.csvdata))={}".format(np.any(np.isnan(self.csvdata))))

        np.save(os.path.join(self.data_dir, "/train/preprocessed"), self.csvdata)
        print("Saved .npy file")

        self.normData = self.normalizeByColumn(self.csvdata)
        print("normData[0]={}".format(self.normData[0]))
        print("normData[1]={}".format(self.normData[1]))
        print("normData[1]={}".format(self.normData[2]))

        np.save(os.path.join(self.data_dir, "/train/u0norm"), self.csvdata)


    def load_preprocessed(self, data_file):
        self.raw_data = np.load(data_file, allow_pickle=True)

        # goes thru the list, and only keeps the text entries that have more
        # than seq_length points
        self.data = []
        self.valid_data = []
        counter = 0

        # every 1 in 20 (5%) will be used for validation data
        cur_data_counter = 0

        for data in self.raw_data:
            if len(data) > (self.seq_length + 2):
                # removes large gaps from the data
                data = np.minimum(data, self.limit)
                data = np.maximum(data, -self.limit)
                data = np.array(data, dtype=np.float32)
                data[:, 1:2] /= self.scale_factor
                cur_data_counter = cur_data_counter + 1
                if cur_data_counter % 20 == 0:
                    self.valid_data.append(data)
                else:
                    self.data.append(data)
                    # number of equiv batches this datapoint is worth
                    counter += int(len(data) / ((self.seq_length + 2)))

        print("train data: {}, valid data: {}".format(
            len(self.data), len(self.valid_data)))
        # minus 1, since we want the ydata to be a shifted version of x data
        self.num_batches = int(counter / self.batch_size)

    def validation_data(self):
        # returns validation data
        x_batch = []
        y_batch = []
        for i in range(self.batch_size):
            data = self.valid_data[i % len(self.valid_data)]
            idx = 0
            x_batch.append(np.copy(data[idx:idx + self.seq_length]))
            y_batch.append(np.copy(data[idx + 1:idx + self.seq_length + 1]))
        return x_batch, y_batch

    def next_batch(self):
        # returns a randomised, seq_length sized portion of the training data
        x_batch = []
        y_batch = []
        for i in range(self.batch_size):
            data = self.data[self.pointer]
            # number of equiv batches this datapoint is worth
            n_batch = int(len(data) / ((self.seq_length + 2)))
            idx = random.randint(0, len(data) - self.seq_length - 2)
            x_batch.append(np.copy(data[idx:idx + self.seq_length]))
            y_batch.append(np.copy(data[idx + 1:idx + self.seq_length + 1]))
            # adjust sampling probability.
            if random.random() < (1.0 / float(n_batch)):
                # if this is a long datapoint, sample this data more with
                # higher probability
                self.tick_batch_pointer()
        return x_batch, y_batch

    def tick_batch_pointer(self):
        self.pointer += 1
        if (self.pointer >= len(self.data)):
            self.pointer = 0

    def reset_batch_pointer(self):
        self.pointer = 0
