import transformers
import numpy as np
import csv
import os
import logging
import time
import utils
import config

class Processor:
    def __init__(self, config):
        self.config = config
        self.data_dir = config.data_dir

    def process(self):
        for file in os.listdir(self.data_dir):
            self.preprocess(file)
    
    def preprocess(self, file):
        input_dir = self.data_dir + file
        output_dir = self.data_dir + file[:-4] + '.npz'
        if os.path.exists(output_dir) is True:
            return
        word_list = []
        label_list = []
        words = []
        labels = []
        if 'train' in file:
            with open(input_dir, encoding='utf-8-sig') as f:
                row = csv.reader(f, delimiter=',')
                next(row)
                i = 0
                for line in row:
                    if line[0] == '':
                        if i==0:
                            words_ = words
                            labels_ = labels
                            i += 1
                            continue
                        if len(words) > 511:
                            word_list.append(words_)
                            label_list.append(labels_)
                            words = words[len(words_):]
                            labels = labels[len(labels_):]
                            continue
                        word_list.append(words)
                        label_list.append(labels)
                        words = []
                        labels = []
                        i = 0
                    else:
                        words.append(line[0])
                        labels.append(line[1])
            np.savez_compressed(output_dir, words=word_list, labels=label_list)
            logging.info("--------{} data process DONE!--------".format('train'))
        '''
        else:
            with open(input_dir, encoding='utf-8-sig') as f:
                row = csv.reader(f, delimiter=',')
                next(row)
                for line in row:
                    if line[0] == '':
                        word_list.append(words)
                        label_list.append(labels)
                        words = []
                        labels = []
                    else:
                        words.append(line[1])
                        labels.append('O')
            np.savez_compressed(output_dir, words=word_list, labels=label_list)
            logging.info("--------{} data process DONE!--------".format('test'))
        '''
            
if __name__=="__main__":
    utils.set_logger(os.getcwd() + time.strftime('%Y%m%d%H%M', time.localtime(time.time())) + '.log')
    processor = Processor(config)
    processor.process()
    logging.info("--------Process Done!--------")
