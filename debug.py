#_*_ coding:utf-8 _*_ 
from searcher import vector1d, vector2d, vector3d
import math
from Searcher import Searcher
import numpy as np
import time

def init_wfst_decoder(fst_dir, context_path = None):
    blank_skip_thresh = 0.75
    nbest = 4
    blank_id = 0
    beam = 15.0
    lattice_beam = 5.0
    max_active = 2000
    acoustic_scale = 1.8
    ac_cost_margin = 18.0
    length_penalty = -1.0
    sentence_blank = False
    context_score = 3.0
    wfst_decoder_opts = {}
    wfst_decoder_opts['fst_dir'] = fst_dir
    wfst_decoder_opts['nbest'] = nbest
    wfst_decoder_opts['beam'] = beam
    wfst_decoder_opts['max_active'] = max_active
    wfst_decoder_opts['lattice_beam'] = lattice_beam
    wfst_decoder_opts['acoustic_scale'] = acoustic_scale
    wfst_decoder_opts['blank_skip_thresh'] = blank_skip_thresh

    # 初始化wfst解码器  集成了 context graph
    wfst_decoder = Searcher(**wfst_decoder_opts)
    wfst_decoder.set_ac_cost_margin(ac_cost_margin)
    wfst_decoder.set_length_penalty(length_penalty)
    wfst_decoder.set_sentence_blank(sentence_blank)
    if (context_path != None):
        print("load context from {}".format(context_path))
        wfst_decoder.load_context(context_path, context_score)
    wfst_decoder.init_wfst_decoder()

    return wfst_decoder

def partition_arg_topK(matrix, K, axis=0):
    """
    perform topK based on np.argpartition
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: 0 or 1. dimension to be sorted.
    :return:
    """
    a_part = np.argpartition(matrix, K, axis=axis)
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K]

def get_length(fst_dir):
    tok_num = 0
    with open(fst_dir+"/units.txt", 'r') as units_reader:
        for line in units_reader.readlines():
            tok_num += 1
    return tok_num



class wfst():
    def __init__(self,fst_dir) -> None:
        self.wfst_decoder = init_wfst_decoder(fst_dir)
        self.tok_num = get_length(fst_dir)
        self.low_log_prob = -85

    def decoder(self,matrix1):
        row_template = vector1d([self.low_log_prob]*self.tok_num)
        topk = partition_arg_topK(matrix1, 20, axis=1)
        logp = vector2d()
        for frame in range(0, matrix1.shape[0] // 2):
            cur_prob = matrix1[frame]
            row = vector1d(row_template)
            for j in topk[frame]:
                row[j] = cur_prob[j]
            logp.append(row) 
        self.wfst_decoder.search(logp)
        self.wfst_decoder.update_result()
        res = self.wfst_decoder.result()
        logp = vector2d()
        for frame in range(matrix1.shape[0] // 2, matrix1.shape[0]):
            cur_prob = matrix1[frame]
            
            row = vector1d(row_template)
            for j in topk[frame]:
                row[j] = cur_prob[j]
            logp.append(row) 
        self.wfst_decoder.search(logp)
        self.wfst_decoder.update_result()
        res = self.wfst_decoder.result()


        self.wfst_decoder.finalize_search()    
        self.wfst_decoder.update_result()
        res = self.wfst_decoder.result()

        return res

        














# if __name__=="__main__":
#     fst_dir = "small"

#     wfst_decoder = init_wfst_decoder(fst_dir)
    
#     tok_num = 0
#     low_log_prob = -85
#     with open(fst_dir+"/units.txt", 'r') as units_reader:
#         for line in units_reader.readlines():
#             tok_num += 1
#     row_template = vector1d([low_log_prob] * tok_num)

#     # 随机生成一个矩阵模拟输入
#     rd = np.random.RandomState(888) 
#     matrix1 = rd.random((16, tok_num)) # 随机生成一个 [0,1) 的浮点数 ，5x5的矩阵
#     for i in range(matrix1.shape[0]):
#         matrix1[i][:] = -matrix1[i][:]

#     start = time.time()
#     # 获取topk的值  减轻计算量
#     topk = partition_arg_topK(matrix1, 20, axis=1)
    
#     logp = vector2d()
#     for frame in range(0, matrix1.shape[0] // 2):
#         cur_prob = matrix1[frame]
#         row = vector1d(row_template)
#         for j in topk[frame]:
#             row[j] = cur_prob[j]
#         logp.append(row) 
#     wfst_decoder.search(logp)
#     wfst_decoder.update_result()
#     res = wfst_decoder.result()
#     print("partial result : {}".format(res))
        
    
#     logp = vector2d()
#     for frame in range(matrix1.shape[0] // 2, matrix1.shape[0]):
#         cur_prob = matrix1[frame]
        
#         row = vector1d(row_template)
#         for j in topk[frame]:
#             row[j] = cur_prob[j]
#         logp.append(row) 
#     wfst_decoder.search(logp)
#     wfst_decoder.update_result()
#     res = wfst_decoder.result()
#     print("partial result : {}".format(res))

#     wfst_decoder.finalize_search()    
#     wfst_decoder.update_result()
#     res = wfst_decoder.result()
#     print("final result : {}".format(res))
#     end = time.time()
    
#     print("end - start : {}".format(end - start))
#     # wfst_decoder.reset()
