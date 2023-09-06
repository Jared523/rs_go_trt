import triton_python_backend_utils as pb_utils
import numpy as np
import multiprocessing
# pybind11 封装的wfst searcher
from searcher import vector1d, vector2d, vector3d
# 在searcher基础上封装的解码器
from Searcher import Searcher
import json
import os
import math
import time

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        self.model_config = model_config = json.loads(args['model_config'])
        self.max_batch_size = max(model_config["max_batch_size"], 1)

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT0")
        # Convert Triton types to numpy types
        self.out0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

        # Get INPUT configuration
        batch_log_probs = pb_utils.get_input_config_by_name(
            model_config, "batch_log_probs")
        # self.beam_size = batch_log_probs['dims'][-1]
        # self.beam_size = 4

        encoder_config = pb_utils.get_input_config_by_name(
            model_config, "encoder_out")
        self.data_type = pb_utils.triton_string_to_numpy(
            encoder_config['data_type'])

        self.feature_size = encoder_config['dims'][-1]
        
        self.init_wfst_decoder(self.model_config['parameters'])
        print('Initialized WFST Decoding!')

    def init_wfst_decoder(self, parameters):
        num_processes = multiprocessing.cpu_count()
        blank_skip_thresh = 0.75
        nbest = 4
        blank_id = 0
        bidecoder = 0
        fst_dir = None
        beam = 15.0
        lattice_beam = 5.0
        max_active = 2000
        acoustic_scale = 1.8
        ac_cost_margin = 18.0
        length_penalty = -1.0
        sentence_blank = False
        context_path = None
        context_score = 3.0
        for li in parameters.items():
            key, value = li
            value = value["string_value"]
            if key == "num_processes":
                num_processes = int(value)
            elif key == "blank_id":
                blank_id = int(value)
            elif key == "blank_skip_thresh":
                blank_skip_thresh = float(value)
            elif key == "fst_dir":
                fst_dir = value
            elif key == "beam":
                beam = float(value)
            elif key == "lattice_beam":
                lattice_beam = float(value)
            elif key == "max_active":
                max_active = float(value)
            elif key == "acoustic_scale":
                acoustic_scale = float(acoustic_scale)
            elif key == "nbest":
                nbest = int(value)
            elif key == "bidecoder":
                bidecoder = int(value)
            elif key == "rescoring":
                rescoring = bool(value)
            elif key == "tok_num":
                tok_num = int(value)
            elif key == "context_path":
                context_path = value
            elif key == "context_score":
                context_score = float(value)
            elif key == "vocabulary":
                vocab_path = value
        # rescoring的decoder beam和nbest参数保持一致
        self.beam_size = nbest
        # 为了和graph_searcher保持接口一致性
        assert(fst_dir)
        wfst_decoder_opts = {}
        wfst_decoder_opts['fst_dir'] = fst_dir
        wfst_decoder_opts['nbest'] = nbest
        wfst_decoder_opts['beam'] = beam
        wfst_decoder_opts['max_active'] = max_active
        wfst_decoder_opts['lattice_beam'] = lattice_beam
        wfst_decoder_opts['acoustic_scale'] = acoustic_scale
        wfst_decoder_opts['blank_skip_thresh'] = blank_skip_thresh

        # 初始化wfst解码器  集成了 context graph
        self.wfst_decoder = Searcher(**wfst_decoder_opts)
        self.wfst_decoder.set_ac_cost_margin(ac_cost_margin)
        self.wfst_decoder.set_length_penalty(length_penalty)
        self.wfst_decoder.set_sentence_blank(sentence_blank)
        if (context_path != "None"):
            print("load context from {}".format(context_path))
            self.wfst_decoder.load_context(context_path, context_score)

        self.low_log_prob = -85
        self.num_processes = num_processes
        # 在传入decoder前就滤除部分帧 尽量减少生成matrix的时间
        self.blank_id = blank_id
        self.blank_skip_thresh_log = math.log(blank_skip_thresh)
        self.tok_num = tok_num

        # list转vector非常耗费时间  每一个vector从模块初始化
        self.row_template = vector1d([self.low_log_prob] * self.tok_num)
        
        # rescoring 需要组件  目前先不考虑
        _, vocab = self.load_vocab(vocab_path)
        self.vocabulary = vocab
        self.bidecoder = bidecoder
        self.rescoring = rescoring
        sos = eos = len(vocab) - 1
        self.sos = sos
        self.eos = eos

    def load_vocab(self, vocab_file):
        """
        load lang_char.txt
        """
        id2vocab = {}
        with open(vocab_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                char, id = line.split()
                id2vocab[int(id)] = char
        vocab = [0] * len(id2vocab)
        for id, char in id2vocab.items():
            vocab[id] = char
        return id2vocab, vocab

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them. You
        # should avoid storing any of the input Tensors in the class attributes
        # as they will be overridden in subsequent inference requests. You can
        # make a copy of the underlying NumPy array and store it if it is
        # required.

        batch_encoder_out, batch_encoder_lens = [], []
        batch_log_probs, batch_log_probs_idx = [], []
        batch_count = []

        encoder_max_len = 0
        hyps_max_len = 0
        total = 0
        batch_logps = vector3d()
        start = time.time()
        for request in requests:
            # Perform inference on the request and append it to responses list...
            in_0 = pb_utils.get_input_tensor_by_name(request, "encoder_out")
            in_1 = pb_utils.get_input_tensor_by_name(request, "encoder_out_lens")
            in_2 = pb_utils.get_input_tensor_by_name(request, "batch_log_probs")
            in_3 = pb_utils.get_input_tensor_by_name(request, "batch_log_probs_idx")

            batch_encoder_out.append(in_0.as_numpy())
            encoder_max_len = max(encoder_max_len, batch_encoder_out[-1].shape[1])

            cur_b_lens = in_1.as_numpy()
            batch_encoder_lens.append(cur_b_lens)
            cur_batch = cur_b_lens.shape[0]
            batch_count.append(cur_batch)

            cur_b_log_probs = in_2.as_numpy()
            cur_b_log_probs_idx = in_3.as_numpy()
            
            # 利用blank_skip_threshold过滤 尽量减少matrix构建时间
            for i in range(cur_batch):
                # 单个句子的log prob
                logp = vector2d()
                # 获取当前句子的长度
                cur_len = cur_b_lens[i]
                cur_probs = cur_b_log_probs[i][0:cur_len, :].tolist()  # T X Beam
                cur_idxs = cur_b_log_probs_idx[i][0:cur_len, :].tolist()  # T x Beam
                for frame in range(cur_len):
                    cur_prob = cur_probs[frame]
                    cur_idx = cur_idxs[frame]
                    # 从这里构建matrix
                    if self.blank_id == cur_idx[0]:
                        blank_log_prob = cur_prob[0]
                        if blank_log_prob > self.blank_skip_thresh_log:
                            continue
                    
                    row = vector1d(self.row_template)
                    for j in range(len(cur_prob)):
                        row[cur_idx[j]] = cur_prob[j]
                    logp.append(row) 

                    # 如果碰到空语音就append 一个空的   
                    # 或者直接在searcher里面写? 如果输入空 输入也返回空
                if len(logp) == 0:
                    row_empty = vector1d(self.row_template)
                    row_empty[self.blank_id] = -0.025
                    logp.append(row_empty)
                batch_logps.append(logp)
                total += 1

        end_1 = time.time()

        nj = min(len(batch_logps), 10)
        # 使用 decoder_batch 来处理任务
        hyps_wfst = self.wfst_decoder.search(batch_logps, nj)
        end = time.time()
        score_hyps = []
        for hyp in hyps_wfst:
            nbest_sin = []
            for hyp_sin in hyp:
                nbest_sin.append([hyp_sin[1], hyp_sin[0]])
            score_hyps.append(nbest_sin)

        if True:
            # 一次可能处理多个batch  将多个batch依次发送出去
            st = 0
            for b in batch_count:
                sents = np.array([hyps_wfst[st:st + b][0][0][2]])
                out0 = pb_utils.Tensor("OUTPUT0", sents.astype(self.out0_dtype))
                inference_response = pb_utils.InferenceResponse(output_tensors=[out0])
                responses.append(inference_response)
                st += b
            
            return responses
        
        # # 进行rescoring 
        all_hyps = []
        all_ctc_score = []
        max_seq_len = 0
        for seq_cand in score_hyps:
            # if candidates less than beam size
            if len(seq_cand) != self.beam_size:
                seq_cand = list(seq_cand)
                seq_cand += (self.beam_size - len(seq_cand)) * [(-float("INF"), (0,))]

            for score, hyps in seq_cand:
                all_hyps.append(list(hyps))
                all_ctc_score.append(score)
                max_seq_len = max(len(hyps), max_seq_len)

        beam_size = self.beam_size
        feature_size = self.feature_size
        hyps_max_len = max_seq_len + 2
        in_ctc_score = np.zeros((total, beam_size), dtype=self.data_type)
        in_hyps_pad_sos_eos = np.ones(
            (total, beam_size, hyps_max_len), dtype=np.int64) * self.eos
        if self.bidecoder:
            in_r_hyps_pad_sos_eos = np.ones(
                (total, beam_size, hyps_max_len), dtype=np.int64) * self.eos

        in_hyps_lens_sos = np.ones((total, beam_size), dtype=np.int32)

        in_encoder_out = np.zeros((total, encoder_max_len, feature_size),
                                  dtype=self.data_type)
        in_encoder_out_lens = np.zeros(total, dtype=np.int32)
        st = 0
        for b in batch_count:
            t = batch_encoder_out.pop(0)
            in_encoder_out[st:st + b, 0:t.shape[1]] = t
            in_encoder_out_lens[st:st + b] = batch_encoder_lens.pop(0)
            for i in range(b):
                for j in range(beam_size):
                    cur_hyp = all_hyps.pop(0)
                    cur_len = len(cur_hyp) + 2
                    in_hyp = [self.sos] + cur_hyp + [self.eos]
                    in_hyps_pad_sos_eos[st + i][j][0:cur_len] = in_hyp
                    in_hyps_lens_sos[st + i][j] = cur_len - 1
                    if self.bidecoder:
                        r_in_hyp = [self.sos] + cur_hyp[::-1] + [self.eos]
                        in_r_hyps_pad_sos_eos[st + i][j][0:cur_len] = r_in_hyp
                    in_ctc_score[st + i][j] = all_ctc_score.pop(0)
            st += b
        in_encoder_out_lens = np.expand_dims(in_encoder_out_lens, axis=1)
        in_tensor_0 = pb_utils.Tensor("encoder_out", in_encoder_out)
        in_tensor_1 = pb_utils.Tensor("encoder_out_lens", in_encoder_out_lens)
        in_tensor_2 = pb_utils.Tensor("hyps_pad_sos_eos", in_hyps_pad_sos_eos)
        in_tensor_3 = pb_utils.Tensor("hyps_lens_sos", in_hyps_lens_sos)
        input_tensors = [in_tensor_0, in_tensor_1, in_tensor_2, in_tensor_3]
        if self.bidecoder:
            in_tensor_4 = pb_utils.Tensor("r_hyps_pad_sos_eos", in_r_hyps_pad_sos_eos)
            input_tensors.append(in_tensor_4)
        in_tensor_5 = pb_utils.Tensor("ctc_score", in_ctc_score)
        input_tensors.append(in_tensor_5)

        inference_request = pb_utils.InferenceRequest(
            model_name='decoder',
            requested_output_names=['best_index'],
            inputs=input_tensors)

        inference_response = inference_request.exec()

        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
        else:
            # Extract the output tensors from the inference response.
            best_index = pb_utils.get_output_tensor_by_name(inference_response,
                                                            'best_index')
            best_index = best_index.as_numpy()
            hyps = []
            idx = 0
            for cands, cand_lens in zip(in_hyps_pad_sos_eos, in_hyps_lens_sos):
                best_idx = best_index[idx][0]
                # best_cand_len = cand_lens[best_idx] - 1  # remove sos
                # best_cand = cands[best_idx][1: 1 + best_cand_len].tolist()
                best_cand = hyps_wfst[idx][best_idx][2]
                hyps.append(best_cand)
                idx += 1

            st = 0
            for b in batch_count:
                sents = np.array(hyps[st:st + b])
                out0 = pb_utils.Tensor("OUTPUT0", sents.astype(self.out0_dtype))
                inference_response = pb_utils.InferenceResponse(output_tensors=[out0])
                responses.append(inference_response)
                st += b
        
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
