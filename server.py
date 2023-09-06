#!/bin/python3

import asyncio
import websockets
import numpy as np
import torch
import json
import torch.nn.functional as F
from engine import log_add
from collections import defaultdict
import kaldi_native_fbank as knf
from torch.nn.utils.rnn import pad_sequence
from debug import wfst
import time
class AsrDecoder():
    def __init__(self,  decoder,decoding_window,decoding_chunk_size,asr_context,left_chunk,subsampling,final_out_dim) -> None:
        self.decoder = decoder
        self.decoding_window=decoding_window
        self.decoding_chunk_size=decoding_chunk_size
        self.asr_context=asr_context
        self.left_chunk=left_chunk
        self.subsampling=subsampling
        self.final_out_dim=final_out_dim


    def inference(self, feats,offset,att_cache,cnn_cache):
        # print(feats)
        # print(feats.shape())
        asr_context=self.asr_context
        ctc_probs=[]
        maxlen=0
        num_frames = feats.size(1)
        la=self.left_chunk*self.decoding_chunk_size
        stride = self.subsampling * self.decoding_chunk_size
        decoding_window = self.decoding_window
        for cur in range(0, num_frames - asr_context + 1, stride):
                end = min(cur + decoding_window, num_frames)
                chunk_xs = feats[:, cur:end, :]
                if not chunk_xs.size(1)==decoding_window:
                    padnum=decoding_window-chunk_xs.size(1)
                    chunk_xs=F.pad(chunk_xs,[0,0,0,padnum,0,0])
                chunk_xs=np.array(chunk_xs)
                # engine = get_engine(onnx_file_path,engine_file_path)

                # print('!!!!!!')
                # print('裁剪前')
                # print(chunk_xs.shape)
                # print(offset.shape)
                # print(att_cache.shape)
                # print(cnn_cache.shape)
                if att_cache.shape[2]>=la:
                    att_cache_=att_cache[:,:,-la:,:].copy()
                else:
                    att_cache_=att_cache.copy()
                # print('裁剪后',att_cache.shape)

                trt_out_puts=self.decoder.infer(chunk_xs,offset,att_cache_,cnn_cache,cur)
                offset[0][0]=offset[0][0]+self.decoding_chunk_size
                att_cache=trt_out_puts[0]
                cnn_cache=trt_out_puts[1]
                ctc_probs.append(trt_out_puts[2])
                maxlen=maxlen+1
        return ctc_probs,maxlen,offset,att_cache,cnn_cache
    

    def _ctc_prefix_beam_search(self,ctc_probs,maxlen):
        ctc_probs=torch.tensor(ctc_probs).reshape(-1,self.final_out_dim)
        cur_hyps = [(tuple(), (0.0, -float('inf')))]
        beam_size=10
        for t in range(0, maxlen*self.decoding_chunk_size):
                logp = ctc_probs[t]  # (vocab_size,)
                        # key: prefix, value (pb, pnb), default value(-inf, -inf)
                next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
                        # 2.1 First beam prune: select topk best
                top_k_logp, top_k_index = logp.topk(beam_size)  # (beam_size,)
                for s in top_k_index:
                    s = s.item()
                    ps = logp[s].item()
                    for prefix, (pb, pnb) in cur_hyps:
                        last = prefix[-1] if len(prefix) > 0 else None
                        if s == 0:  # blank
                            n_pb, n_pnb = next_hyps[prefix]
                            n_pb = log_add([n_pb, pb + ps, pnb + ps])
                            next_hyps[prefix] = (n_pb, n_pnb)
                        elif s == last:
                                    #  Update *ss -> *s;
                            n_pb, n_pnb = next_hyps[prefix]
                            n_pnb = log_add([n_pnb, pnb + ps])
                            next_hyps[prefix] = (n_pb, n_pnb)
                                    # Update *s-s -> *ss, - is for blank
                            n_prefix = prefix + (s, )
                            n_pb, n_pnb = next_hyps[n_prefix]
                            n_pnb = log_add([n_pnb, pb + ps])
                            next_hyps[n_prefix] = (n_pb, n_pnb)
                        else:
                            n_prefix = prefix + (s, )
                            n_pb, n_pnb = next_hyps[n_prefix]
                            n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                            next_hyps[n_prefix] = (n_pb, n_pnb)

                        # 2.2 Second beam prune
                next_hyps = sorted(next_hyps.items(),
                                        key=lambda x: log_add(list(x[1])),
                                        reverse=True)
                cur_hyps = next_hyps[:beam_size]
        hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in cur_hyps]
        hyps=hyps[0]
            # hyps = [hyps]
        content = []
        for w in hyps[0]:
            if w == eos:
                break
            content.append(char_dict[w])
        return content

class Buffer():
    def __init__(self,subsampling,asr_context,decoding_chunk_size) -> None:
        self.offset=np.array([[0]],dtype=np.int32)
        self.att_cache = np.zeros([8,4,12,128],dtype=np.int32)
        self.cnn_cache = np.zeros([8,1,256,14],dtype=np.int32)
        self.data_buffer = []
        self.subsampling = subsampling 
        self.asr_context = asr_context
        self.decoding_chunk_size = decoding_chunk_size
        self.max_sil_frame = 6
        self.stride = self.subsampling * self.decoding_chunk_size
        self.decoding_window = (self.decoding_chunk_size - 1) * self.subsampling + self.asr_context 

    def receive(self,feat):
        self.data_buffer.append(feat)
    
    def judgment(self):
        feats = np.concatenate(self.data_buffer)
        print("输出数据前",feats.shape)
        if feats.shape[1]< decoding_window:
             return False
        else:
             return True

    def replay(self):
        feat_to_predict = np.concatenate(self.data_buffer)
        # print(feat_to_predict.size())
        # feat_to_predict = feats[:self.decoding_window]
        # self.data_buffer = [feats[self.stride:]]
        return feat_to_predict,self.offset,self.att_cache,self.cnn_cache
    def judgment_replay(self):
        feat_to_predict = np.concatenate([self.data_buffer[-1:]])
        return feat_to_predict,self.offset,self.att_cache,self.cnn_cache
    def is_endpoint(self,ctc_probs):
        print("开始判断是否为结束点")
        ctc_probs_s = torch.tensor(ctc_probs)
        ctc_probs_s = ctc_probs_s.reshape(1,-1,6901)
        print(ctc_probs_s.shape)
        ctc_probs_s = torch.exp(ctc_probs_s)[0,:,0]
        print("ctc形状",ctc_probs_s.size())
        is_blank = ctc_probs_s > 0.8
        if sum(is_blank[-1 * self.max_sil_frame:]) == self.max_sil_frame:
            print("已检测到结束点")
            return True
        else:
            print("未检测到结束点")
            return False
    def reset(self):
        self.data_buffer=[]
        self.offset=np.array([[0]],dtype=np.int32)
        self.att_cache = np.zeros([8,4,12,128],dtype=np.int32)
        self.cnn_cache = np.zeros([8,1,256,14],dtype=np.int32)
     
# def outline_judgment(feats,,offset,att_cache,cnn_cache):
    


async def hello(websocket,path):
    websocket.binaryType = 'arraybuffer'
    
    while True:
        # 语音是二进制的   转换一下
        data = await websocket.recv()
        start=time.clock()

        # print(data)
        # print(len(data))
        # 转换一下
        data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / np.power(2, 15)
        # print(data)
        # print(data.shape)
        # if 'data_' not in dir():
        #      data_=np.array([])
        if 'offset' not in dir():
             offset=np.array([[0]],dtype=np.int32) 
        if 'att_cache' not in dir():
             att_cache = np.zeros([8,4,12,128],dtype=np.int32) #(num_layer,num_head,att_dim,att_out_dim)
        if 'cnn_cache' not in dir():
             cnn_cache = np.zeros([8,1,256,14],dtype=np.int32) #(num_layer,1,cnn_out_dim,cnn_dim)

        opts = knf.FbankOptions()
        opts.frame_opts.dither = 0
        opts.frame_opts.frame_length_ms=25
        opts.frame_opts.frame_shift_ms=10
        opts.mel_opts.num_bins = 80
        fbank = knf.OnlineFbank(opts)
        data=data*(1 << 15)

        # data=np.concatenate((data_, data))
        fbank.accept_waveform(16000, data.tolist())
        feats=[]
        for i in range(fbank.num_frames_ready):
                feats.append(fbank.get_frame(i))
        # feats=[torch.tensor(feats)]
        # feats = pad_sequence(feats,batch_first=True,padding_value=0)
        # print(feats.size())
        buffer.receive(feats)



        # feats,offset,att_cache,cnn_cache=buffer.judgment_replay()  
        feats=[torch.tensor(feats)] 
        feats = pad_sequence(feats,batch_first=True,padding_value=0)    
        # print(feats.size()) 
        ctc_probs,maxlen,offset,att_cache,cnn_cache=asr.inference(feats,offset,att_cache,cnn_cache)
        # print("att形状",att_cache.size(2))
        # print(ctc_probs.shape)


        if buffer.is_endpoint(ctc_probs) and buffer.judgment():
            print("beging decoding")
            # data_=np.array([])
            real_feats,real_offset,real_att_cache,real_cnn_cache=buffer.replay()
            # print()             
            real_feats=[torch.tensor(real_feats)] 
            real_feats = pad_sequence(real_feats,batch_first=True,padding_value=0)   
            ctc_probs,maxlen,real_offset,real_att_cache,real_cnn_cache=asr.inference(real_feats,real_offset,real_att_cache,real_cnn_cache)
            buffer.offset = real_offset
            buffer.att_cache = real_att_cache
            buffer.cnn_cache = real_cnn_cache
            # print(ctc_probs.shape)
            # print(maxlen)
            # content=asr._ctc_prefix_beam_search(ctc_probs,maxlen)
            ctc_probs=ctc_probs.reshape(-1,final_out_dim)
            content = wfst_decoder(ctc_probs)
            buffer.reset()
            json_data = {"status": "ok", "final_result": content}
            json_string = json.dumps(json_data)
            await websocket.send(json_string)

        else:
            # data_=data.copy()            
            content=[]
            json_data = {"status": "ok", "final_result": content}
            json_string = json.dumps(json_data)
            await websocket.send(json_string)
        # end=time.clock()
        # print("转写时间{}".format(end-start))
        # print(content)
        # json_data = {"status": "ok", "result": content}
        # json_string = json.dumps(json_data)
        # await websocket.send(json_string)

if __name__ == "__main__":
    from engine import SP_engine ,read_symbol_table
    # 在此处加载模型
    symbol_table = read_symbol_table('dict.txt')
    onnx_file_path = 'whole_model_20_floded.onnx'
    engine_file_path = "whole_model_20.trt"
    fst_dir = "small"
    wfst_decoder=wfst(fst_dir)












    offset = np.array([[0]],dtype=np.int32)
    att_cache = np.zeros([8,4,12,128],dtype=np.int32)
    cnn_cache = np.zeros([8,1,256,14],dtype=np.int32)

    char_dict = {v: k for k, v in symbol_table.items()}
    eos = len(char_dict) - 1
    #所用模型的参数
    final_out_dim=len(char_dict)
    decoding_chunk_size=12
    subsampling=6
    asr_context=11
    num_layer=8
    num_head=4
    batch_size=1
    left_chunk=10
    input_fbank_dim=80
    cnn_module_kernel=15
    att_dim=decoding_chunk_size*left_chunk
    cnn_dim=cnn_module_kernel-1
    att_out_dim=128
    cnn_out_dim=256
    stride = subsampling * decoding_chunk_size
    decoding_window =  (decoding_chunk_size - 1) * subsampling + asr_context 
    buffer = Buffer(subsampling,asr_context,decoding_chunk_size)



    # decoding_chunk_size=12
    # subsampling=4
    # stride = subsampling * decoding_chunk_size
    # decoding_window = 51
    decoder=SP_engine(onnx_file_path,
                    num_layer,
                    num_head,
                    decoding_window,
                    left_chunk,
                    input_fbank_dim,
                    decoding_chunk_size,
                    att_dim,
                    cnn_dim,
                    att_out_dim,
                    cnn_out_dim,
                    final_out_dim,
                    engine_file_path)
    asr = AsrDecoder(decoder,decoding_window,decoding_chunk_size,asr_context,left_chunk,subsampling,final_out_dim)


    chunk_xs = np.zeros([1, decoding_window, 80],dtype=np.int32)
    # chunk_xs=np.array(chunk_xs)
    for i in range(10):
        # warmup_start = time.time()
        trt_out_puts=asr.decoder.infer(chunk_xs,offset,att_cache,cnn_cache,0)
        # print("warmup takes {}".format(time.time() - warmup_start))
        print(i)
    # offset = np.array([[0]],dtype=np.int32)
    # att_cache = np.zeros([12,4,12,128],dtype=np.int32)
    # cnn_cache = np.zeros([12,1,256,7],dtype=np.int32)
    # data_=np.array([]) 

    start_server = websockets.serve(hello, "0.0.0.0", 10088)

    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
