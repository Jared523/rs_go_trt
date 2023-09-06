#!/bin/python3
import searcher

class Searcher():
    def __init__(self, **config):
        self.s = searcher.init(config['fst_dir'])
        self.set_nbest(config['nbest'])
        self.set_beam(config['beam'])
        self.set_max_active(config['max_active'])
        self.set_lattice_beam(config['lattice_beam'])
        self.set_acoustic_scale(config['acoustic_scale'])
        self.set_blank_skip_thresh(config['blank_skip_thresh'])

    def __del__(self):
        searcher.free(self.s)

    def set_nbest(self, n):
        searcher.set_nbest(self.s, n)

    def set_beam(self, n):
        searcher.set_beam(self.s, n)

    def set_max_active(self, n):
        searcher.set_max_active(self.s, n)

    def set_lattice_beam(self, n):
        searcher.set_lattice_beam(self.s, n)
    
    def set_acoustic_scale(self, n):
        searcher.set_acoustic_scale(self.s, n)

    def set_blank_skip_thresh(self, n):
        searcher.set_blank_skip_thresh(self.s, n)

    def set_ac_cost_margin(self, n):
        searcher.set_ac_cost_margin(self.s, n)

    def set_length_penalty(self, n):
        searcher.set_length_penalty(self.s, n)
    
    def set_sentence_blank(self, n):
        searcher.set_sentence_blank(self.s, n)
    
    def load_context(self, context_path, context_score):
        searcher.load_context(self.s, context_path, context_score)

    def search(self, logp):
        searcher.search(self.s, logp)
    
    def reset(self):
        searcher.reset(self.s)
    
    def finalize_search(self):
        searcher.finalize_search(self.s)
    
    def update_result(self):
        searcher.update_result(self.s)
    
    def init_wfst_decoder(self):
        searcher.init_wfst_decoder(self.s)

    def result(self):
        return searcher.result(self.s)
    