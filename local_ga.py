import copy
from ga_model import *
from utils import RandomGenerator
from evaluate import evaluate
import threading
import time

class workerThread(threading.Thread):
    def __init__(self, tid, args):
        threading.Thread.__init__(self)
        self.tid = tid
        self.args = args

    def run(self):
        start = time.time()
        evaluate_worker(*self.args)

def evaluate_worker(env, model, max_eval, cuda, env_seed, index, results):
    for i in index:
        results[i] = evaluate(env, model[i], max_eval=max_eval, cuda=cuda, env_seed=env_seed)

class GA:
    def __init__(self, population, compressed_models=None, cuda=False, seed=None, env_seed=2018):
        assert seed is not None
        self.population = population
        self.cuda=cuda
        self.env_seed=env_seed
        self.rng = RandomGenerator(rng=seed)
        self.models = [CompressedModel(start_rng=self.rng.generate()) for _ in range(population)] if compressed_models is None else compressed_models
    # Note: the paper says "20k frames", but there are 4 frames per network
    # evaluation, so we cap at 5k evaluations
    def get_best_models(self, env, max_eval=5000, num_worker=16):
        '''
        results = []
        for m in self.models:
            results.append(evaluate(env, m, max_eval=max_eval, cuda=self.cuda,
                env_seed=self.env_seed))
        '''
        results = [None] * self.population
        ins_per_th = self.population // num_worker if self.population % num_worker == 0 else self.population//num_worker+1
        threads = []
        for tid in range(num_worker):
            i_start = tid * ins_per_th
            i_end = (tid+1) * ins_per_th
            i_end = i_end if i_end < self.population else self.population
            threads.append(workerThread(tid, (env, self.models, max_eval, self.cuda, self.env_seed, range(i_start,i_end), results)))
            threads[tid].start()
        for th in threads:
            th.join()
    
        used_frames = sum([r[1] for r in results])
        scores = [r[0] for r in results]
        scored_models = list(zip(self.models, scores))
        scored_models.sort(key=lambda x: x[1], reverse=True)
        return scored_models, used_frames

    def evolve_iter(self, env, sigma=0.005, truncation=5, max_eval=5000):
        scored_models, used_frames = self.get_best_models(env, max_eval=max_eval )
        scores = [s for _, s in scored_models]
        median_score = np.median(scores)
        mean_score = np.mean(scores)
        max_score = scored_models[0][1]
        scored_models_tr = scored_models[:truncation]
        
        # Elitism
        self.models = [scored_models_tr[0][0]]
        for _ in range(self.population - 1):
            self.models.append(copy.deepcopy(self.rng.choice(scored_models_tr)[0]))
            self.models[-1].evolve(sigma, self.rng.generate())
            
        return median_score, mean_score, max_score, used_frames, scored_models

