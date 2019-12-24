import os
if not os.sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import json
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial
from datetime import datetime
from argparse import ArgumentParser
from multiprocessing import Process
from hyperopt import fmin, hp, tpe, Trials, STATUS_OK
from hyperopt.mongoexp import MongoTrials
from pgportfolio.shortcuts.train import train
from pgportfolio.others.constants import HP_SPACE
from pgportfolio.shortcuts.main import objective
from pgportfolio.shortcuts.backtest import backtest
from pgportfolio.others.tools import serialize, cal_out_of_sample_perf



parser = ArgumentParser(description="Automatic tune hyperparameters")
parser.add_argument("-m", "--mode", default="train", type=str)
parser.add_argument("-l", "--lag", default=2, type=int)
parser.add_argument("-n", "--num", default=1, type=int)
parser.add_argument("-s", "--start", default="20180101")
parser.add_argument("-e", "--end", default="20190609")
parser.add_argument("--train_option", default="normal")
parser.add_argument("-r", "--rounds", default="10", type=int)

args = parser.parse_args()

def main():
    global objective
    if args.mode=="train":
        train(args.num)
    elif args.mode=="backtest":
        backtest(args.num)
    elif args.mode=="hyperopt":
        if args.train_option=="mongo":
            for _ in range(args.rounds):
                trials = MongoTrials(f'mongo://localhost:1234/my_db/jobs')
                pid = os.fork()
                if pid == 0:
#                     processes = [Process(
#                         target = os.system("hyperopt-mongo-worker --mongo=localhost:1234/my_db --poll-interval=0.1") 
#                     ) for _ in range(args.workers)]
#                     for p in processes:
#                         p.start()
                        
#                     for p in processes:
#                         p.join()
                    continue
                else:
                    main_path, types = "../..", "mongo_workers"
                    objective = partial(objective, args = args, main_path = main_path, types = types)
                    best = fmin(fn=objective,
                        space=HP_SPACE,
                        algo=tpe.suggest,
                        max_evals=len(trials._trials) + 8, 
                        trials = trials
                    )
                    serialize(best)
                    with open(f"{main_path}/{types}/best_net_config.json", 'w') as of:
                        json.dump(best, of)
        elif args.train_option=="normal":
            for _ in range(args.rounds):
                p = Path(f"./agents/trials")
                p.mkdir(parents=True, exist_ok=True)
                if os.path.isfile(f"./agents/trials/trials.p"):
                    with open(f"./agents/trials/trials.p", "rb") as file_trials:
                        trials = pickle.load(file_trials)
                    print("trials loaded")
                else:
                    trials = Trials()
                objective = partial(objective, args = args)
                best = fmin(fn=objective,
                    space=HP_SPACE,
                    algo=tpe.suggest,
                    max_evals=len(trials._dynamic_trials)+10, 
                    trials = trials
                )
                serialize(best)
                with open(f"./agents/best_net_config.json", 'w') as of:
                    json.dump(best, of)
                with open(f"./agents/trials/trials.p", "wb") as file_trials:    
                    pickle.dump(trials, file_trials)
        else:
            raise NameError(f"train_option should be set to be mongo or normal, but {args.train_option} is supplied!")

if __name__=="__main__":
    main()
