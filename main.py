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
from datetime import datetime
from argparse import ArgumentParser
from multiprocessing import Process
from hyperopt import fmin, hp, tpe, Trials, STATUS_OK
from hyperopt.mongoexp import MongoTrials
from pgportfolio.shortcuts.train import train
from pgportfolio.others.constants import HP_SPACE
from pgportfolio.others.tools import serialize, cal_out_of_sample_perf
from pgportfolio.shortcuts.backtest import backtest

parser = ArgumentParser(description="Automatic tune hyperparameters")
parser.add_argument("-m", "--mode", default="train", type=str)
parser.add_argument("-l", "--lag", default=2, type=int)
parser.add_argument("-n", "--num", default=1, type=int)
parser.add_argument("-s", "--start", default="20180101")
parser.add_argument("-e", "--end", default="20190609")
parser.add_argument("--train_option", default="normal")
parser.add_argument("-w", "--workers", default="4", type=int)
parser.add_argument("-r", "--rounds", default="10", type=int)

args = parser.parse_args()

def _make_and_copy(config=None):
    from shutil import copyfile
    number = 1
    pn = 'agents/agent_0001'
    if os.path.exists(pn):
        number = max([
            int(dire[len('agent_'):]) for dire \
            in os.listdir(f'agents') \
            if dire.startswith('agent')
        ]) + 1
        s = "{:04d}".format(number)
        pn = f'agents/agent_{s}'

    for dire in ['summary', 'session', 'results']:
        p = Path(pn+'/'+dire)
        if not p.is_dir():
            p.mkdir(parents=True, exist_ok=True)
            logging.warning('%s not found and created' %(pn+'/'+dire))
    if not Path(pn+'/net_config.json').is_file():
        if config is None:
            copyfile('agents/sample.json', pn+'/net_config.json')
            logging.warning('net_config.json copied from sample')
        elif config is not None:
            logging.warning("assume config provided is complete")
            with open(f"{pn}/net_config.json", 'w') as of:
                serialize(config)
                json.dump(config, of)
    return pn, number

def objective(kwargs):
    from json import loads
    with open(f"./agents/sample.json") as file:
        config = json.loads(file.read())
    d = kwargs["layers"][2]
    if d["kernel_size"]!=d["strides"]:
        return 1e04
    config.update(kwargs)
    _, num = _make_and_copy(config=config)
    os.system(f"python main.py --mode=train --num={num}")
    os.system(f"python main.py --mode=backtest --lag={args.lag} --num={num}")
    return {
        "loss": -cal_out_of_sample_perf(num), 
        "status": STATUS_OK
    }

def main():
    if args.mode=="train":
        train(args.num)
    elif args.mode=="backtest":
        backtest(args.num)
    elif args.mode=="hyperopt":
        if args.train_option=="mongo":
            for _ in range(args.round):
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
                    best = fmin(fn=objective,
                        space=HP_SPACE,
                        algo=tpe.suggest,
                        max_evals=len(trials._trials) + 8, 
                        trials = trials
                    )
                    serialize(best)
                    with open(f"./agents/best_net_config.json", 'w') as of:
                        json.dump(best, of)
        elif args.train_option=="normal":
            p = Path(f"./agents/trials")
            p.mkdir(parents=True, exist_ok=True)
            if os.path.isfile(f"./agents/trials/trials.p"):
                with open(f"./agents/trials/trials.p", "rb") as file_trials:
                    trials = pickle.load(file_trials)
                print("trials loaded")
            else:
                trials = Trials()
            best = fmin(fn=objective,
                space=HP_SPACE,
                algo=tpe.suggest,
                max_evals=len(trials._dynamic_trials)+1, 
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
