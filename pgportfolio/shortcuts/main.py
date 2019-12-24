import os
import json
import logging
from pathlib import Path
from pgportfolio.others.tools import serialize, cal_out_of_sample_perf
from hyperopt import STATUS_OK

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

def objective(kwargs, args, main_path = ".", types = "agents"):
    from json import loads
    import os
    with open(f"{main_path}/{types}/sample.json") as file:
        config = json.loads(file.read())
    d = kwargs["layers"][2]
    if d["kernel_size"]!=d["strides"]:
        return 1e04
    config.update(kwargs)
    _, num = _make_and_copy(config=config)
    os.system(f"python {main_path}/main.py --mode=train --num={num}")
    os.system(f"python {main_path}/main.py --mode=backtest --lag={args.lag} --num={num}")
    return {
        "loss": -cal_out_of_sample_perf(num), 
        "status": STATUS_OK
    }
