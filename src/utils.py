import subprocess
import os
from six import BytesIO
import joblib


def runcmd(cmd):
    p = subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    return p


def launch_workers(n_workers):
    processes = []
    for i in range(0, n_workers):
        cmd = 'python3 run_worker.py --thread={}'.format(i)
        print('Executing ' + cmd)
        processes.append(runcmd(cmd))
    return processes


def cluster_spec(n_workers=2, port=12333):
    cluster = {}
    host = '127.0.0.1'
    cluster['ps'] = ['{}:{}'.format(host, port)]
    port += 1
    all_workers = []
    for _ in range(n_workers):
        all_workers.append('{}:{}'.format(host, port))
        port += 1
    cluster['worker'] = all_workers
    return cluster


def dump_object(data):
    # converts whatever to string
    s = BytesIO()
    joblib.dump(data, s)
    return s.getvalue()


def load_object(string):
    # converts string to whatever was dumps'ed in it
    return joblib.load(BytesIO(string))
