from setproctitle import setproctitle
import random
import string
import subprocess
import time

'''
Permette di avviare il servizio Tor con un nome random
'''
def randomize_process_name():
    name = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    setproctitle(name)
def start_tor():
    while True:
        randomize_process_name()
        subprocess.run(["tor"])
        time.sleep(10)
start_tor()
