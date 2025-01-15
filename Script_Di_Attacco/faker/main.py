from Relay_scraper import get_relay_ips
from dummy_traffic import start
from concurrent.futures import ThreadPoolExecutor
import subprocess
import os

Ips=get_relay_ips(5)
print(Ips)
subprocess.Popen(["sudo", "-u", os.getenv("SUDO_USER"), "/usr/bin/torbrowser-launcher"], start_new_session=True)
while True:
    if Ips:
        with ThreadPoolExecutor(max_workers=len(Ips)) as executor:
            futures = []
            for ip in Ips:
                future = executor.submit(start,ip)
                futures.append(future)
        print("Simulazione completata.")
    else:
        print("Non sono stati trovati indirizzi IP.")