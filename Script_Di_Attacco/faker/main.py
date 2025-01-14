from Relay_scraper import get_relay_ips
from dummy_traffic import start
from concurrent.futures import ThreadPoolExecutor


Ips=get_relay_ips(5)
print(Ips)

if Ips:
    with ThreadPoolExecutor(max_workers=len(Ips)) as executor:
        futures = []
        for ip in Ips:
            future = executor.submit(start,ip)
            futures.append(future)
    print("Simulazione completata.")
else:
    print("Non sono stati trovati indirizzi IP.")