import subprocess
import requests
import time
'''
Spoofa quando il servizio tor è attivo e fa due richieste per scoprire l'ip reale e l'ip di tor della vittima
TODO rifare l'operazione di rivelazione degli Ip ad ogni ricreazione del circuito Tor
TODO trovare sito migliore di ifconfig
'''
def leak():
    url = "https://ifconfig.me"
    try:
        response = requests.get(url, timeout=10)
        print(f"Real IP: {response.text.strip()}")
    except requests.exceptions.RequestException as e:
        print(f"Errore durante la richiesta: {e}")
    
    proxies = {
        "http": "socks5h://127.0.0.1:9050",
        "https": "socks5h://127.0.0.1:9050",
    }
    try:
        response_tor = requests.get(url, proxies=proxies, timeout=10)
        print("Tor IP: ", response_tor.text.strip())
    except requests.exceptions.RequestException as e:
        print(f"Errore durante la richiesta tramite Tor: {e}")

def check_service_status(service_name):
    try:
        result = subprocess.run(
            ["systemctl", "is-active", service_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Errore: {e}")
        return False

def wait_for_service(service_name, check_interval=5):
    print(f"Attesa che il servizio '{service_name}' diventi attivo...")
    while not check_service_status(service_name):
        print(f"Il servizio '{service_name}' non è attivo. Riprovo tra {check_interval} secondi...")
        time.sleep(check_interval)
    print(f"Il servizio '{service_name}' è attivo!")
    return True

if wait_for_service("tor"):
    leak()
