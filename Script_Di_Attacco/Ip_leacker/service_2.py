import socket
import subprocess
import requests
import time
'''
Rivela quando c'è aperto un socket in 9050 e fa delle richieste per svelare l'indirizzo ip reale dell'utente
TODO Riavviare la deanonimizzazione ogni timeout e ricreazione del circuito tor
TODO Utilizzare sito migliore di ifconfig
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
def is_tor_proxy_active(host='127.0.0.1', port=9050):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)  # Timeout breve
        try:
            sock.connect((host, port))
            return True
        except (socket.error, socket.timeout):
            return False

if __name__ == "__main__":
    while not is_tor_proxy_active():
        print("Tor Browser non è aperto riprovo tra 5 secondi...")
        time.sleep(5)
    leak()

