from mitmproxy import http
from scapy.all import IP, TCP, Raw, send
import random
import time

def request(flow: http.HTTPFlow) -> None:
    packet = IP(dst=flow.request.host) / TCP(dport=flow.request.port) / flow.request.content
    print(f"Pacchetto intercettato:\n{packet.summary()}")
    delay = random.uniform(0.01, 0.1)
    print(f"Applicato ritardo di {delay:.2f} secondi.")
    time.sleep(delay)
    try:
        if packet.haslayer(IP) and packet.haslayer(TCP) and packet.haslayer(Raw):
            payload = packet[Raw].load
            current_length = len(payload)
            max_length = 1500
            if current_length < max_length:
                padding_length = random.randint(1, max_length - current_length)
                payload += b'\x00' * padding_length
                packet[Raw].load = payload

                del packet[IP].len
                del packet[IP].chksum
                del packet[TCP].chksum
                send(packet, verbose=0)
                print("Pacchetto inviato con successo.")
                print("Grandezza pacchetto ",len(payload))
            else:
                print("Pacchetto troppo grande per aggiungere padding.")
                send(packet, verbose=0)
        else:
            print("Pacchetto non conforme per la modifica.")
            send(packet, verbose=0)
        print("Pacchetto inviato con Scapy.")
    except Exception as e:
        print(f"Errore durante l'invio del pacchetto: {e}")
