import random
import time
import os
from scapy.all import *
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

def generate_rsa_keys():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    return private_key, public_key

def encrypt_with_aes(data):
    aes_key = os.urandom(32)
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(aes_key), modes.CFB(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(data) + encryptor.finalize()
    return aes_key, iv, encrypted_data

def encrypt_aes_key_with_rsa(public_key, aes_key):
    encrypted_key = public_key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return encrypted_key

def create_realistic_tcp_packet(src_ip, dst_ip, src_port, dst_port, seq, ack, flags, payload):
    ip = IP(src=src_ip, dst=dst_ip)
    tcp = TCP(sport=src_port, dport=dst_port, seq=seq, ack=ack, flags=flags)
    return ip / tcp / Raw(load=payload)

def calculate_payload_size(desired_packet_size):
    ip_header_size = 20
    tcp_header_size = 20
    headers_size = ip_header_size + tcp_header_size
    return desired_packet_size - headers_size

def simulate_tor_traffic(src_ip, dst_ip, src_port, dst_port, public_key, num_packets=10, packet_size=606):
    seq = random.randint(1, 2**32 - 1)
    ack = random.randint(1, 2**32 - 1)
    
    payload_size = calculate_payload_size(packet_size)

    for i in range(num_packets):
        data = os.urandom(payload_size - 290)
        aes_key, iv, encrypted_data = encrypt_with_aes(data)
        encrypted_aes_key = encrypt_aes_key_with_rsa(public_key, aes_key)
        final_payload = encrypted_aes_key + iv + encrypted_data

        flags = "PA" if i > 0 else "S"

        packet = create_realistic_tcp_packet(src_ip, dst_ip, src_port, dst_port, seq, ack, flags, final_payload)
        send(packet, verbose=False)

        seq, ack = forge_tcp_response(packet)

        time.sleep(random.uniform(0.05, 0.15))

def forge_tcp_response(packet):
    ip_layer = packet[IP]
    tcp_layer = packet[TCP]
    
    payload = bytes(packet[Raw].load) if packet.haslayer(Raw) else b""

    ip = IP(src=ip_layer.dst, dst=ip_layer.src)
    seq = tcp_layer.ack
    ack = tcp_layer.seq + len(payload)

    tcp = TCP(
        sport=tcp_layer.dport, dport=tcp_layer.sport,
        seq=seq,
        ack=ack,
        flags="A"
    )
    
    response_packet = ip / tcp / Raw(load=payload)

    send(response_packet, verbose=False)
    
    return seq, ack

def start(dst_ip, src_ip="192.168.1.16"):
    private_key, public_key = generate_rsa_keys()
    src_port = 9050
    dst_port = 9001
    simulate_tor_traffic(src_ip, dst_ip, src_port, dst_port, public_key, num_packets=20, packet_size=606)
