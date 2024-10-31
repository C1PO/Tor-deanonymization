import pyshark
import csv
import sys
import os

def convert_pcap_to_csv(pcap_file, output_folder):
    try:
        cap = pyshark.FileCapture(pcap_file)
        file_name = os.path.basename(pcap_file)
        csv_filename = os.path.join(output_folder, f"{file_name}.csv")
        
        header = [
            "Timestamp", "Size", "Source IP", "Source Port", "Destination IP", "Destination Port",
            "Protocol", "Sequence Number", "Acknowledgment Number", "TSval", "TSecr"
        ]
        
        with open(csv_filename, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)
            for packet in cap:
                if 'TCP' in packet:
                    try:
                        src_ip = packet.ip.src
                        size = packet.length
                        dst_ip = packet.ip.dst
                        src_port = packet.tcp.srcport
                        dst_port = packet.tcp.dstport
                        protocol = packet.transport_layer
                        seq = packet.tcp.seq
                        ack = packet.tcp.ack
                        timestamp = packet.sniff_time
                        tcp_tsval = packet.tcp.options_timestamp_tsval if 'options_timestamp_tsval' in packet.tcp.field_names else None
                        tcp_tsecr = packet.tcp.options_timestamp_tsecr if 'options_timestamp_tsecr' in packet.tcp.field_names else None
                        writer.writerow([
                            timestamp, size, src_ip, src_port, dst_ip, dst_port,
                            protocol, seq, ack, tcp_tsval, tcp_tsecr
                        ])
                    except AttributeError as e:
                        print(f"Errore nell'estrazione di un campo in {pcap_file}: {e}")
        print(f"Dati salvati in {csv_filename}")
    except Exception as e:
        print(f"Errore nella conversione del file {pcap_file}: {e}")

if len(sys.argv) < 2:
    print("-" * 50)
    print("Utilizza il programma come segue:")
    print("'python main.py cartella_pcaps path_dove_salvare_csv'")
    print("La cartella può essere con o senza / ma deve esistere")
    print("-" * 50)
    sys.exit()

input_folder = sys.argv[1]
output_folder = sys.argv[2] if len(sys.argv) > 2 else input_folder

if not os.path.isdir(input_folder):
    print(f"Errore: la cartella {input_folder} non esiste.")
    sys.exit()

if not os.path.isdir(output_folder):
    print(f"Errore: la cartella {output_folder} non esiste.")
    sys.exit()

for filename in os.listdir(input_folder):
    if filename.endswith(".pcap"):
        pcap_path = os.path.join(input_folder, filename)
        convert_pcap_to_csv(pcap_path, output_folder)
