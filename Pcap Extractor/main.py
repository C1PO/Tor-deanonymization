import pyshark
import csv
import sys
import os

if len(sys.argv)<2 or not(sys.argv[1].endswith(".pcap")):
    print("-" * 50)
    print("Utilizza il programma come segue:")
    print("'python main.py file_pcap'")
    print("'python main.py file_pcap path_dove_salvare_il_csv'")
    print("La cartella può essere con o senza / ma deve esistere")
    print("-" * 50)
    exit
else:
    cap = pyshark.FileCapture(sys.argv[1])
    file_name = os.path.basename(sys.argv[1])
    if len(sys.argv) == 3:
        if sys.argv[2].endswith("/"):
            csv_filename = f'{sys.argv[2]}{file_name}.csv'
        else:
            csv_filename = f'{sys.argv[2]}/{file_name}.csv'
    else:
        csv_filename = f'{sys.argv[1]}.csv'
    header = [
        "Timestamp","Size","Source IP", "Source Port", "Destination IP", "Destination Port",
        "Protocol", "Sequence Number", "Acknowledgment Number", "TSval", "TSecr"
    ]
    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for packet in cap:
            if 'TCP' in packet:
                try:
                    src_ip = packet.ip.src
                    size=packet.length
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
                        timestamp,size,src_ip, src_port, dst_ip, dst_port,
                        protocol, seq, ack, tcp_tsval, tcp_tsecr
                    ])

                except AttributeError as e:
                    print(f"Errore nell'estrazione di un campo: {e}")
    print(f"Dati salvati in {csv_filename}")
