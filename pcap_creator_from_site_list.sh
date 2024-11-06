#!/bin/bash

# Configura i nomi dei container
CLIENT_CONTAINER="client-1"
EXIT_CONTAINER="exit-1"

# Percorsi delle directory sulla scrivania
DESKTOP_PATH=~/Scrivania
CLIENT_PCAP_PATH="$DESKTOP_PATH/pcap/client"
EXIT_PCAP_PATH="$DESKTOP_PATH/pcap/exit"

# Crea le directory se non esistono
mkdir -p "$CLIENT_PCAP_PATH"
mkdir -p "$EXIT_PCAP_PATH"

# Carica la lista di siti dal file sites.txt
SITES=()
while IFS= read -r line; do
    SITES+=("$line")
done < "sites.txt"

# Tempo di attesa tra le richieste curl (in secondi)
WAIT_TIME=10

# Funzione per fermare lo sniffing e salvare i file pcap
stop_sniffing_and_save() {
    local entry_pcap="$1"
    local exit_pcap="$2"
    local site_name="$3"

    echo "Terminando catture per $site_name..."
    kill $ENTRY_PID $EXIT_PID
    wait $ENTRY_PID $EXIT_PID 2>/dev/null

    # Sposta i file pcap nelle rispettive cartelle sulla scrivania
    docker cp "$CLIENT_CONTAINER:$entry_pcap" "$CLIENT_PCAP_PATH/$site_name.pcap"
    docker cp "$EXIT_CONTAINER:$exit_pcap" "$EXIT_PCAP_PATH/$site_name.pcap"

    # Rimuove i file pcap dai container
    docker exec "$CLIENT_CONTAINER" rm -f "$entry_pcap"
    docker exec "$EXIT_CONTAINER" rm -f "$exit_pcap"
}

# Imposta il trap per catturare SIGINT (Ctrl+C)
trap "echo 'Interruzione manuale. Terminazione dello script.'; exit 0" SIGINT

# Per ogni sito nella lista, avvia una nuova sessione di sniffing
for SITE in "${SITES[@]}"; do
    # Rinomina il sito per usarlo come nome del file (sostituisce i caratteri non validi)
    SITE_NAME=$(echo "$SITE" | sed 's|https\?://||; s|[./]|_|g')

    # Percorsi dei file pcap per il sito corrente
    ENTRY_PCAP="/mnt/${SITE_NAME}_entry.pcap"
    EXIT_PCAP="/mnt/${SITE_NAME}_exit.pcap"

    # Avvia lo sniffing per il sito corrente
    docker exec "$CLIENT_CONTAINER" tcpdump -i eth0 -w "$ENTRY_PCAP" &
    ENTRY_PID=$!
    docker exec "$EXIT_CONTAINER" tcpdump -i eth0 -w "$EXIT_PCAP" &
    EXIT_PID=$!

    # Esegui la richiesta curl tramite il circuito Tor
    echo "Visitando $SITE tramite Tor..."
    docker exec "$CLIENT_CONTAINER" curl --socks5 10.5.1.2:9050 "$SITE"

    # Attendi un tempo specificato tra le richieste
    sleep "$WAIT_TIME"

    # Ferma lo sniffing e sposta i file pcap
    stop_sniffing_and_save "$ENTRY_PCAP" "$EXIT_PCAP" "$SITE_NAME"
done

echo "Script completato. I file pcap sono stati salvati nelle directory 'client' ed 'exit' all'interno della cartella pcap sulla tua scrivania."
