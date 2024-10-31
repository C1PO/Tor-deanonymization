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

# Lista di siti da visitare
SITES=(
    "http://example.com"
    "http://example.org"
    "http://example.net"
    "http://openai.com"
    "http://wikipedia.org"
    "http://github.com"
    "http://stackoverflow.com"
    "http://reddit.com"
    "http://mozilla.org"
    "http://gnu.org"
    "http://python.org"
    "http://linux.org"
    "http://apple.com"
    "http://microsoft.com"
    "http://oracle.com"
    "http://cloudflare.com"
    "http://digitalocean.com"
    "http://aws.amazon.com"
    "http://azure.microsoft.com"
    "http://ibm.com"
    "http://twitter.com"
    "http://facebook.com"
    "http://linkedin.com"
    "http://instagram.com"
    "http://youtube.com"
    "http://netflix.com"
    "http://imdb.com"
    "http://bbc.com"
    "http://cnn.com"
    "http://nytimes.com"
    "http://forbes.com"
    "http://bloomberg.com"
    "http://reuters.com"
    "http://theguardian.com"
    "http://espn.com"
    "http://weather.com"
    "http://nasa.gov"
    "http://who.int"
    "http://cdc.gov"
    "http://unesco.org"
    "http://mit.edu"
    "http://stanford.edu"
    "http://harvard.edu"
    "http://cam.ac.uk"
    "http://ox.ac.uk"
    "http://nature.com"
    "http://sciencedirect.com"
    "http://springer.com"
    "http://researchgate.net"
    "http://arxiv.org"
)


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
