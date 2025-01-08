# Permette di creare una porta randome per il servizio Tor sovrascrivendo il torrc
TORRC_FILE="/etc/tor/torrc"

if [ ! -f "$TORRC_FILE" ]; then
    echo "Errore: il file di configurazione Tor non esiste."
    exit 1
fi

cp "$TORRC_FILE" "$TORRC_FILE.bak"

SOCKS_PORT=$((RANDOM % (65535 - 1024 + 1) + 1024))
while ss -tulnp | grep -q ":$SOCKS_PORT"; do
    SOCKS_PORT=$((RANDOM % (65535 - 1024 + 1) + 1024))
done

if grep -qE '^[#]*\s*SocksPort' "$TORRC_FILE"; then
    sed -i "s|^[#]*\s*SocksPort .*|SocksPort $SOCKS_PORT|" "$TORRC_FILE"
else
    echo "SocksPort $SOCKS_PORT" | tee -a "$TORRC_FILE"
fi

generate_random_name() {
    tr -dc A-Za-z0-9 </dev/urandom | head -c 10
}

random_name=$(generate_random_name)

exec -a "$random_name" /usr/bin/tor "$@"
