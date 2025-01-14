# Attacco provava a fare una richiesta socks5 su qualsiasi porta aperta per scoprire quella di tor
SOCKS_PORT=1024
while true; do
    if ss -tulnp | grep -q ":$SOCKS_PORT"; then
    if curl --socks5 127.0.0.1:$SOCKS_PORT https://icanhazip.com; then
        echo "Tor è stato avviato sulla porta $SOCKS_PORT."
        curl --socks5 127.0.0.1:$SOCKS_PORT https://icanhazip.com
        curl --socks5 127.0.0.1:$SOCKS_PORT https://icanhazip.com
        curl https://icanhazip.com
        break
        fi
    fi
    echo "Tor non è nella $SOCKS_PORT"
    SOCKS_PORT=$((SOCKS_PORT + 1))
done
