FROM debian:12.7

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install apt-transport-https wget gpg iproute2 iputils-ping sudo vim procps bash-completion curl tcpdump bsdmainutils telnet net-tools nano htop -y 
RUN echo "deb [signed-by=/usr/share/keyrings/deb.torproject.org-keyring.gpg] https://deb.torproject.org/torproject.org bookworm main" > /etc/apt/sources.list.d/tor.list
RUN echo "deb-src [signed-by=/usr/share/keyrings/deb.torproject.org-keyring.gpg] https://deb.torproject.org/torproject.org bookworm main" >> /etc/apt/sources.list.d/tor.list
RUN wget -qO- https://deb.torproject.org/torproject.org/A3C4F0F979CAA22CDBA8F512EE8CBC9E886DDD89.asc | gpg --dearmor | tee /usr/share/keyrings/deb.torproject.org-keyring.gpg >/dev/null
RUN apt-get update && apt-get install tor deb.torproject.org-keyring nyx -y

# Delete the default torrc file since we are going to replace it
RUN rm /etc/tor/torrc
