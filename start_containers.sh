docker run --name da-1 --ip 10.5.1.10 --network testing-tor -d da-1 sudo -u debian-tor tor
docker run --name da-2 --ip 10.5.1.11 --network testing-tor -d da-2 sudo -u debian-tor tor
docker run --name da-3 --ip 10.5.1.12 --network testing-tor -d da-3 sudo -u debian-tor tor
docker run --name relay-1 --ip 10.5.1.20 --network testing-tor -d relay-1 sudo -u debian-tor tor
docker run --name relay-2 --ip 10.5.1.21 --network testing-tor -d relay-2 sudo -u debian-tor tor
docker run --name relay-3 --ip 10.5.1.22 --network testing-tor -d relay-3 sudo -u debian-tor tor
docker run --name relay-4 --ip 10.5.1.23 --network testing-tor -d relay-4 sudo -u debian-tor tor
docker run --name exit-1 --ip 10.5.1.30 --network testing-tor -d exit-1 sudo -u debian-tor tor
docker run --name exit-2 --ip 10.5.1.31 --network testing-tor -d exit-2 sudo -u debian-tor tor
docker run --name client-1 --network testing-tor --ip 10.5.1.2 -p 9050:9050 -d client-1 sudo -u debian-tor tor
docker run --name webserver --ip 10.5.1.40 -p 8080:80 --network testing-tor -d webserver
