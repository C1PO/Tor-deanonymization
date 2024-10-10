## Create the docker network
``` docker network create testing-tor --subnet 10.5.0.0/16 ```

## Load the images 
``` docker load -i imgs.tar ```

## Start the containers
``` ./start_containers ```

## Verify the connection
``` curl --socks5 10.5.1.2:9050 10.5.1.40:80 ```

## Stop the containers
``` ./stop_containers ```
