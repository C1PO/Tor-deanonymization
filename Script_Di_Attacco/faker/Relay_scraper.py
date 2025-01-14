
import subprocess

def get_relay_ips(Number_Ip=10):
    try:
        command = (
            f'curl -s "https://deviceandbrowserinfo.com/api/ip_address/tor"'
        )
        result = subprocess.check_output(command, shell=True, text=True)
        result=result.replace("\"","")
        result=result.replace("[","")
        relay_ips = result.strip().split(",")

        valid_ips=relay_ips[0:Number_Ip]
        return valid_ips
    except subprocess.CalledProcessError as e:
        print(f"Errore durante l'esecuzione del comando: {e}")
        return []
