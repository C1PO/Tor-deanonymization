import subprocess
import os

def run_tcpdump_in_container(container_name="exit-1", interface="eth0", output_file="/tmp/tcpdump_output.pcap"):
    """
    Runs tcpdump in the specified container and copies the output file to the host machine.

    :param container_name: Name of the Docker container
    :param interface: Network interface to sniff (default is eth0)
    :param output_file: Path inside the container where tcpdump will save the output
    """
    try:
        # Command to run tcpdump inside the container
        tcpdump_cmd = f"tcpdump -i {interface} -w {output_file}"

        # Execute tcpdump inside the container
        print(f"Running tcpdump in container {container_name} on interface {interface}...")
        subprocess.run(["docker", "exec", "-d", container_name, "sh", "-c", tcpdump_cmd], check=True)
        print(f"Tcpdump started in container {container_name}. Capturing packets...")

        # Give some time to capture packets
        input("Press Enter to stop tcpdump and retrieve the file...")

        # Stop tcpdump (by stopping the process)
        subprocess.run(["docker", "exec", container_name, "pkill", "tcpdump"], check=True)
        print("Tcpdump stopped.")

        # Copy the file from the container to the host machine
        local_output = os.path.join(os.getcwd(), f"{container_name}_output.pcap")
        print(f"Copying tcpdump output from container to {local_output}...")
        subprocess.run(["docker", "cp", f"{container_name}:{output_file}", local_output], check=True)
        print(f"File copied successfully to {local_output}.")

        # Optionally, remove the file from the container to clean up
        subprocess.run(["docker", "exec", container_name, "rm", output_file], check=True)
        print("Cleaned up the output file inside the container.")

    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("Process interrupted by user.")

if __name__ == "__main__":
    # Example usage: capture packets on relay-1's eth0 interface
    run_tcpdump_in_container(container_name="exit-1", interface="eth0")
    #run_tcpdump_in_container(container_name="relay-1", interface="eth0")