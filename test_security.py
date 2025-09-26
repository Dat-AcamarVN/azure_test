import os
import subprocess

def run_command(input_str):
    cmd = f"echo {input_str}"  # Tiềm năng command injection
    subprocess.run(cmd, shell=True)

run_command("test")