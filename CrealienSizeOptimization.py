#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import subprocess

# global k
# k=0

# def getAlienCrewsize():
#     return k

def execute_python_file(file_path):
    for j in range(3,20,2):
        # global k
        # k=j
        for i in range(10):
           try:
              completed_process = subprocess.run(['python', file_path], capture_output=True, text=True)
              if completed_process.returncode == 0:
                 print("Execution successful.")
                 print("Output:")
                 print(completed_process.stdout)
              else:
                 print(f"Error: Failed to execute '{file_path}'.")
                 print("Error output:")
                 print(completed_process.stderr)
           except FileNotFoundError:
              print(f"Error: The file '{file_path}' does not exist.")


# file_path ="/Users/admin/Documents/Harshith 520/Project2/Bot3.py"
# file_path ="/Users/admin/Documents/Harshith 520/Project2/myBot1.py"
file_path ="/Users/admin/Documents/Harshith 520/Project2/Test_Grid.py"

execute_python_file(file_path)    
