import streamlit as st
import time
import os
import shutil
import subprocess
import smtplib
from email.mime.text import MIMEText
import webbrowser # For opening URLs in browser
import zipfile # For zipping/unzipping files
import json # For parsing Docker inspect output and JSON input for ML prediction
import platform # Required for system information in Windows tasks

# Import ML libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import joblib # For model persistence

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, r2_score, silhouette_score,
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, roc_curve # Added roc_curve for ROC plot
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Interactive plotting library
import plotly.express as px
import plotly.graph_objects as go

# Conditional imports for libraries that might not be present by default
try:
    import pyautogui # For simulating mouse/keyboard interactions
except ImportError:
    pyautogui = None

try:
    import cv2 # For webcam access and video recording
except ImportError:
    cv2 = None

try:
    import psutil # For system information (disk usage, processes)
except ImportError:
    psutil = None

try:
    import wmi # For Windows Management Instrumentation (e.g., listing installed programs)
except ImportError:
    wmi = None
    # st.warning("`wmi` not found. Install with `pip install wmi` for some Windows functionalities.") # Removed warning to avoid early Streamlit call

try:
    import pywhatkit # For sending WhatsApp messages
except ImportError:
    pywhatkit = None

try:
    import paramiko # For SSH connections
except ImportError:
    paramiko = None
    # st.warning("`paramiko` not found. Install with `pip install paramiko` for SSH functionality.") # Removed warning to avoid early Streamlit call

try:
    import xgboost as xgb # For XGBoost models
except ImportError:
    xgb = None
    # st.warning("`xgboost` not found. Install with `pip install xgboost` for XGBoost models.") # Removed warning to avoid early Streamlit call

try:
    import lightgbm as lgb # For LightGBM models
except ImportError:
    lgb = None
    # st.warning("`lightgbm` not found. Install with `pip install lightgbm` for LightGBM models.") # Removed warning to avoid early Streamlit call

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


# --- Set Page Config (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="Multi-Task Dashboard",
    page_icon="🛠️",
    layout="wide"
)
# --- End Set Page Config ---


# --- Required Libraries Installation Notes ---
# The following libraries are built-in Python libraries and do NOT require 'pip install':
# time, os, shutil, subprocess, smtplib, email.mime.text, webbrowser, zipfile, io, json, platform
#
# To make all other functional tasks work, run these commands in your terminal:
# pip install streamlit
# pip install twilio
# pip install pyautogui
# pip install opencv-python
# pip install psutil
# pip install wmi # For Windows-specific system info
# pip install pywhatkit
# pip install paramiko
# pip install pandas scikit-learn matplotlib seaborn plotly
# pip install xgboost lightgbm # For advanced ML models
# ---------------------------------------------

# --- GUI Styling ---
st.markdown("""
<style>
/* Import modern, legible font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

:root {
  --bg: #0b1020;
  --bg-alt: #111733;
  --surface: #161e3f;
  --surface-2: #1d2752;
  --primary: #4f7cff;
  --primary-600: #3b68ff;
  --primary-700: #2f56e6;
  --text: #e6e9f2;
  --muted: #9aa4c7;
  --border: rgba(255,255,255,0.08);
  --success: #22c55e;
  --warning: #f59e0b;
  --danger: #ef4444;
}

/* App background and base layout */
.stApp {
  background: radial-gradient(1200px 800px at 10% 0%, #0e1530 0%, var(--bg) 50%, #090d1a 100%);
}

/* Main content container */
section.main > div.block-container {
  padding-top: 1.5rem;
  padding-bottom: 3rem;
  max-width: 1320px;
}

/* Typography */
html, body, [class^="css"] {
  font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  color: var(--text);
}

h1, h2, h3 {
  letter-spacing: 0.2px;
  margin-top: 0.25rem;
  margin-bottom: 0.75rem;
}

h1 { font-weight: 700; }
h2, h3 { font-weight: 600; }

/* Horizontal rule spacing */
hr { 
  border: 0; height: 1px; 
  background: linear-gradient(to right, transparent, var(--border), transparent);
  margin: 1.25rem 0 1.25rem 0;
}

/* Sidebar */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, var(--bg-alt) 0%, #0d1431 100%);
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .block-container {
  padding-top: 1rem;
}

/* Buttons - unified style */
div.stButton > button {
  background: linear-gradient(180deg, var(--primary) 0%, var(--primary-700) 100%);
  color: white;
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 12px;
  padding: 0.6rem 1rem;
  font-weight: 600;
  font-size: 0.98rem;
  letter-spacing: 0.2px;
  box-shadow: 0 6px 16px rgba(79,124,255,0.25);
  transition: all 150ms ease;
}
div.stButton > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 10px 20px rgba(79,124,255,0.28);
  filter: brightness(1.02);
}
div.stButton > button:active {
  transform: translateY(0);
  filter: brightness(0.98);
}

/* Large button grid for main menu */
.btn-grid-lg div.stButton > button {
  width: 100%;
  min-height: 150px;
  font-size: 1.1rem;
}

/* Medium button grid for sub-menus */
.btn-grid div.stButton > button {
  width: 100%;
  min-height: 120px;
}

/* Inputs */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input,
[data-testid="stTimeInput"] input,
[data-testid="stSelectbox"] div[role="combobox"],
[data-testid="stMultiSelect"] div[role="combobox"] {
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));
  border: 1px solid var(--border);
  color: var(--text);
  border-radius: 12px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
}

[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder {
  color: var(--muted);
}

/* File uploader */
[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] {
  background: rgba(255,255,255,0.02);
  border: 1px dashed var(--border);
  border-radius: 14px;
}

/* Slider */
[data-baseweb="slider"] > div {
  background: var(--surface);
}

/* Tables / DataFrames */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
}
[data-testid="stDataFrame"] * {
  color: var(--text) !important;
}

/* Metrics and alerts */
.stAlert { border-radius: 12px; border: 1px solid var(--border); }
.stSuccess { background: rgba(34,197,94,0.08); }
.stWarning { background: rgba(245,158,11,0.08); }
.stError { background: rgba(239,68,68,0.08); }

/* Cards utility (optional wrapper via markdown if needed) */
.card {
  background: linear-gradient(180deg, var(--surface) 0%, var(--surface-2) 100%);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1rem 1.25rem;
  box-shadow: 0 8px 24px rgba(0,0,0,0.22);
}

/* Compact markdown lists spacing */
ul, ol { margin-top: 0.25rem; }

</style>
""", unsafe_allow_html=True)
# --- End GUI Styling ---


# --- SSH Execution Function ---
def execute_ssh_command(host, username, password, command):
    """Executes a command over SSH and returns stdout and stderr."""
    if not paramiko:
        return "", "Paramiko library not found. Please install it with `pip install paramiko`."

    try:
        with st.spinner(f"Executing '{command}' on {host}..."):
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(hostname=host, username=username, password=password, timeout=10) # Increased timeout slightly
            
            # Execute the command
            stdin, stdout, stderr = client.exec_command(command)
            output = stdout.read().decode('utf-8').strip()
            error = stderr.read().decode('utf-8').strip()
            
            client.close()
            return output, error
    except paramiko.AuthenticationException:
        return "", "Authentication failed. Please check your username and password."
    except paramiko.SSHException as ssh_err:
        return "", f"SSH connection error: {ssh_err}. Ensure SSH server is running and accessible (e.g., SSH service is active, firewall allows port 22)."
    except Exception as e:
        return "", f"An unexpected error occurred during SSH command execution: {e}"

# --- Windows Task Functions ---

def display_windows_system_info_tasks():
    st.subheader("System Information & Monitoring ")

    if st.button("Get OS Information"):
        st.write(f"**Operating System:** {os.name}")
        st.write(f"**Platform:** {platform.system()} {platform.release()} ({platform.version()})")
        st.write(f"**Machine:** {platform.machine()}")
        st.write(f"**Processor:** {platform.processor()}")

    if st.button("Monitor CPU & RAM Usage"):
        if psutil:
            st.write(f"**CPU Usage:** {psutil.cpu_percent(interval=1)}%")
            ram = psutil.virtual_memory()
            st.write(f"**RAM Usage:** {ram.percent}% (Used: {ram.used / (1024**3):.2f} GB / Total: {ram.total / (1024**3):.2f} GB)")
        else:
            st.error("`psutil` not found. Install with `pip install psutil` for this functionality.")

    if st.button("List Running Processes"):
        if psutil:
            st.write("### Running Processes (Top 20 by CPU)")
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
                try:
                    pinfo = proc.as_dict(attrs=['pid', 'name', 'cpu_percent', 'memory_info'])
                    pinfo['memory_mb'] = pinfo['memory_info'].rss / (1024 * 1024)
                    processes.append(pinfo)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
            
            processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
            st.dataframe(pd.DataFrame(processes[:20]).drop(columns=['memory_info']))
        else:
            st.error("`psutil` not found. Install with `pip install psutil` for this functionality.")


def display_windows_file_folder_operations_tasks():
    st.subheader("File & Folder Operations")
    st.text_input("Base Path (Server-side)", key="base_file_path_input", value=os.getcwd()) # Default to current working directory

    # Create New Folder
    new_folder_name = st.text_input("New Folder Name", key="create_folder_name_file_ops")
    if st.button("Create New Folder"):
        if new_folder_name:
            try:
                folder_path = os.path.join(st.session_state.base_file_path_input, new_folder_name)
                os.makedirs(folder_path, exist_ok=True)
                st.success(f"Folder '{folder_path}' created successfully on the server.")
            except Exception as e:
                st.error(f"Error creating folder: {e}")
        else:
            st.warning("Please enter a folder name.")

    # Delete File/Folder
    delete_path_input = st.text_input("File/Folder to Delete (Full Path)", key="delete_path_input_file_ops")
    if st.button("Delete File/Folder"):
        if delete_path_input:
            try:
                if os.path.isfile(delete_path_input):
                    os.remove(delete_path_input)
                    st.success(f"File '{delete_path_input}' deleted successfully on the server.")
                elif os.path.isdir(delete_path_input):
                    shutil.rmtree(delete_path_input)
                    st.success(f"Folder '{delete_path_input}' and its contents deleted successfully on the server.")
                else:
                    st.warning("Path does not exist or is not a file/folder on the server.")
            except Exception as e:
                st.error(f"Error deleting: {e}")
        else:
            st.warning("Please enter a path to delete.")

    # Copy File/Folder
    source_path_copy = st.text_input("Source Path to Copy", key="source_copy_file_ops")
    destination_path_copy = st.text_input("Destination Path for Copy", key="dest_copy_file_ops")
    if st.button("Copy File/Folder"):
        if source_path_copy and destination_path_copy:
            try:
                if os.path.isfile(source_path_copy):
                    shutil.copy2(source_path_copy, destination_path_copy)
                    st.success(f"File '{source_path_copy}' copied to '{destination_path_copy}' on the server.")
                elif os.path.isdir(source_path_copy):
                    shutil.copytree(source_path_copy, destination_path_copy)
                    st.success(f"Folder '{source_path_copy}' copied to '{destination_path_copy}' on the server.")
                else:
                    st.warning("Source path does not exist or is not a file/folder on the server.")
            except Exception as e:
                st.error(f"Error copying: {e}")
        else:
            st.warning("Please enter source and destination paths.")

    # Move File/Folder
    source_path_move = st.text_input("Source Path to Move", key="source_move_file_ops")
    destination_path_move = st.text_input("Destination Path for Move", key="dest_move_file_ops")
    if st.button("Move File/Folder"):
        if source_path_move and destination_path_move:
            try:
                shutil.move(source_path_move, destination_path_move)
                st.success(f"'{source_path_move}' moved to '{destination_path_move}' on the server.")
            except Exception as e:
                st.error(f"Error moving: {e}")
        else:
            st.warning("Please enter source and destination paths.")

    # Rename File/Folder
    old_name = st.text_input("Old Name (Full Path)", key="old_name_file_ops")
    new_name = st.text_input("New Name (Full Path)", key="new_name_file_ops")
    if st.button("Rename File/Folder"):
        if old_name and new_name:
            try:
                os.rename(old_name, new_name)
                st.success(f"'{old_name}' renamed to '{new_name}' on the server.")
            except Exception as e:
                st.error(f"Error renaming: {e}")
        else:
            st.warning("Please enter old and new names.")

    # Search for Files
    search_path = st.text_input("Search Path (e.g., C:\\)", key="search_path_input_file_ops", value=os.getcwd())
    search_term = st.text_input("Search Term (e.g., .txt)", key="search_term_input_file_ops")
    if st.button("Search for Files"):
        if search_path and search_term:
            found_files = []
            for root, _, files in os.walk(search_path):
                for file in files:
                    if search_term.lower() in file.lower():
                        found_files.append(os.path.join(root, file))
            if found_files:
                st.write("### Found Files:")
                for f in found_files:
                    st.write(f)
            else:
                st.info("No files found matching the search term.")
        else:
            st.warning("Please enter a search path and term.")

    # Compress Folder (ZIP)
    folder_to_zip = st.text_input("Folder to Compress (Full Path)", key="folder_to_zip_file_ops")
    zip_output_name = st.text_input("Output Zip File Name (e.g., my_archive.zip)", key="zip_output_name_file_ops")
    if st.button("Compress Folder (ZIP)"):
        if folder_to_zip and zip_output_name:
            try:
                shutil.make_archive(zip_output_name.replace(".zip", ""), 'zip', folder_to_zip)
                st.success(f"Folder '{folder_to_zip}' compressed to '{zip_output_name}' on the server.")
            except Exception as e:
                st.error(f"Error compressing folder: {e}")
        else:
            st.warning("Please enter folder to compress and output zip name.")

    # Extract Files (Unzip)
    zip_file_to_extract = st.text_input("Zip File to Extract (Full Path)", key="zip_file_to_extract_file_ops")
    extract_destination = st.text_input("Extraction Destination Folder", key="extract_destination_file_ops", value=os.getcwd())
    if st.button("Extract Files (Unzip)"):
        if zip_file_to_extract and extract_destination:
            try:
                with zipfile.ZipFile(zip_file_to_extract, 'r') as zip_ref:
                    zip_ref.extractall(extract_destination)
                st.success(f"'{zip_file_to_extract}' extracted to '{extract_destination}' on the server.")
            except FileNotFoundError:
                st.error("Zip file not found.")
            except zipfile.BadZipFile:
                st.error("Invalid zip file.")
            except Exception as e:
                st.error(f"Error extracting files: {e}")
        else:
            st.warning("Please enter zip file and extraction destination.")

    st.markdown("---")
    st.write("### Advanced File Operations")
    file_content_search_path = st.text_input("Path to search for content (e.g., C:\\MyDocs)", key="file_content_search_path")
    file_content_search_term = st.text_input("Text to search within files", key="file_content_search_term")
    if st.button("Search Text in Files"):
        if file_content_search_path and file_content_search_term:
            found_files_with_content = []
            for root, _, files in os.walk(file_content_search_path):
                for file in files:
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            if file_content_search_term.lower() in f.read().lower():
                                found_files_with_content.append(filepath)
                    except Exception as e:
                        st.warning(f"Could not read file {filepath}: {e}")
            if found_files_with_content:
                st.write("### Files Containing Search Term:")
                for f in found_files_with_content:
                    st.write(f)
            else:
                st.info("No files found containing the search term.")
        else:
            st.warning("Please enter a search path and term.")

def display_application_management_tasks():
    st.subheader("Application Management Tasks")

    # Removed "List Installed Programs" and "Manage Startup Programs" as requested.

    if st.button("Open Calculator"):
        try:
            subprocess.Popen(["calc.exe"])
            st.success("Attempted to open Calculator.")
        except FileNotFoundError:
            st.error("Calculator (calc.exe) not found.")
        except Exception as e:
            st.error(f"Error opening Calculator: {e}")

    if st.button("Open Notepad"):
        try:
            subprocess.Popen(["notepad.exe"])
            st.success("Attempted to open Notepad.")
        except FileNotFoundError:
            st.error("Notepad (notepad.exe) not found.")
        except Exception as e:
            st.error(f"Error opening Notepad: {e}")

    if st.button("Open Command Prompt"):
        try:
            subprocess.Popen(["cmd.exe"])
            st.success("Attempted to open Command Prompt.")
        except FileNotFoundError:
            st.error("Command Prompt (cmd.exe) not found.")
        except Exception as e:
            st.error(f"Error opening Command Prompt: {e}")

    if st.button("Open PowerShell"):
        try:
            subprocess.Popen(["powershell.exe"])
            st.success("Attempted to open PowerShell.")
        except FileNotFoundError:
            st.error("PowerShell (powershell.exe) not found.")
        except Exception as e:
            st.error(f"Error opening PowerShell: {e}")

    if st.button("Open Control Panel"):
        try:
            subprocess.Popen(["control.exe"])
            st.success("Attempted to open Control Panel.")
        except FileNotFoundError:
            st.error("Control Panel (control.exe) not found.")
        except Exception as e:
            st.error(f"Error opening Control Panel: {e}")

    if st.button("Open Settings App"):
        try:
            subprocess.Popen(["start", "ms-settings:"], shell=True)
            st.success("Attempted to open Windows Settings app.")
        except Exception as e:
            st.error(f"Error opening Settings app: {e}")

    # New functionalities as requested
    st.markdown("---")
    st.write("### Web Applications & Browser")

    if st.button("Open Chrome"):
        try:
            # Attempt to open Chrome directly. This assumes Chrome is in PATH or its default install location.
            # On Windows, 'start chrome' often works, or specify full path.
            subprocess.Popen(["start", "chrome"], shell=True)
            st.success("Attempted to open Chrome. Ensure Chrome is installed and in your system's PATH.")
        except FileNotFoundError:
            st.error("Chrome executable not found. Please ensure Chrome is installed and its path is correctly configured.")
        except Exception as e:
            st.error(f"Error opening Chrome: {e}")

    if st.button("Open YouTube"):
        try:
            webbrowser.open("https://www.youtube.com/")
            st.success("Opening YouTube in your default web browser.")
        except Exception as e:
            st.error(f"Error opening YouTube: {e}")

    if st.button("Open Google Photos"):
        try:
            webbrowser.open("https://photos.google.com/")
            st.success("Opening Google Photos in your default web browser.")
        except Exception as e:
            st.error(f"Error opening Google Photos: {e}")

    if st.button("Open ChatGPT"):
        try:
            webbrowser.open("https://chat.openai.com/")
            st.success("Opening ChatGPT in your default web browser.")
        except Exception as e:
            st.error(f"Error opening ChatGPT: {e}")

    if st.button("Open Spotify Web Player"):
        try:
            webbrowser.open("https://open.spotify.com/")
            st.success("Opening Spotify Web Player in your default web browser.")
        except Exception as e:
            st.error(f"Error opening Spotify Web Player: {e}")

    st.markdown("---")
    st.write("### System Management")

    service_name_win = st.text_input("Windows Service Name (e.g., 'Spooler')", key="win_service_name")
    col_win_svc1, col_win_svc2, col_win_svc3 = st.columns(3)
    with col_win_svc1:
        if st.button("Start Service", key="start_win_svc"):
            if service_name_win:
                try:
                    subprocess.run(["net", "start", service_name_win], check=True, capture_output=True, text=True)
                    st.success(f"Service '{service_name_win}' started.")
                except subprocess.CalledProcessError as e:
                    st.error(f"Error starting service: {e.stderr}")
            else: st.warning("Please enter a service name.")
    with col_win_svc2:
        if st.button("Stop Service", key="stop_win_svc"):
            if service_name_win:
                try:
                    subprocess.run(["net", "stop", service_name_win], check=True, capture_output=True, text=True)
                    st.success(f"Service '{service_name_win}' stopped.")
                except subprocess.CalledProcessError as e:
                    st.error(f"Error stopping service: {e.stderr}")
            else: st.warning("Please enter a service name.")
    with col_win_svc3:
        if st.button("Restart Service", key="restart_win_svc"):
            if service_name_win:
                try:
                    st.info(f"Attempting to restart service '{service_name_win}'...")
                    subprocess.run(["net", "stop", service_name_win], check=True, capture_output=True, text=True)
                    time.sleep(1) # Give it a moment to stop
                    subprocess.run(["net", "start", service_name_win], check=True, capture_output=True, text=True)
                    st.success(f"Service '{service_name_win}' restarted.")
                except subprocess.CalledProcessError as e:
                    st.error(f"Error restarting service: {e.stderr}")
            else: st.warning("Please enter a service name.")

    if st.button("List All Services"):
        try:
            result = subprocess.run(["sc", "query", "state=", "all"], check=True, capture_output=True, text=True)
            st.code(result.stdout)
        except subprocess.CalledProcessError as e:
            st.error(f"Error listing services: {e.stderr}")

    st.markdown("---")
    st.write("### Scheduled Tasks")
    if st.button("List Scheduled Tasks"):
        try:
            result = subprocess.run(["schtasks", "/query", "/fo", "list", "/v"], check=True, capture_output=True, text=True)
            st.code(result.stdout)
        except subprocess.CalledProcessError as e:
            st.error(f"Error listing scheduled tasks: {e.stderr}")

    task_name_sch = st.text_input("Scheduled Task Name", key="sch_task_name")
    col_sch_task1, col_sch_task2 = st.columns(2)
    with col_sch_task1:
        if st.button("Disable Task", key="disable_sch_task"):
            if task_name_sch:
                try:
                    subprocess.run(["schtasks", "/change", "/tn", task_name_sch, "/disable"], check=True, capture_output=True, text=True)
                    st.success(f"Scheduled task '{task_name_sch}' disabled.")
                except subprocess.CalledProcessError as e:
                    st.error(f"Error disabling task: {e.stderr}")
            else: st.warning("Please enter a task name.")
    with col_sch_task2:
        if st.button("Enable Task", key="enable_sch_task"):
            if task_name_sch:
                try:
                    subprocess.run(["schtasks", "/change", "/tn", task_name_sch, "/enable"], check=True, capture_output=True, text=True)
                    st.success(f"Scheduled task '{task_name_sch}' enabled.")
                except subprocess.CalledProcessError as e:
                    st.error(f"Error enabling task: {e.stderr}")
            else: st.warning("Please enter a task name.")

def display_connectivity_network_tasks():
    st.subheader("Connectivity & Network Tasks")
    # Ping a Host
    ping_host = st.text_input("Host to Ping", key="ping_host", value="google.com")
    if st.button("Ping Host"):
        if ping_host:
            try:
                command = ["ping", "-n", "4", ping_host] if os.name == 'nt' else ["ping", "-c", "4", ping_host]
                result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
                st.code(result.stdout)
                if result.stderr:
                    st.error(result.stderr)
            except subprocess.CalledProcessError as e:
                st.error(f"Ping failed: {e.stderr}")
            except subprocess.TimeoutExpired:
                st.error(f"Ping command timed out after 10 seconds.")
            except Exception as e:
                st.error(f"Error executing ping: {e}")
        else:
            st.warning("Please enter a host to ping.")

    # Trace Route to Host
    tracert_host = st.text_input("Host for Trace Route", key="tracert_host", value="google.com")
    if st.button("Trace Route"):
        if tracert_host:
            try:
                command = ["tracert", tracert_host] if os.name == 'nt' else ["traceroute", tracert_host]
                result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
                st.code(result.stdout)
                if result.stderr:
                    st.error(result.stderr)
            except subprocess.CalledProcessError as e:
                st.error(f"Trace route failed: {e.stderr}")
            except subprocess.TimeoutExpired:
                st.error(f"Trace route command timed out after 30 seconds.")
            except Exception as e:
                st.error(f"Error executing trace route: {e}")
        else:
            st.warning("Please enter a host for trace route.")

    if st.button("View Wi-Fi Profiles (No Passwords)"):
        try:
            st.info("Listing Wi-Fi profiles. Passwords are not displayed for security reasons and require specific administrative permissions.")
            result = subprocess.run(["netsh", "wlan", "show", "profile"], capture_output=True, text=True, check=True)
            st.code(result.stdout)
            if result.stderr:
                st.error(result.stderr)
        except subprocess.CalledProcessError as e:
            st.error(f"Failed to show Wi-Fi profiles: {e.stderr}. (Requires Administrator privileges)")
        except Exception as e:
            st.error(f"Error viewing Wi-fi profiles: {e}")

    st.markdown("---")
    st.write("### Network Drive & Folder Sharing")
    share_folder_path = st.text_input("Folder Path to Share", key="share_folder_path")
    share_name = st.text_input("Share Name", key="share_name")
    if st.button("Share Folder"):
        if share_folder_path and share_name:
            try:
                result = subprocess.run(["net", "share", share_name, share_folder_path, "/grant:Everyone,Full"], capture_output=True, text=True, check=True)
                st.code(result.stdout)
                st.success(f"Folder '{share_folder_path}' shared as '{share_name}'. (Requires Administrator privileges)")
            except subprocess.CalledProcessError as e:
                st.error(f"Failed to share folder: {e.stderr}. (Requires Administrator privileges)")
            except Exception as e:
                st.error(f"Error sharing folder: {e}")
            else:
                st.warning("Please enter folder path and share name.")

    map_drive_path = st.text_input("Network Path to Map (e.g., \\\\server\\share)", key="map_drive_path")
    drive_letter = st.text_input("Drive Letter (e.g., Z:)", key="drive_letter")
    if st.button("Map Network Drive"):
        if map_drive_path and drive_letter:
            try:
                result = subprocess.run(["net", "use", drive_letter, map_drive_path], capture_output=True, text=True, check=True)
                st.code(result.stdout)
                st.success(f"Network path '{map_drive_path}' mapped to '{drive_letter}'.")
            except subprocess.CalledProcessError as e:
                st.error(f"Failed to map network drive: {e.stderr}. (May require credentials or admin rights)")
            except Exception as e:
                st.error(f"Error mapping network drive: {e}")
            else:
                st.warning("Please enter network path and drive letter.")

    st.markdown("---")
    st.write("### Network Adapter Management")
    adapter_name = st.text_input("Network Adapter Name (e.g., 'Ethernet')", key="adapter_name")
    if st.button("Disable Network Adapter"):
        if adapter_name:
            try:
                st.warning(f"Attempting to disable '{adapter_name}'. This will disconnect you from the network. Requires Administrator privileges.")
                subprocess.run(["netsh", "interface", "set", "interface", adapter_name, "admin=disable"], capture_output=True, text=True, check=True)
                st.success(f"Network adapter '{adapter_name}' disabled.")
            except subprocess.CalledProcessError as e:
                st.error(f"Failed to disable adapter: {e.stderr}. (Requires Administrator privileges)")
            except Exception as e:
                st.error(f"Error disabling adapter: {e}")
            else:
                st.warning("Please enter the network adapter name.")

    if st.button("Enable Network Adapter"):
        if adapter_name:
            try:
                st.warning(f"Attempting to enable '{adapter_name}'. Requires Administrator privileges.")
                subprocess.run(["netsh", "interface", "set", "interface", adapter_name, "admin=enable"], capture_output=True, text=True, check=True)
                st.success(f"Network adapter '{adapter_name}' enabled.")
            except subprocess.CalledProcessError as e:
                st.error(f"Failed to enable adapter: {e.stderr}. (Requires Administrator privileges)")
            except Exception as e:
                st.error(f"Error enabling adapter: {e}")
            else:
                st.warning("Please enter the network adapter name.")

def display_windows_tasks_detail():
    st.title(f"{st.session_state.selected_category} - {st.session_state.selected_sub_category}")

    # Back button to sub-menu
    if st.button(f"⬅️ Back to {st.session_state.selected_category} Sub-Categories", key="back_to_win_sub"):
        st.session_state.current_view = "windows_sub_menu"
        st.session_state.selected_sub_category = None # Clear sub-category when going back
        st.rerun()

    st.markdown("---")

    # Display tasks based on selected sub-category
    if st.session_state.selected_sub_category == "Camera":
        display_camera_operations_tasks() # Call the specific function for Camera
    elif st.session_state.selected_sub_category == "Messaging & Communication":
        display_messaging_communication_tasks() # Call the specific function for Messaging
    elif st.session_state.selected_sub_category == "File & Folder Operations":
        display_windows_file_folder_operations_tasks() # Updated to include advanced file ops
    elif st.session_state.selected_sub_category == "Open Applications":
        display_application_management_tasks() # Updated to include system management
    elif st.session_state.selected_sub_category == "Connectivity & Network":
        display_connectivity_network_tasks() # Call the specific function for Connectivity
    elif st.session_state.selected_sub_category == "System Power Operations":
        display_system_power_operations_tasks() # Call the specific function for Power Ops
    elif st.session_state.selected_sub_category == "System Monitoring & Info":
        display_windows_system_info_tasks() # New sub-category for system info

# Moved content of display_windows_tasks_detail into separate functions for clarity and proper routing
def display_camera_operations_tasks():
    st.subheader("Camera Operations")
    st.info("Captured photos and recorded videos will be saved to your system's Downloads folder.")

    # Get Downloads path
    downloads_path = os.path.expanduser('~/Downloads')
    if not os.path.exists(downloads_path):
        try:
            os.makedirs(downloads_path)
            st.success(f"Created Downloads directory: {downloads_path}")
        except Exception as e:
            st.error(f"Could not create Downloads directory: {e}. Please ensure it exists or create it manually.")
            downloads_path = os.getcwd() # Fallback to current working directory

    st.markdown("**Take Photo**")
    if st.button("Take Photo", key="task_take_photo"):
        if cv2:
            st.info("Capturing photo from default webcam...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not open webcam. Make sure it's not in use and drivers are installed.")
            else:
                try:
                    ret, frame = cap.read()
                    if ret:
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        photo_filename = f"captured_photo_{timestamp}.png"
                        photo_filepath = os.path.join(downloads_path, photo_filename)
                        cv2.imwrite(photo_filepath, frame)
                        st.success(f"Photo captured successfully and saved to '{photo_filepath}'.")
                        st.image(frame, channels="BGR", caption="Captured Photo")
                    else:
                        st.error("Failed to capture photo.")
                except Exception as e:
                    st.error(f"Error during photo capture: {e}")
                finally:
                    cap.release()
        else:
            st.error("OpenCV not imported. Please install `opencv-python`.")


    st.markdown("---")
    st.markdown("**Record Video**")
    record_duration = st.slider("Recording Duration (seconds)", 1, 10, 3, key="record_duration_task")
    if st.button(f"Record {record_duration} Seconds of Video", key="task_record_video"):
        if cv2:
            st.info(f"Recording video for {record_duration} seconds from default webcam...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not open webcam. Make sure it's not in use and drivers are installed.")
            else:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    video_filename = f"recorded_video_{timestamp}.avi"
                    video_filepath = os.path.join(downloads_path, video_filename)

                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    if width == 0 or height == 0:
                        width, height = 640, 480
                        st.warning("Could not detect webcam resolution, using default 640x480.")
                    out = cv2.VideoWriter(video_filepath, fourcc, 20.0, (width, height))

                    start_time = time.time()
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    while(cap.isOpened()):
                        ret, frame = cap.read()
                        if ret:
                            out.write(frame)
                            elapsed_time = time.time() - start_time
                            progress = min(1.0, elapsed_time / record_duration)
                            progress_bar.progress(progress)
                            status_text.text(f"Recording... {int(elapsed_time)}/{record_duration} seconds")
                            if elapsed_time > record_duration:
                                break
                        else:
                            break
                    cap.release()
                    out.release()
                    st.success(f"Video recorded successfully to '{video_filepath}'.")
                    status_text.empty()
                    progress_bar.empty()
                except Exception as e:
                    st.error(f"Error during video recording: {e}")
                finally:
                    if cap.isOpened(): cap.release()
                    if 'out' in locals() and out.isOpened(): out.release()
        else:
            st.error("OpenCV not imported. Please install `opencv-python`.")

def display_messaging_communication_tasks():
    st.subheader("Messaging & Communication")

    # WhatsApp Message (Functional - Server-side, requires pywhatkit setup)
    st.markdown("**WhatsApp Message **")
    whatsapp_number = st.text_input("WhatsApp Number (e.g., +919876543210)", key="whatsapp_num_task")
    whatsapp_message = st.text_area("WhatsApp Message", key="whatsapp_msg_task")
    if st.button("Send WhatsApp Message", key="task_whatsapp"):
        if whatsapp_number and whatsapp_message:
            if pywhatkit:
                try:
                    st.info("Opening WhatsApp Web in your browser to send the message. Please ensure you are logged in.")
                    pywhatkit.sendwatmsg_instantly(whatsapp_number, whatsapp_message)
                    st.success("WhatsApp message command issued. Check your browser.")
                except Exception as e:
                    st.error(f"Failed to send WhatsApp message: {e}. Make sure pywhatkit is installed and WhatsApp Web is accessible.")
            else:
                st.error("PyWhatKit not imported. Please install it with `pip install pywhatkit`.")
        else:
            st.error("Please enter a WhatsApp number and message.")

    # Text Message using Twilio 
    st.markdown("---")
    st.markdown("**Twilio SMS & Call **")
    twilio_to_number = st.text_input("Twilio Recipient Number (e.g., +1234567890)", key="twilio_to_num_task")
    twilio_message = st.text_area("Twilio Text Message", key="twilio_msg_task")
    if st.button("Send Text Message via Twilio", key="task_twilio_sms"):
        if twilio_to_number and twilio_message:
            try:
                from twilio.rest import Client
                account_sid = "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" # Your Account SID
                auth_token = "your_auth_token"                   # Your Auth Token
                twilio_phone_number = "+15017122661"             # Your Twilio phone number
                client = Client(account_sid, auth_token)
                message = client.messages.create(to=twilio_to_number, from_=twilio_phone_number, body=twilio_message)
                st.success(f"Message sent successfully! SID: {message.sid}")
            except ImportError: st.error("Twilio library not found. Please install it: `pip install twilio`")
            except Exception as e: st.error(f"Failed to send SMS via Twilio. Check credentials and network: {e}")
        else: st.error("Please enter recipient number and message.")

    # Call using Twilio (Functional - Server-side, requires Twilio setup)
    twilio_call_number = st.text_input("Twilio Call Number (e.g., +1234567890)", key="twilio_call_num_task")
    if st.button("Make Call via Twilio", key="task_twilio_call"):
        if twilio_call_number:
            try:
                from twilio.rest import Client
                account_sid = "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" # Your Account SID
                auth_token = "your_auth_token"                   # Your Auth Token
                twilio_phone_number = "+15017122661"             # Your Twilio phone number
                client = Client(account_sid, auth_token)
                call = client.calls.create(to=twilio_call_number, from_=twilio_phone_number, url="http://demo.twilio.com/docs/voice.xml")
                st.success(f"Call initiated successfully! SID: {call.sid}")
            except ImportError: st.error("Twilio library not found. Please install it: `pip install twilio`")
            except Exception as e: st.error(f"Failed to make call via Twilio. Check credentials and network: {e}")
        else: st.error("Please enter a call number.")

    # Email (Functional - Server-side, requires SMTP setup)
    st.markdown("---")
    st.markdown("**Send Email**")
    email_recipient = st.text_input("Email Recipient", key="email_rec_task")
    email_subject = st.text_input("Email Subject", key="email_sub_task")
    email_body = st.text_area("Email Body", key="email_body_task")
    email_sender = st.text_input("Your Email Address (Sender)", key="email_sender_addr_task")
    email_password = st.text_input("Your Email Password (App Password Recommended)", type="password", key="email_sender_pass_task")
    if st.button("Send Email", key="task_send_email"):
        if email_recipient and email_subject and email_body and email_sender and email_password:
            try:
                msg = MIMEText(email_body)
                msg['Subject'] = email_subject
                msg['From'] = email_sender
                msg['To'] = email_recipient
                smtp_server = "smtp.gmail.com"
                smtp_port = 465
                with smtplib.SMTP_SSL(smtp_server, smtp_port) as smtp:
                    smtp.login(email_sender, email_password)
                    smtp.send_message(msg)
                st.success("Email sent successfully!")
            except Exception as e: st.error(f"Failed to send email. Check sender credentials, SMTP settings, and network: {e}")
        else: st.error("Please fill in all email fields.")

def display_system_power_operations_tasks():
    st.subheader("System Power Operations ")
    st.error("⚠️ **These actions will immediately affect the machine ** Use with extreme caution.")

    if st.button("Shutdown", key="task_shutdown"):
        if st.checkbox("Confirm Shutdown", key="confirm_shutdown_task"):
            st.warning("Shutting down in 5 seconds...")
            time.sleep(5)
            try: os.system('shutdown /s /t 1'); st.success("Shutdown command issued.")
            except Exception as e: st.error(f"Error issuing shutdown command: {e}")
        else: st.info("Check the confirmation box to enable shutdown.")

    if st.button("Restart", key="task_restart"):
        if st.checkbox("Confirm Restart", key="confirm_restart_task"):
            st.warning("Restarting in 5 seconds...")
            time.sleep(5)
            try: os.system('shutdown /r /t 1'); st.success("Restart command issued.")
            except Exception as e: st.error(f"Error issuing restart command: {e}")
        else: st.info("Check the confirmation box to enable restart.")

    if st.button("Sleep", key="task_sleep"):
        if st.checkbox("Confirm Sleep", key="confirm_sleep_task"):
            st.warning("Attempting to put the system to sleep...")
            try: subprocess.Popen(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"]); st.success("Sleep command issued.")
            except Exception as e: st.error(f"Error issuing sleep command: {e}")
        else: st.info("Check the confirmation box to enable sleep.")


# --- Linux Task Sub-Categories and Details ---
def display_linux_system_info_tasks(host, username, password):
    st.subheader("Linux System Information")
    st.info("Retrieve basic system details from the remote Linux machine.")

    if st.button("Get Hostname"):
        output, error = execute_ssh_command(host, username, password, "hostname")
        if output: st.code(output)
        if error: st.error(error)

    if st.button("Get Kernel Version (uname -a)"):
        output, error = execute_ssh_command(host, username, password, "uname -a")
        if output: st.code(output)
        if error: st.error(error)

    if st.button("Check Disk Usage (df -h)"):
        output, error = execute_ssh_command(host, username, password, "df -h")
        if output: st.code(output)
        if error: st.error(error)

    if st.button("Check Memory Usage (free -h)"):
        output, error = execute_ssh_command(host, username, password, "free -h")
        if output: st.code(output)
        if error: st.error(error)

    if st.button("Get System Uptime (uptime)"):
        output, error = execute_ssh_command(host, username, password, "uptime")
        if output: st.code(output)
        if error: st.error(error)

    if st.button("Get OS Release Info (cat /etc/os-release)"):
        output, error = execute_ssh_command(host, username, password, "cat /etc/os-release")
        if output: st.code(output)
        if error: st.error(error)

    if st.button("List CPU Information (lscpu)"):
        output, error = execute_ssh_command(host, username, password, "lscpu")
        if output: st.code(output)
        if error: st.error(error)

    if st.button("List Block Devices (lsblk)"):
        output, error = execute_ssh_command(host, username, password, "lsblk")
        if output: st.code(output)
        if error: st.error(error)

def display_linux_file_system_tasks(host, username, password):
    st.subheader("Linux File System Management")
    st.info("Perform file and folder operations on the remote Linux machine.")

    # List Directory Contents
    ls_path = st.text_input("Path to list (ls -l)", key="ls_path", value="~")
    if st.button("List Directory Contents"):
        output, error = execute_ssh_command(host, username, password, f"ls -l {ls_path}")
        if output: st.code(output)
        if error: st.error(error)

    # Get Current Working Directory
    if st.button("Get Current Working Directory (pwd)"):
        output, error = execute_ssh_command(host, username, password, "pwd")
        if output: st.code(output)
        if error: st.error(error)

    # Create Directory
    mkdir_path = st.text_input("Directory to create (mkdir)", key="mkdir_path")
    if st.button("Create Directory"):
        if mkdir_path:
            # Added sudo for mkdir as it might be used in restricted paths
            output, error = execute_ssh_command(host, username, password, f"sudo mkdir {mkdir_path}")
            if output: st.code(output)
            if error: st.error(error)
            if not error: st.success(f"Directory '{mkdir_path}' created.")
        else: st.warning("Please enter a directory name.")

    # Remove File/Directory
    rm_path = st.text_input("File/Directory to remove (rm -rf)", key="rm_path")
    if st.button("Remove File/Directory (rm -rf)"):
        if rm_path:
            st.warning(f"This will permanently delete '{rm_path}'. Confirm to proceed. Requires `sudo`.")
            if st.checkbox(f"Confirm deletion of {rm_path}", key=f"confirm_rm_{rm_path}"):
                # Added sudo for rm -rf
                output, error = execute_ssh_command(host, username, password, f"sudo rm -rf {rm_path}")
                if output: st.code(output)
                if error: st.error(error)
                if not error: st.success(f"'{rm_path}' removed.")
        else: st.warning("Please enter a file/directory to remove.")

    # Copy File/Directory
    cp_source = st.text_input("Source path to copy (cp)", key="cp_source")
    cp_dest = st.text_input("Destination path for copy (cp)", key="cp_dest")
    if st.button("Copy File/Directory"):
        if cp_source and cp_dest:
            st.warning(f"Copying to/from system paths may require `sudo`.")
            # Added sudo for cp -r
            output, error = execute_ssh_command(host, username, password, f"sudo cp -r {cp_source} {cp_dest}")
            if output: st.code(output)
            if error: st.error(error)
            if not error: st.success(f"'{cp_source}' copied to '{cp_dest}'.")
        else: st.warning("Please enter source and destination paths.")

    # Move File/Directory
    mv_source = st.text_input("Source path to move (mv)", key="mv_source")
    mv_dest = st.text_input("Destination path for move (mv)", key="mv_dest")
    if st.button("Move File/Directory"):
        if mv_source and mv_dest:
            st.warning(f"Moving to/from system paths may require `sudo`.")
            # Added sudo for mv
            output, error = execute_ssh_command(host, username, password, f"sudo mv {mv_source} {mv_dest}")
            if output: st.code(output)
            if error: st.error(error)
            if not error: st.success(f"'{mv_source}' moved to '{mv_dest}'.")
        else: st.warning("Please enter source and destination paths.")

    # View File Content
    cat_file = st.text_input("File to view (cat)", key="cat_file")
    if st.button("View File Content (cat)"):
        if cat_file:
            st.warning("Viewing restricted files may require `sudo`.")
            # Added sudo for cat for consistency if user tries to view system logs/configs
            output, error = execute_ssh_command(host, username, password, f"sudo cat {cat_file}")
            if output: st.code(output)
            if error: st.error(error)
        else: st.warning("Please enter a file path.")

    # Create/Overwrite File Content
    echo_file = st.text_input("File to write to (echo >)", key="echo_file")
    echo_content = st.text_area("Content to write", key="echo_content")
    if st.button("Create/Overwrite File"):
        if echo_file and echo_content:
            st.warning("Writing to restricted files/paths may require `sudo`.")
            # Escape single quotes in content for shell command
            escaped_content = echo_content.replace("'", "'\\''")
            # Added sudo for echo >
            output, error = execute_ssh_command(host, username, password, f"sudo echo '{escaped_content}' > {echo_file}")
            if output: st.code(output)
            if error: st.error(error)
            if not error: st.success(f"Content written to '{echo_file}'.")
        else: st.warning("Please enter file path and content.")

    # Change File Permissions
    chmod_path = st.text_input("File/Directory for chmod", key="chmod_path")
    chmod_perms = st.text_input("Permissions (e.g., 755)", key="chmod_perms")
    if st.button("Change Permissions (chmod)"):
        if chmod_path and chmod_perms:
            st.warning("Changing permissions of system files may require `sudo`.")
            # Added sudo for chmod
            output, error = execute_ssh_command(host, username, password, f"sudo chmod {chmod_perms} {chmod_path}")
            if output: st.code(output)
            if error: st.error(error)
            if not error: st.success(f"Permissions of '{chmod_path}' changed to '{chmod_perms}'.")
        else: st.warning("Please enter path and permissions.")

    # Change File Ownership
    chown_path = st.text_input("File/Directory for chown", key="chown_path")
    chown_owner_group = st.text_input("Owner:Group (e.g., user:group)", key="chown_owner_group")
    if st.button("Change Ownership (chown)"):
        if chown_path and chown_owner_group:
            st.warning("This command requires `sudo` privileges on the remote machine. Ensure your user has `NOPASSWD` configured for `sudo` to avoid hanging.")
            output, error = execute_ssh_command(host, username, password, f"sudo chown {chown_owner_group} {chown_path}")
            if output: st.code(output)
            if error: st.error(error)
            if not error: st.success(f"Ownership of '{chown_path}' changed to '{chown_owner_group}'. (Requires sudo)")
        else: st.warning("Please enter path and owner:group.")

    # Find Files
    find_path = st.text_input("Path to search (find)", key="find_path", value="~")
    find_name = st.text_input("File name pattern (e.g., *.log)", key="find_name")
    if st.button("Find Files"):
        if find_path and find_name:
            st.warning("Searching restricted directories may require `sudo`.")
            # Added sudo for find
            output, error = execute_ssh_command(host, username, password, f"sudo find {find_path} -name '{find_name}'")
            if output: st.code(output)
            if error: st.error(error)
        else: st.warning("Please enter search path and file name pattern.")


def display_linux_process_management_tasks(host, username, password):
    st.subheader("Linux Process Management")
    st.info("Manage running processes on the remote Linux machine.")

    if st.button("List All Processes (ps aux)"):
        st.warning("Listing all processes may require `sudo` to see details for other users/root processes.")
        # Added sudo for ps aux
        output, error = execute_ssh_command(host, username, password, "sudo ps aux")
        if output: st.code(output)
        if error: st.error(error)

    if st.button("View Top Processes (top -bn1 | head -n 10)"):
        st.warning("Viewing top processes may require `sudo` to see full details.")
        # Added sudo for top
        output, error = execute_ssh_command(host, username, password, "sudo top -bn1 | head -n 10")
        if output: st.code(output)
        if error: st.error(error)

    kill_pid = st.text_input("PID to Kill", key="kill_pid")
    if st.button("Kill Process (kill -9)"):
        if kill_pid:
            st.warning(f"This will forcefully terminate PID {kill_pid}. Confirm to proceed. Requires `sudo` if not your process.")
            if st.checkbox(f"Confirm kill PID {kill_pid}", key=f"confirm_kill_{kill_pid}"):
                # Added sudo for kill
                output, error = execute_ssh_command(host, username, password, f"sudo kill -9 {kill_pid}")
                if output: st.code(output)
                if error: st.error(error)
                if not error: st.success(f"Process with PID {kill_pid} killed.")
        else: st.warning("Please enter a PID to kill.")

    service_name_status = st.text_input("Service Name (for systemctl status)", key="service_name_status")
    if st.button("Check Service Status"):
        if service_name_status:
            st.warning("Checking status for some services may require `sudo`.")
            # Added sudo for systemctl status
            output, error = execute_ssh_command(host, username, password, f"sudo systemctl status {service_name_action}")
            if output: st.code(output)
            if error: st.error(error)
        else: st.warning("Please enter a service name.")

def display_linux_network_tasks(host, username, password):
    st.subheader("Linux Network Management")
    st.info("Perform network-related tasks on the remote Linux machine.")

    if st.button("Show IP Addresses (ip a)"):
        output, error = execute_ssh_command(host, username, password, "sudo ip a")
        if output: st.code(output)
        if error: st.error(error)

    ping_target = st.text_input("Host to Ping (ping -c 4)", key="ping_target", value="google.com")
    if st.button("Ping Host"):
        if ping_target:
            output, error = execute_ssh_command(host, username, password, f"ping -c 4 {ping_target}")
            if output: st.code(output)
            if error: st.error(error)
        else: st.warning("Please enter a host to ping.")

    if st.button("List Open Ports (ss -tuln)"):
        st.warning("Listing all open ports may require `sudo` to see details for all processes.")
        output, error = execute_ssh_command(host, username, password, "sudo ss -tuln")
        if output: st.code(output)
        if error: st.error(error)

    if st.button("List Open Ports (netstat -tuln)"):
        output, error = execute_ssh_command(host, username, password, "sudo netstat -tuln")
        if output: st.code(output)
        if error: st.error(error)

    curl_url = st.text_input("URL to Curl", key="curl_url", value="http://example.com")
    if st.button("Fetch URL Content (curl)"):
        if curl_url:
            output, error = execute_ssh_command(host, username, password, f"curl -s {curl_url}")
            if output: st.code(output)
            if error: st.error(error)
        else: st.warning("Please enter a URL.")

    st.markdown("---")
    st.write("### Network Diagnostics")
    dns_host = st.text_input("Host for DNS Lookup (e.g., google.com)", key="dns_host")
    if st.button("Perform DNS Lookup (dig)"):
        if dns_host:
            output, error = execute_ssh_command(host, username, password, f"dig {dns_host}")
            if output: st.code(output)
            if error: st.error(error)
        else: st.warning("Please enter a hostname for DNS lookup.")

    if st.button("Show Active Network Connections (ss -tunlp)"):
        st.warning("Requires `sudo` to see process names/PIDs.")
        output, error = execute_ssh_command(host, username, password, "sudo ss -tunlp")
        if output: st.code(output)
        if error: st.error(error)

def display_linux_user_management_tasks(host, username, password):
    st.subheader("Linux User & Group Management")
    st.info("Manage users and groups on the remote Linux machine.")

    if st.button("Get Current User (whoami)"):
        output, error = execute_ssh_command(host, username, password, "whoami")
        if output: st.code(output)
        if error: st.error(error)

    if st.button("Get User ID and Group Info (id)"):
        output, error = execute_ssh_command(host, username, password, "id")
        if output: st.code(output)
        if error: st.error(error)

    if st.button("View /etc/passwd (User Accounts)"):
        st.warning("Viewing /etc/passwd may require `sudo` if permissions are restricted.")
        output, error = execute_ssh_command(host, username, password, "sudo cat /etc/passwd")
        if output: st.code(output)
        if error: st.error(error)

    if st.button("View /etc/group (Groups)"):
        st.warning("Viewing /etc/group may require `sudo` if permissions are restricted.")
        output, error = execute_ssh_command(host, username, password, "sudo cat /etc/group")
        if output: st.code(output)
        if error: st.error(error)

    if st.button("Check Sudo Access (sudo whoami)"):
        st.warning("This command requires `sudo` privileges on the remote machine. Ensure your user has `NOPASSWD` configured for `sudo` to avoid hanging.")
        output, error = execute_ssh_command(host, username, password, "sudo whoami")
        if output: st.code(output)
        if error: st.error(error)

    st.markdown("---")
    st.write("### User & Group Operations")
    new_username = st.text_input("New Username to Create", key="new_linux_user")
    if st.button("Create New User"):
        if new_username:
            st.warning("This requires `sudo`.")
            output, error = execute_ssh_command(host, username, password, f"sudo useradd -m {new_username}")
            if output: st.code(output)
            if error: st.error(error)
            if not error: st.success(f"User '{new_username}' created.")
        else: st.warning("Please enter a username.")

    user_to_delete = st.text_input("Username to Delete", key="del_linux_user")
    if st.button("Delete User"):
        if user_to_delete:
            st.warning("This will delete the user and their home directory. Requires `sudo`.")
            if st.checkbox(f"Confirm deletion of user {user_to_delete}", key=f"confirm_del_user_{user_to_delete}"):
                output, error = execute_ssh_command(host, username, password, f"sudo userdel -r {user_to_delete}")
                if output: st.code(output)
                if error: st.error(error)
                if not error: st.success(f"User '{user_to_delete}' deleted.")
        else: st.warning("Please enter a username.")

    user_for_password = st.text_input("User to Set Password For", key="passwd_linux_user")
    new_password = st.text_input("New Password", type="password", key="new_linux_password")
    if st.button("Set User Password"):
        if user_for_password and new_password:
            st.warning("This requires `sudo`. The password will be sent in plain text over SSH.")
            # Use chpasswd for non-interactive password setting
            output, error = execute_ssh_command(host, username, password, f"echo '{user_for_password}:{new_password}' | sudo chpasswd")
            if output: st.code(output)
            if error: st.error(error)
            if not error: st.success(f"Password set for user '{user_for_password}'.")
        else: st.warning("Please enter username and new password.")

def display_linux_package_management_tasks(host, username, password):
    st.subheader("Linux Package Management (RHEL/CentOS - dnf/yum)")
    st.info("Manage software packages on the remote Linux machine. Many operations require `sudo`.")

    if st.button("List Installed Packages (dnf list installed | head -n 20)"):
        output, error = execute_ssh_command(host, username, password, "dnf list installed | head -n 20")
        if output: st.code(output)
        if error: st.error(error)

    package_to_install = st.text_input("Package to Install (e.g., nano)", key="pkg_install")
    if st.button("Install Package (sudo dnf install -y)"):
        if package_to_install:
            st.warning(f"This will install '{package_to_install}'. Requires `sudo`. Ensure your user has `NOPASSWD` configured for `sudo` to avoid hanging.")
            output, error = execute_ssh_command(host, username, password, f"sudo dnf install -y {package_to_install}")
            if output: st.code(output)
            if error: st.error(error)
            if not error: st.success(f"Package '{package_to_install}' installation command issued.")
        else: st.warning("Please enter a package name.")

    package_to_remove = st.text_input("Package to Remove (e.g., nano)", key="pkg_remove")
    if st.button("Remove Package (sudo dnf remove -y)"):
        if package_to_remove:
            st.warning(f"This will remove '{package_to_remove}'. Confirm to proceed. Requires `sudo`. Ensure your user has `NOPASSWD` configured for `sudo` to avoid hanging.")
            if st.checkbox(f"Confirm removal of {package_to_remove}", key=f"confirm_rm_pkg_{package_to_remove}"):
                output, error = execute_ssh_command(host, username, password, f"sudo dnf remove -y {package_to_remove}")
                if output: st.code(output)
                if error: st.error(error)
                if not error: st.success(f"Package '{package_to_remove}' removal command issued.")
        else: st.warning("Please enter a package name.")

    if st.button("Update All Packages (sudo dnf update -y)"):
        st.warning("This will update all packages on the system. Requires `sudo` and can take time. Ensure your user has `NOPASSWD` configured for `sudo` to avoid hanging.")
        if st.checkbox("Confirm full system update", key="confirm_dnf_update"):
            output, error = execute_ssh_command(host, username, password, "sudo dnf update -y")
            if output: st.code(output)
            if error: st.error(error)
            if not error: st.success("System update command issued.")

    package_to_search = st.text_input("Package to Search (e.g., httpd)", key="pkg_search")
    if st.button("Search Package (dnf search)"):
        if package_to_search:
            output, error = execute_ssh_command(host, username, password, f"dnf search {package_to_search}")
            if output: st.code(output)
            if error: st.error(error)
        else: st.warning("Please enter a package name to search.")

    st.markdown("---")
    st.write("### Advanced Package Operations")
    pkg_to_list_files = st.text_input("Package to list files for (e.g., httpd)", key="pkg_list_files")
    if st.button("List Package Files (rpm -ql)"):
        if pkg_to_list_files:
            output, error = execute_ssh_command(host, username, password, f"rpm -ql {pkg_to_list_files}")
            if output: st.code(output)
            if error: st.error(error)
        else: st.warning("Please enter a package name.")

    if st.button("Check for Available Updates (dnf check-update)"):
        output, error = execute_ssh_command(host, username, password, "dnf check-update")
        if output: st.code(output)
        if error: st.error(error)

def display_linux_service_management_tasks(host, username, password):
    st.subheader("Linux Service Management (systemctl)")
    st.info("Manage system services on the remote Linux machine. Most operations require `sudo`.")

    service_name_action = st.text_input("Service Name (e.g., httpd)", key="service_name_action")

    if st.button("Start Service (sudo systemctl start)"):
        if service_name_action:
            st.warning("This command requires `sudo` privileges on the remote machine. Ensure your user has `NOPASSWD` configured for `sudo` to avoid hanging.")
            output, error = execute_ssh_command(host, username, password, f"sudo systemctl start {service_name_action}")
            if output: st.code(output)
            if error: st.error(error)
            if not error: st.success(f"Service '{service_name_action}' start command issued.")
        else: st.warning("Please enter a service name.")

    if st.button("Stop Service (sudo systemctl stop)"):
        if service_name_action:
            st.warning("This command requires `sudo` privileges on the remote machine. Ensure your user has `NOPASSWD` configured for `sudo` to avoid hanging.")
            output, error = execute_ssh_command(host, username, password, f"sudo systemctl stop {service_name_action}")
            if output: st.code(output)
            if error: st.error(error)
            if not error: st.success(f"Service '{service_name_action}' stop command issued.")
        else: st.warning("Please enter a service name.")

    if st.button("Restart Service (sudo systemctl restart)"):
        if service_name_action:
            st.warning("This command requires `sudo` privileges on the remote machine. Ensure your user has `NOPASSWD` configured for `sudo` to avoid hanging.")
            output, error = execute_ssh_command(host, username, password, f"sudo systemctl restart {service_name_action}")
            if output: st.code(output)
            if error: st.error(error)
            if not error: st.success(f"Service '{service_name_action}' restart command issued.")
        else: st.warning("Please enter a service name.")

    if st.button("Enable Service (sudo systemctl enable)"):
        if service_name_action:
            st.warning("This command requires `sudo` privileges on the remote machine. Ensure your user has `NOPASSWD` configured for `sudo` to avoid hanging.")
            output, error = execute_ssh_command(host, username, password, f"sudo systemctl enable {service_name_action}")
            if output: st.code(output)
            if error: st.error(error)
            if not error: st.success(f"Service '{service_name_action}' enable command issued.")
        else: st.warning("Please enter a service name.")

    if st.button("Disable Service (sudo systemctl disable)"):
        if service_name_action:
            st.warning("This command requires `sudo` privileges on the remote machine. Ensure your user has `NOPASSWD` configured for `sudo` to avoid hanging.")
            output, error = execute_ssh_command(host, username, password, f"sudo systemctl disable {service_name_action}")
            if output: st.code(output)
            if error: st.error(error)
            if not error: st.success(f"Service '{service_name_action}' disable command issued.")
        else: st.warning("Please enter a service name.")

def display_linux_log_management_tasks(host, username, password):
    st.subheader("Linux Log Management")

    log_file_path = st.text_input("Log File Path (e.g., /var/log/messages)", key="log_file_path", value="/var/log/messages")
    num_lines = st.slider("Number of lines to show (tail -n)", 5, 100, 20, key="num_lines_log")

    if st.button("View Last Lines of Log File"):
        if log_file_path:
            st.warning("Viewing restricted log files may require `sudo`.")
            output, error = execute_ssh_command(host, username, password, f"sudo tail -n {num_lines} {log_file_path}")
            if output: st.code(output)
            if error: st.error(error)
        else: st.warning("Please enter a log file path.")

    if st.button("View Journalctl Logs (last 20 lines)"):
        output, error = execute_ssh_command(host, username, password, "sudo journalctl -xe | tail -n 20")
        if output: st.code(output)
        if error: st.error(error)

    st.markdown("---")
    st.write("### Cron Job Management")
    cron_user = st.text_input("User for Cron Jobs (leave blank for current user)", key="cron_user")
    if st.button("List Cron Jobs"):
        cmd = f"crontab -l"
        if cron_user:
            cmd = f"sudo crontab -l -u {cron_user}"
        output, error = execute_ssh_command(host, username, password, cmd)
        if output: st.code(output)
        if error: st.error(error)

    new_cron_job = st.text_area("New Cron Job Entry (e.g., '0 0 * * * /path/to/script.sh')", key="new_cron_job")
    if st.button("Add Cron Job"):
        if new_cron_job:
            cmd = f"(crontab -l; echo '{new_cron_job}') | crontab -"
            if cron_user:
                cmd = f"sudo bash -c \"(crontab -l -u {cron_user} 2>/dev/null; echo '{new_cron_job}') | crontab -u {cron_user} -\""
            st.warning("Adding cron jobs requires careful syntax. Ensure your entry is correct.")
            output, error = execute_ssh_command(host, username, password, cmd)
            if output: st.code(output)
            if error: st.error(error)
            if not error: st.success("Cron job added.")
        else: st.warning("Please enter a cron job entry.")

def display_linux_firewall_tasks(host, username, password):
    st.subheader("Linux Firewall Management (firewalld)")

    if st.button("List All Firewall Rules (firewall-cmd --list-all)"):
        output, error = execute_ssh_command(host, username, password, "sudo firewall-cmd --list-all")
        if output: st.code(output)
        if error: st.error(error)

    port_to_manage = st.text_input("Port (e.g., 8080)", key="fw_port")
    protocol = st.selectbox("Protocol", ["tcp", "udp"], key="fw_protocol")
    zone = st.text_input("Zone (e.g., public)", value="public", key="fw_zone")

    col_fw1, col_fw2 = st.columns(2)
    with col_fw1:
        if st.button("Open Port (Permanent)", key="open_port_perm"):
            if port_to_manage:
                cmd = f"sudo firewall-cmd --zone={zone} --add-port={port_to_manage}/{protocol} --permanent && sudo firewall-cmd --reload"
                output, error = execute_ssh_command(host, username, password, cmd)
                if output: st.code(output)
                if error: st.error(error)
                if not error: st.success(f"Port {port_to_manage}/{protocol} opened permanently in zone {zone}.")
            else: st.warning("Please enter a port.")
    with col_fw2:
        if st.button("Close Port (Permanent)", key="close_port_perm"):
            if port_to_manage:
                cmd = f"sudo firewall-cmd --zone={zone} --remove-port={port_to_manage}/{protocol} --permanent && sudo firewall-cmd --reload"
                output, error = execute_ssh_command(host, username, password, cmd)
                if output: st.code(output)
                if error: st.error(error)
                if not error: st.success(f"Port {port_to_manage}/{protocol} closed permanently in zone {zone}.")
            else: st.warning("Please enter a port.")

def display_linux_ssh_key_management_tasks(host, username, password):
    st.subheader("Linux SSH Key Management")

    if st.button("View Public Key (cat ~/.ssh/id_rsa.pub)"):
        output, error = execute_ssh_command(host, username, password, "cat ~/.ssh/id_rsa.pub")
        if output: st.code(output)
        if error: st.error(error)

    public_key_to_add = st.text_area("Public Key to Add to authorized_keys", key="public_key_to_add")
    if st.button("Add Public Key to authorized_keys"):
        if public_key_to_add:
            st.warning("This will add the provided public key to the current user's `~/.ssh/authorized_keys` file. Requires correct permissions.")
            # Ensure .ssh directory and authorized_keys file exist and have correct permissions
            cmd = f"mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo '{public_key_to_add}' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
            output, error = execute_ssh_command(host, username, password, cmd)
            if output: st.code(output)
            if error: st.error(error)
            if not error: st.success("Public key added to authorized_keys.")
        else: st.warning("Please paste a public key.")


def display_linux_tasks_detail():
    st.title(f"{st.session_state.selected_category} - {st.session_state.selected_sub_category}")

    # Back button to sub-menu
    if st.button(f"⬅️ Back to {st.session_state.selected_category} Sub-Categories", key="back_to_linux_sub"):
        st.session_state.current_view = "linux_sub_menu"
        st.session_state.selected_sub_category = None # Clear sub-category when going back
        st.rerun()

    st.markdown("---")

    # Ensure SSH credentials are available and connected
    if not st.session_state.get('ssh_connected', False):
        st.error("Please connect to the Linux machine first by entering SSH credentials on the previous page.")
        return

    host = st.session_state.ssh_host
    username = st.session_state.ssh_username
    password = st.session_state.ssh_password

    # Display tasks based on selected sub-category
    if st.session_state.selected_sub_category == "System Information":
        display_linux_system_info_tasks(host, username, password)
    elif st.session_state.selected_sub_category == "File System Management":
        display_linux_file_system_tasks(host, username, password)
    elif st.session_state.selected_sub_category == "Process Management":
        display_linux_process_management_tasks(host, username, password)
    elif st.session_state.selected_sub_category == "Network Management":
        display_linux_network_tasks(host, username, password)
    elif st.session_state.selected_sub_category == "User & Group Management":
        display_linux_user_management_tasks(host, username, password)
    elif st.session_state.selected_sub_category == "Package Management":
        display_linux_package_management_tasks(host, username, password)
    elif st.session_state.selected_sub_category == "Service Management":
        display_linux_service_management_tasks(host, username, password)
    elif st.session_state.selected_sub_category == "Log & Cron Management":
        display_linux_log_management_tasks(host, username, password)
    elif st.session_state.selected_sub_category == "Firewall Management":
        display_linux_firewall_tasks(host, username, password)
    elif st.session_state.selected_sub_category == "SSH Key Management":
        display_linux_ssh_key_management_tasks(host, username, password)


# --- Docker Task Functions ---
def display_docker_container_management_tasks(host, username, password):
    st.subheader("Docker Container Management")
    st.info("Manage Docker containers on the remote Linux machine. Requires Docker to be installed and running.")

    if st.button("List Running Containers (docker ps)"):
        output, error = execute_ssh_command(host, username, password, "sudo docker ps")
        if output: st.code(output)
        if error: st.error(error)

    if st.button("List All Containers (docker ps -a)"):
        output, error = execute_ssh_command(host, username, password, "sudo docker ps -a")
        if output: st.code(output)
        if error: st.error(error)

    container_name_id = st.text_input("Container Name/ID", key="docker_container_name_id")

    if st.button("Start Container"):
        if container_name_id:
            output, error = execute_ssh_command(host, username, password, f"sudo docker start {container_name_id}")
            if output: st.success(f"Container '{container_name_id}' started.")
            if error: st.error(error)
        else: st.warning("Please enter a container name or ID.")

    if st.button("Stop Container"):
        if container_name_id:
            output, error = execute_ssh_command(host, username, password, f"sudo docker stop {container_name_id}")
            if output: st.success(f"Container '{container_name_id}' stopped.")
            if error: st.error(error)
        else: st.warning("Please enter a container name or ID.")

    if st.button("Restart Container"):
        if container_name_id:
            output, error = execute_ssh_command(host, username, password, f"sudo docker restart {container_name_id}")
            if output: st.success(f"Container '{container_name_id}' restarted.")
            if error: st.error(error)
        else: st.warning("Please enter a container name or ID.")
    
    if st.button("Remove Container (docker rm)"):
        if container_name_id:
            st.warning(f"This will permanently remove container '{container_name_id}'. Confirm to proceed.")
            if st.checkbox(f"Confirm removal of container {container_name_id}", key=f"confirm_rm_container_{container_name_id}"):
                output, error = execute_ssh_command(host, username, password, f"sudo docker rm {container_name_id}")
                if output: st.success(f"Container '{container_name_id}' removed.")
                if error: st.error(error)
        else: st.warning("Please enter a container name or ID.")

    if st.button("View Container Logs (docker logs)"):
        if container_name_id:
            output, error = execute_ssh_command(host, username, password, f"sudo docker logs {container_name_id}")
            if output: st.code(output)
            if error: st.error(error)
        else: st.warning("Please enter a container name or ID.")

    run_image_name = st.text_input("Image to Run (e.g., nginx:latest)", key="docker_run_image")
    run_container_name = st.text_input("New Container Name (optional)", key="docker_new_container_name")
    run_ports = st.text_input("Ports (e.g., -p 80:80)", key="docker_run_ports")
    run_options = st.text_input("Other Options (e.g., -d --name myweb)", key="docker_run_options")
    if st.button("Run New Container"):
        if run_image_name:
            cmd = f"sudo docker run {run_ports} {run_options} {run_image_name}"
            if run_container_name:
                cmd = f"sudo docker run {run_ports} {run_options} --name {run_container_name} {run_image_name}"
            output, error = execute_ssh_command(host, username, password, cmd)
            if output: st.success(f"Container from '{run_image_name}' initiated.")
            if error: st.error(error)
        else: st.warning("Please enter an image name to run.")
    
    exec_container_name_id = st.text_input("Container for Exec", key="docker_exec_container_name_id")
    exec_command = st.text_input("Command to Exec (e.g., ls -l /app)", key="docker_exec_command")
    if st.button("Execute Command in Container"):
        if exec_container_name_id and exec_command:
            output, error = execute_ssh_command(host, username, password, f"sudo docker exec {exec_container_name_id} {exec_command}")
            if output: st.code(output)
            if error: st.error(error)
        else: st.warning("Please enter container name/ID and command.")

    if st.button("Inspect Container"):
        if container_name_id:
            output, error = execute_ssh_command(host, username, password, f"sudo docker inspect {container_name_id}")
            if output: st.json(output)
            if error: st.error(error)
        else: st.warning("Please enter a container name or ID.")

    st.markdown("---")
    st.write("### Advanced Container Operations")
    container_stats_name = st.text_input("Container Name/ID for Stats", key="container_stats_name")
    if st.button("View Container Resource Stats (docker stats)"):
        if container_stats_name:
            output, error = execute_ssh_command(host, username, password, f"sudo docker stats --no-stream {container_stats_name}")
            if output: st.code(output)
            if error: st.error(error)
        else: st.warning("Please enter a container name or ID.")

def display_docker_image_management_tasks(host, username, password):
    st.subheader("Docker Image Management")

    if st.button("List Images (docker images)"):
        output, error = execute_ssh_command(host, username, password, "sudo docker images")
        if output: st.code(output)
        if error: st.error(error)

    pull_image_name = st.text_input("Image to Pull (e.g., ubuntu:latest)", key="docker_pull_image")
    if st.button("Pull Image"):
        if pull_image_name:
            output, error = execute_ssh_command(host, username, password, f"sudo docker pull {pull_image_name}")
            if output: st.success(f"Image '{pull_image_name}' pulled.")
            if error: st.error(error)
        else: st.warning("Please enter an image name to pull.")

    remove_image_name_id = st.text_input("Image Name/ID to Remove", key="docker_remove_image")
    if st.button("Remove Image (docker rmi)"):
        if remove_image_name_id:
            st.warning(f"This will remove image '{remove_image_name_id}'. Confirm to proceed.")
            if st.checkbox(f"Confirm removal of image {remove_image_name_id}", key=f"confirm_rm_image_{remove_image_name_id}"):
                output, error = execute_ssh_command(host, username, password, f"sudo docker rmi {remove_image_name_id}")
                if output: st.success(f"Image '{remove_image_name_id}' removed.")
                if error: st.error(error)
        else: st.warning("Please enter an image name or ID.")

    inspect_image_name_id = st.text_input("Image Name/ID to Inspect", key="docker_inspect_image")
    if st.button("Inspect Image"):
        if inspect_image_name_id:
            output, error = execute_ssh_command(host, username, password, f"sudo docker inspect {inspect_image_name_id}")
            if output: st.json(output)
            if error: st.error(error)
        else: st.warning("Please enter an image name or ID.")

    tag_source_image = st.text_input("Source Image (Name:Tag)", key="docker_tag_source")
    tag_target_image = st.text_input("Target Image (NewName:NewTag)", key="docker_tag_target")
    if st.button("Tag Image"):
        if tag_source_image and tag_target_image:
            output, error = execute_ssh_command(host, username, password, f"sudo docker tag {tag_source_image} {tag_target_image}")
            if output: st.success(f"Image '{tag_source_image}' tagged as '{tag_target_image}'.")
            if error: st.error(error)
        else: st.warning("Please enter source and target image names.")

    st.markdown("---")
    st.write("### Image Building")
    dockerfile_content = st.text_area("Dockerfile Content", height=200, key="dockerfile_content")
    image_name_to_build = st.text_input("New Image Name (e.g., myapp:latest)", key="image_name_to_build")
    if st.button("Build Image from Dockerfile"):
        if dockerfile_content and image_name_to_build:
            st.warning("Building images requires creating a temporary file on the remote server. Ensure the user has write permissions in the target directory.")
            # Create a temporary directory and Dockerfile content on the remote machine
            temp_dir = f"/tmp/docker_build_{int(time.time())}"
            dockerfile_path = f"{temp_dir}/Dockerfile"
            
            # Create directory
            output, error = execute_ssh_command(host, username, password, f"mkdir -p {temp_dir}")
            if error: st.error(f"Error creating temp dir: {error}"); return

            # Write Dockerfile content
            escaped_dockerfile_content = dockerfile_content.replace("'", "'\\''")
            output, error = execute_ssh_command(host, username, password, f"echo '{escaped_dockerfile_content}' > {dockerfile_path}")
            if error: st.error(f"Error writing Dockerfile: {error}"); return

            # Build image
            build_cmd = f"sudo docker build -t {image_name_to_build} {temp_dir}"
            output, error = execute_ssh_command(host, username, password, build_cmd)
            if output: st.code(output)
            if error: st.error(error)
            if not error: st.success(f"Image '{image_name_to_build}' built successfully.")

            # Clean up temporary directory (optional but good practice)
            execute_ssh_command(host, username, password, f"rm -rf {temp_dir}")
        else: st.warning("Please provide Dockerfile content and a new image name.")

    st.markdown("---")
    st.write("### Registry Operations")
    registry_image_name = st.text_input("Image to Push/Pull (e.g., myregistry/myimage:tag)", key="registry_image_name")
    registry_username = st.text_input("Registry Username (optional)", key="registry_username")
    registry_password = st.text_input("Registry Password (optional)", type="password", key="registry_password")

    col_reg1, col_reg2 = st.columns(2)
    with col_reg1:
        if st.button("Push Image to Registry"):
            if registry_image_name:
                login_cmd = ""
                if registry_username and registry_password:
                    login_cmd = f"echo '{registry_password}' | sudo docker login --username {registry_username} --password-stdin && "
                cmd = f"{login_cmd}sudo docker push {registry_image_name}"
                output, error = execute_ssh_command(host, username, password, cmd)
                if output: st.code(output)
                if error: st.error(error)
                if not error: st.success(f"Image '{registry_image_name}' pushed.")
            else: st.warning("Please enter an image name for the registry.")
    with col_reg2:
        if st.button("Pull Image from Registry"):
            if registry_image_name:
                login_cmd = ""
                if registry_username and registry_password:
                    login_cmd = f"echo '{registry_password}' | sudo docker login --username {registry_username} --password-stdin && "
                cmd = f"{login_cmd}sudo docker pull {registry_image_name}"
                output, error = execute_ssh_command(host, username, password, cmd)
                if output: st.code(output)
                if error: st.error(error)
                if not error: st.success(f"Image '{registry_image_name}' pulled.")
            else: st.warning("Please enter an image name for the registry.")


def display_docker_network_management_tasks(host, username, password):
    st.subheader("Docker Network Management")
    st.info("Manage Docker networks on the remote Linux machine.")

    if st.button("List Networks (docker network ls)"):
        output, error = execute_ssh_command(host, username, password, "sudo docker network ls")
        if output: st.code(output)
        if error: st.error(error)

    create_network_name = st.text_input("Network Name to Create", key="docker_create_network")
    if st.button("Create Network"):
        if create_network_name:
            output, error = execute_ssh_command(host, username, password, f"sudo docker network create {create_network_name}")
            if output: st.success(f"Network '{create_network_name}' created.")
            if error: st.error(error)
        else: st.warning("Please enter a network name.")

    remove_network_name = st.text_input("Network Name to Remove", key="docker_remove_network")
    if st.button("Remove Network"):
        if remove_network_name:
            st.warning(f"This will remove network '{remove_network_name}'. Confirm to proceed.")
            if st.checkbox(f"Confirm removal of network {remove_network_name}", key=f"confirm_rm_network_{remove_network_name}"):
                output, error = execute_ssh_command(host, username, password, f"sudo docker network rm {remove_network_name}")
                if output: st.success(f"Network '{remove_network_name}' removed.")
                if error: st.error(error)
        else: st.warning("Please enter a network name.")

    inspect_network_name = st.text_input("Network Name to Inspect", key="docker_inspect_network")
    if st.button("Inspect Network"):
        if inspect_network_name:
            output, error = execute_ssh_command(host, username, password, f"sudo docker inspect {inspect_network_name}")
            if output: st.json(output)
            if error: st.error(error)
        else: st.warning("Please enter a network name.")

def display_docker_volume_management_tasks(host, username, password):
    st.subheader("Docker Volume Management")
    st.info("Manage Docker volumes on the remote Linux machine.")

    if st.button("List Volumes (docker volume ls)"):
        output, error = execute_ssh_command(host, username, password, "sudo docker volume ls")
        if output: st.code(output)
        if error: st.error(error)

    create_volume_name = st.text_input("Volume Name to Create", key="docker_create_volume")
    if st.button("Create Volume"):
        if create_volume_name:
            output, error = execute_ssh_command(host, username, password, f"sudo docker volume create {create_volume_name}")
            if output: st.success(f"Volume '{create_volume_name}' created.")
            if error: st.error(error)
        else: st.warning("Please enter a volume name.")

    remove_volume_name = st.text_input("Volume Name to Remove", key="docker_remove_volume")
    if st.button("Remove Volume"):
        if remove_volume_name:
            st.warning(f"This will remove volume '{remove_volume_name}'. Confirm to proceed.")
            if st.checkbox(f"Confirm removal of volume {remove_volume_name}", key=f"confirm_rm_volume_{remove_volume_name}"):
                output, error = execute_ssh_command(host, username, password, f"sudo docker volume rm {remove_volume_name}")
                if output: st.success(f"Volume '{remove_volume_name}' removed.")
                if error: st.error(error)
        else: st.warning("Please enter a volume name.")

    inspect_volume_name = st.text_input("Volume Name to Inspect", key="docker_inspect_volume")
    if st.button("Inspect Volume"):
        if inspect_volume_name:
            output, error = execute_ssh_command(host, username, password, f"sudo docker inspect {inspect_volume_name}")
            if output: st.json(output)
            if error: st.error(error)
        else: st.warning("Please enter a volume name.")

def display_docker_system_tasks(host, username, password):
    st.subheader("Docker System Information & Cleanup")
    st.info("Get Docker system-wide information and perform cleanup operations.")

    if st.button("Show Docker Info (docker info)"):
        output, error = execute_ssh_command(host, username, password, "sudo docker info")
        if output: st.code(output)
        if error: st.error(error)

    if st.button("Show Docker Version (docker version)"):
        output, error = execute_ssh_command(host, username, password, "sudo docker version")
        if output: st.code(output)
        if error: st.error(error)

    if st.button("Show Disk Usage (docker system df)"):
        output, error = execute_ssh_command(host, username, password, "sudo docker system df")
        if output: st.code(output)
        if error: st.error(error)

    st.markdown("---")
    st.write("### Docker Cleanup")
    if st.button("Prune System (docker system prune -f)"):
        st.warning("This will remove all stopped containers, dangling images, unused networks, and build cache. Confirm to proceed.")
        if st.checkbox("Confirm Docker system prune", key="confirm_docker_prune"):
            output, error = execute_ssh_command(host, username, password, "sudo docker system prune -f")
            if output: st.success("Docker system pruned.")
            if error: st.error(error)
    
    if st.button("Prune Containers (docker container prune -f)"):
        st.warning("This will remove all stopped containers. Confirm to proceed.")
        if st.checkbox("Confirm Docker container prune", key="confirm_docker_container_prune"):
            output, error = execute_ssh_command(host, username, password, "sudo docker container prune -f")
            if output: st.success("Docker containers pruned.")
            if error: st.error(error)

    if st.button("Prune Images (docker image prune -f)"):
        st.warning("This will remove all dangling images. Confirm to proceed.")
        if st.checkbox("Confirm Docker image prune", key="confirm_docker_image_prune"):
            output, error = execute_ssh_command(host, username, password, "sudo docker image prune -f")
            if output: st.success("Docker images pruned.")
            if error: st.error(error)

    if st.button("Prune Volumes (docker volume prune -f)"):
        st.warning("This will remove all unused local volumes. Confirm to proceed.")
        if st.checkbox("Confirm Docker volume prune", key="confirm_docker_volume_prune"):
            output, error = execute_ssh_command(host, username, password, "sudo docker volume prune -f")
            if output: st.success("Docker volumes pruned.")
            if error: st.error(error)

    if st.button("Prune Networks (docker network prune -f)"):
        st.warning("This will remove all unused networks. Confirm to proceed.")
        if st.checkbox("Confirm Docker network prune", key="confirm_docker_network_prune"):
            output, error = execute_ssh_command(host, username, password, "sudo docker network prune -f")
            if output: st.success("Docker networks pruned.")
            if error: st.error(error)

def display_docker_compose_tasks(host, username, password):
    st.subheader("Docker Compose Management")
    st.info("Manage Docker Compose applications on the remote Linux machine. Requires Docker Compose installed.")

    docker_compose_content = st.text_area("docker-compose.yml Content", height=300, key="docker_compose_content")
    compose_project_path = st.text_input("Project Path (e.g., /opt/my_app)", key="compose_project_path", value="/tmp/docker_compose_project")

    if st.button("Deploy Docker Compose (Up)"):
        if docker_compose_content and compose_project_path:
            st.warning("This will create a temporary directory and `docker-compose.yml` on the remote server.")
            # Create temporary directory and write docker-compose.yml
            compose_file_path = os.path.join(compose_project_path, "docker-compose.yml")
            
            # Create directory
            output, error = execute_ssh_command(host, username, password, f"mkdir -p {compose_project_path}")
            if error: st.error(f"Error creating project dir: {error}"); return

            # Write docker-compose.yml content
            escaped_compose_content = docker_compose_content.replace("'", "'\\''")
            output, error = execute_ssh_command(host, username, password, f"echo '{escaped_compose_content}' > {compose_file_path}")
            if error: st.error(f"Error writing docker-compose.yml: {error}"); return

            # Run docker-compose up
            cmd = f"cd {compose_project_path} && sudo docker-compose up -d"
            output, error = execute_ssh_command(host, username, password, cmd)
            if output: st.code(output)
            if error: st.error(error)
            if not error: st.success("Docker Compose application deployed.")
        else: st.warning("Please provide docker-compose.yml content and a project path.")

    if st.button("Stop Docker Compose (Down)"):
        if compose_project_path:
            cmd = f"cd {compose_project_path} && sudo docker-compose down"
            output, error = execute_ssh_command(host, username, password, cmd)
            if output: st.code(output)
            if error: st.error(error)
            if not error: st.success("Docker Compose application stopped and removed.")
        else: st.warning("Please enter the project path.")

    if st.button("List Docker Compose Services (docker-compose ps)"):
        if compose_project_path:
            cmd = f"cd {compose_project_path} && sudo docker-compose ps"
            output, error = execute_ssh_command(host, username, password, cmd)
            if output: st.code(output)
            if error: st.error(error)
        else: st.warning("Please enter the project path.")

def display_docker_swarm_tasks(host, username, password):
    st.subheader("Docker Swarm Management")
    st.info("Manage Docker Swarm on the remote Linux machine. Requires Docker Swarm enabled.")

    if st.button("Initialize Docker Swarm"):
        st.warning("This will initialize a new Docker Swarm on the host. Only run on one manager node.")
        if st.checkbox("Confirm Swarm Initialization", key="confirm_swarm_init"):
            output, error = execute_ssh_command(host, username, password, "sudo docker swarm init")
            if output: st.code(output)
            if error: st.error(error)
            if not error: st.success("Docker Swarm initialized.")

    swarm_join_token = st.text_input("Swarm Join Token (e.g., from 'docker swarm join-token worker')", key="swarm_join_token")
    swarm_manager_ip = st.text_input("Swarm Manager IP:Port (e.g., 192.168.1.100:2377)", key="swarm_manager_ip")
    if st.button("Join Docker Swarm"):
        if swarm_join_token and swarm_manager_ip:
            st.warning("This node will join the specified Docker Swarm.")
            cmd = f"sudo docker swarm join --token {swarm_join_token} {swarm_manager_ip}"
            output, error = execute_ssh_command(host, username, password, cmd)
            if output: st.code(output)
            if error: st.error(error)
            if not error: st.success("Node joined Docker Swarm.")
        else: st.warning("Please provide join token and manager IP.")

    if st.button("List Swarm Nodes (docker node ls)"):
        output, error = execute_ssh_command(host, username, password, "sudo docker node ls")
        if output: st.code(output)
        if error: st.error(error)

def display_docker_tasks_detail():
    st.title(f"{st.session_state.selected_category} - {st.session_state.selected_sub_category}")

    # Back button to sub-menu
    if st.button(f"⬅️ Back to {st.session_state.selected_category} Sub-Categories", key="back_to_docker_sub"):
        st.session_state.current_view = "docker_sub_menu"
        st.session_state.selected_sub_category = None # Clear sub-category when going back
        st.rerun()

    st.markdown("---")

    # Ensure SSH credentials are available and connected for Docker tasks
    if not st.session_state.get('ssh_connected', False):
        st.error("Please connect to the Linux machine first by entering SSH credentials on the Docker sub-menu page.")
        return

    host = st.session_state.ssh_host
    username = st.session_state.ssh_username
    password = st.session_state.ssh_password

    # Display tasks based on selected sub-category
    if st.session_state.selected_sub_category == "Container Management":
        display_docker_container_management_tasks(host, username, password)
    elif st.session_state.selected_sub_category == "Image Management":
        display_docker_image_management_tasks(host, username, password)
    elif st.session_state.selected_sub_category == "Network Management":
        display_docker_network_management_tasks(host, username, password)
    elif st.session_state.selected_sub_category == "Volume Management":
        display_docker_volume_management_tasks(host, username, password)
    elif st.session_state.selected_sub_category == "System & Info":
        display_docker_system_tasks(host, username, password)
    elif st.session_state.selected_sub_category == "Docker Compose":
        display_docker_compose_tasks(host, username, password)
    elif st.session_state.selected_sub_category == "Docker Swarm":
        display_docker_swarm_tasks(host, username, password)


# --- ML Task Functions ---

def display_ml_upload_filter_tasks():
    st.title("📁 Data Upload & Preprocessing")

    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.session_state.filtered_df = df.copy()

        st.subheader("🔍 Data Preview")
        st.dataframe(df.head())

        st.subheader("🔧 Preprocessing Options")
        col1, col2 = st.columns(2)
        with col1:
            imputation_strategy = st.selectbox(
                "Missing Value Imputation Strategy",
                ["Median (Numeric)", "Most Frequent (Categorical)", "Mean (Numeric)"],
                key="imputation_strategy"
            )
        with col2:
            scaling_strategy = st.selectbox(
                "Feature Scaling Strategy (Numeric)",
                ["None", "StandardScaler", "MinMaxScaler"],
                key="scaling_strategy"
            )
        
        one_hot_encode_cols = st.multiselect(
            "Select columns for One-Hot Encoding (Categorical)",
            [col for col in df.columns if df[col].dtype == 'object' or df[col].dtype.name == 'category'],
            key="ohe_cols"
        )

        if st.button("Apply Preprocessing"):
            processed_df = df.copy()
            numeric_cols = processed_df.select_dtypes(include=np.number).columns
            categorical_cols = processed_df.select_dtypes(include=['object', 'category']).columns

            # Imputation
            for col in processed_df.columns:
                if processed_df[col].isnull().any():
                    if processed_df[col].dtype == 'object' or processed_df[col].dtype.name == 'category':
                        imputer = SimpleImputer(strategy='most_frequent')
                        processed_df[col] = imputer.fit_transform(processed_df[[col]])
                    elif processed_df[col].dtype == 'int64' or processed_df[col].dtype == 'float64':
                        if imputation_strategy == "Median (Numeric)":
                            imputer = SimpleImputer(strategy='median')
                        elif imputation_strategy == "Mean (Numeric)":
                            imputer = SimpleImputer(strategy='mean')
                        else: # Most Frequent (Numeric)
                            imputer = SimpleImputer(strategy='most_frequent')
                        processed_df[col] = imputer.fit_transform(processed_df[[col]])
            st.success("Missing values imputed.")

            # One-Hot Encoding
            if one_hot_encode_cols:
                # Identify columns that are truly categorical and not already numeric after LabelEncoding
                cols_to_ohe = [col for col in one_hot_encode_cols if col in categorical_cols and col not in numeric_cols]
                if cols_to_ohe:
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('cat', OneHotEncoder(handle_unknown='ignore'), cols_to_ohe)
                        ],
                        remainder='passthrough'
                    )
                    # Fit and transform, then convert back to DataFrame
                    transformed_array = preprocessor.fit_transform(processed_df)
                    # Get new column names
                    ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cols_to_ohe)
                    remaining_cols = [col for col in processed_df.columns if col not in cols_to_ohe]
                    new_columns = list(ohe_feature_names) + remaining_cols
                    processed_df = pd.DataFrame(transformed_array, columns=new_columns, index=processed_df.index)
                    st.success(f"Columns {', '.join(cols_to_ohe)} One-Hot Encoded.")
                else:
                    st.info("No suitable categorical columns found for One-Hot Encoding among selected.")

            # Label Encoding for remaining categorical columns not OHE'd
            for col in processed_df.columns:
                if (processed_df[col].dtype == 'object' or processed_df[col].dtype.name == 'category') and col not in one_hot_encode_cols:
                    processed_df[col] = LabelEncoder().fit_transform(processed_df[col])
                    st.info(f"Column '{col}' Label Encoded.")

            # Feature Scaling
            if scaling_strategy != "None":
                numeric_cols_after_ohe = processed_df.select_dtypes(include=np.number).columns.tolist()
                if scaling_strategy == "StandardScaler":
                    scaler = StandardScaler()
                elif scaling_strategy == "MinMaxScaler":
                    scaler = MinMaxScaler()
                
                if numeric_cols_after_ohe:
                    processed_df[numeric_cols_after_ohe] = scaler.fit_transform(processed_df[numeric_cols_after_ohe])
                    st.success(f"Numeric columns scaled using {scaling_strategy}.")
                else:
                    st.warning("No numeric columns found for scaling after imputation and encoding.")

            st.session_state.filtered_df = processed_df
            st.success("Preprocessing complete!")
            st.dataframe(processed_df.head())

        st.subheader("🔧 Multi-Column Filters")
        filter_cols = st.multiselect("Select columns to filter", df.columns.tolist())
        filter_values = {}

        filtered_df = df.copy()
        for col in filter_cols:
            options = df[col].dropna().unique().tolist()
            if options: # Ensure options are not empty
                default_index = 0
                if f"filter_{col}" in st.session_state and st.session_state[f"filter_{col}"] in options:
                    default_index = options.index(st.session_state[f"filter_{col}"])
                val = st.selectbox(f"Filter {col}", options, index=default_index, key=f"filter_{col}")
                filter_values[col] = val
                filtered_df = filtered_df[filtered_df[col] == val]
            else:
                st.warning(f"No unique values found for filtering in column '{col}'.")


        st.session_state.filtered_df = filtered_df

        st.success(f"✅ Filtered rows: {len(filtered_df)}")
        st.dataframe(filtered_df)

        with st.expander("📊 Data Summary", expanded=True):
            st.write("Shape:", filtered_df.shape)
            st.write("📊 Summary Statistics")
            st.dataframe(filtered_df.describe())
            st.write("Missing Values:")
            st.dataframe(filtered_df.isnull().sum().rename("Missing Count"))

        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Download Filtered CSV", data=csv, file_name="filtered_data.csv", mime="text/csv")

    else:
        st.info("Upload a CSV file to start.")

def display_ml_visualize_tasks():
    st.title("📊 Visualize Data")

    df = st.session_state.filtered_df if st.session_state.filtered_df is not None else st.session_state.df

    if df is None:
        st.warning("Please upload and filter your data first on 'Data Upload & Preprocessing' page.")
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        chart_type = st.selectbox("Select Chart Type", ["Line", "Bar", "Area", "Pie", "Histogram", "Scatter"])

        # Helper function for plotting and downloading charts (using Plotly for interactivity)
        def plot_and_download_chart_plotly(df_to_plot, chart_type, x_col=None, y_col=None, pie_col=None, hist_col=None, bins=None):
            fig = None
            if chart_type == "Scatter":
                fig = px.scatter(df_to_plot, x=x_col, y=y_col, title=f"Scatter Plot: {x_col} vs {y_col}")
            elif chart_type == "Bar":
                fig = px.bar(df_to_plot, x=x_col, y=y_col, title=f"Bar Chart: {x_col} vs {y_col}")
            elif chart_type == "Line":
                fig = px.line(df_to_plot, x=x_col, y=y_col, title=f"Line Plot: {x_col} vs {y_col}")
            elif chart_type == "Pie":
                if pie_col and not df_to_plot[pie_col].empty:
                    counts = df_to_plot[pie_col].value_counts().reset_index()
                    counts.columns = [pie_col, 'count']
                    fig = px.pie(counts, values='count', names=pie_col, title=f"Pie Chart of {pie_col}")
                else:
                    st.error("No data or column selected for Pie Chart.")
                    return
            elif chart_type == "Histogram":
                if hist_col and not df_to_plot[hist_col].empty:
                    fig = px.histogram(df_to_plot, x=hist_col, nbins=bins, title=f"Histogram of {hist_col}")
                else:
                    st.error("No data or column selected for Histogram.")
                    return
            elif chart_type == "Area":
                fig = px.area(df_to_plot, x=x_col, y=y_col, title=f"Area Chart: {x_col} vs {y_col}")
            else:
                st.error("Unsupported chart type!")
                return

            if fig:
                st.plotly_chart(fig, use_container_width=True)

                # Download button for Plotly figures (can save as HTML or static image)
                # For simplicity, we'll offer HTML which is interactive.
                html_export = fig.to_html(full_html=False, include_plotlyjs='cdn')
                st.download_button(
                    label=f"📥 Download {chart_type} Chart as HTML",
                    data=html_export,
                    file_name=f"{chart_type.lower()}_chart.html",
                    mime="text/html"
                )


        if chart_type in ["Line", "Bar", "Area", "Scatter"]:
            x_col = st.selectbox("X-axis", numeric_cols, key="x_col_viz")
            y_col = st.selectbox("Y-axis", numeric_cols, key="y_col_viz")
            if st.button("Generate Chart", key="generate_xy_chart"):
                if x_col and y_col:
                    plot_and_download_chart_plotly(df, chart_type, x_col=x_col, y_col=y_col)
                else:
                    st.warning("Please select both X and Y axes.")

        elif chart_type == "Pie":
            if categorical_cols:
                pie_col = st.selectbox("Select categorical column", categorical_cols, key="pie_col_viz")
                if st.button("Generate Pie Chart", key="generate_pie_chart"):
                    if pie_col:
                        plot_and_download_chart_plotly(df, chart_type, pie_col=pie_col)
                    else:
                        st.warning("Please select a categorical column.")
            else:
                st.warning("No categorical columns available for Pie Chart.")

        elif chart_type == "Histogram":
            if numeric_cols:
                hist_col = st.selectbox("Select numeric column", numeric_cols, key="hist_col_viz")
                bins = st.slider("Bins", 5, 50, 10, key="hist_bins_viz")
                if st.button("Generate Histogram", key="generate_hist_chart"):
                    if hist_col:
                        plot_and_download_chart_plotly(df, chart_type, hist_col=hist_col, bins=bins)
                    else:
                        st.warning("Please select a numeric column.")
            else:
                st.warning("No numeric columns available for Histogram.")

def display_ml_trainer_tasks():
    st.title("🧠 Smart ML Trainer")

    df = st.session_state.filtered_df if st.session_state.filtered_df is not None else st.session_state.df

    if df is None:
        st.info("Upload and filter dataset first on 'Data Upload & Preprocessing' page.")
        return

    st.write("### 🔍 Data Preview")
    st.dataframe(df.head())

    # Helper functions for ML Trainer (defined locally to avoid global namespace pollution)
    def auto_target_col(df_input):
        for col in df_input.columns:
            if 2 <= df_input[col].nunique() <= 20: # Heuristic for classification target
                return col
        return df_input.columns[-1] # Default to last column for regression

    target = st.selectbox("🎯 Select Target Column", df.columns, index=df.columns.get_loc(auto_target_col(df)))

    # Determine task type based on target column's unique values
    # Ensure target column is preprocessed before checking nunique
    temp_target_series = df[target].copy()
    if temp_target_series.dtype == 'object' or temp_target_series.dtype.name == 'category':
        temp_target_series = LabelEncoder().fit_transform(temp_target_series.astype(str).fillna(temp_target_series.mode()[0]))
    
    task_type = "Classification" if pd.Series(temp_target_series).nunique() <= 20 else "Regression"
    st.info(f"Detected: {task_type} Problem")

    # Split data into X and y
    X = df.drop(columns=[target])
    y = df[target]

    # Preprocessing pipeline for ML models
    # Identify numerical and categorical columns for the pipeline
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Preprocessing options
    st.sidebar.subheader("ML Preprocessing Options")
    imputation_strategy = st.sidebar.selectbox(
        "Missing Value Imputation",
        ["median", "mean", "most_frequent"],
        key="ml_imputation_strategy"
    )
    scaling_strategy = st.sidebar.selectbox(
        "Feature Scaling",
        ["None", "StandardScaler", "MinMaxScaler"],
        key="ml_scaling_strategy"
    )
    apply_ohe = st.sidebar.checkbox("Apply One-Hot Encoding to Categorical Features", value=True, key="ml_ohe_checkbox")

    # Create preprocessing steps
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=imputation_strategy))
    ])
    if scaling_strategy == "StandardScaler":
        numeric_transformer.steps.append(('scaler', StandardScaler()))
    elif scaling_strategy == "MinMaxScaler":
        numeric_transformer.steps.append(('scaler', MinMaxScaler()))

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
    if apply_ohe:
        categorical_transformer.steps.append(('onehot', OneHotEncoder(handle_unknown='ignore')))
    else:
        # If not OHE, then Label Encode them here if they are still objects/categories
        # This is a bit tricky with ColumnTransformer, often it's better to handle LabelEncoding outside
        # or ensure they are converted to numeric before passing to models.
        # For simplicity, we'll assume LabelEncoder is applied earlier in 'Data Upload & Preprocessing'
        # or that models can handle integer encoded categories. For robust OHE, `apply_ohe` should be used.
        pass # If not OHE, we rely on LabelEncoder from initial preprocessing or model's ability to handle integers.

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough' # Keep other columns (e.g., if some are already numeric and not in numerical_cols)
    )

    # Model Selection & Hyperparameters
    st.sidebar.subheader("Model Configuration")
    selected_model_name = st.sidebar.selectbox("Select ML Model",
                                               ["Random Forest", "Gradient Boosting", "Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine", "Decision Tree", "Naive Bayes", "Linear Regression", "Ridge Regression", "Lasso Regression", "XGBoost", "LightGBM"],
                                               key="selected_ml_model")

    # Model Persistence
    st.sidebar.subheader("Model Persistence")
    model_filename = st.sidebar.text_input("Model Filename (e.g., my_model.joblib)", "trained_model.joblib")
    
    if st.sidebar.button("Save Trained Model", key="save_model_btn"):
        if 'best_trained_model' in st.session_state and st.session_state.best_trained_model is not None:
            try:
                joblib.dump(st.session_state.best_trained_model, model_filename)
                st.sidebar.success(f"Model saved as '{model_filename}'")
            except Exception as e:
                st.sidebar.error(f"Error saving model: {e}")
        else:
            st.sidebar.warning("No model has been trained yet to save.")

    uploaded_model_file = st.sidebar.file_uploader("Upload Model to Load (.joblib)", type=["joblib"], key="load_model_uploader")
    if uploaded_model_file:
        try:
            loaded_model = joblib.load(uploaded_model_file)
            st.session_state.loaded_model = loaded_model
            st.sidebar.success(f"Model '{uploaded_model_file.name}' loaded successfully.")
            st.sidebar.info(f"Loaded model type: {type(loaded_model['classifier_regressor']).__name__}")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}")

    # Train and Evaluate Models Button
    if st.button("🚀 Train & Evaluate Model", key="train_evaluate_btn"):
        # Model Instantiation and Pipeline creation moved inside the button click
        model_instance = None
        model_params = {}

        if selected_model_name == "Random Forest":
            n_estimators = st.sidebar.slider("n_estimators", 50, 500, 100, key="rf_n_estimators_run") # Changed key
            max_depth = st.sidebar.slider("max_depth", 2, 20, 10, key="rf_max_depth_run") # Changed key
            model_params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'random_state': 42}
            model_instance = RandomForestClassifier(**model_params) if task_type == "Classification" else RandomForestRegressor(**model_params)
        elif selected_model_name == "Gradient Boosting":
            n_estimators = st.sidebar.slider("n_estimators", 50, 500, 100, key="gb_n_estimators_run") # Changed key
            learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.5, 0.1, key="gb_learning_rate_run") # Changed key
            model_params = {'n_estimators': n_estimators, 'learning_rate': learning_rate, 'random_state': 42}
            model_instance = GradientBoostingClassifier(**model_params) if task_type == "Classification" else GradientBoostingRegressor(**model_params)
        elif selected_model_name == "Logistic Regression":
            C = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0, key="lr_C_run") # Changed key
            model_params = {'C': C, 'max_iter': 5000}
            model_instance = LogisticRegression(**model_params)
        elif selected_model_name == "Linear Regression":
            model_instance = LinearRegression()
        elif selected_model_name == "K-Nearest Neighbors":
            n_neighbors = st.sidebar.slider("n_neighbors", 1, 20, 5, key="knn_n_neighbors_run") # Changed key
            model_params = {'n_neighbors': n_neighbors}
            model_instance = KNeighborsClassifier(**model_params)
        elif selected_model_name == "Support Vector Machine":
            C_svm = st.sidebar.slider("C (SVM)", 0.1, 10.0, 1.0, key="svm_C_run") # Changed key
            kernel = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], key="svm_kernel_run") # Changed key
            model_params = {'C': C_svm, 'kernel': kernel}
            model_instance = SVC(**model_params) if task_type == "Classification" else SVR(**model_params)
        elif selected_model_name == "Decision Tree":
            max_depth_dt = st.sidebar.slider("max_depth", 2, 20, 10, key="dt_max_depth_run") # Changed key
            model_params = {'max_depth': max_depth_dt, 'random_state': 42}
            model_instance = DecisionTreeClassifier(**model_params)
        elif selected_model_name == "Naive Bayes":
            model_instance = GaussianNB()
        elif selected_model_name == "Ridge Regression":
            alpha_ridge = st.sidebar.slider("alpha (Regularization)", 0.01, 10.0, 1.0, key="ridge_alpha_run") # Changed key
            model_params = {'alpha': alpha_ridge}
            model_instance = Ridge(**model_params)
        elif selected_model_name == "Lasso Regression":
            alpha_lasso = st.sidebar.slider("alpha (Regularization)", 0.01, 10.0, 1.0, key="lasso_alpha_run") # Changed key
            model_params = {'alpha': alpha_lasso}
            model_instance = Lasso(**model_params)
        elif selected_model_name == "XGBoost":
            if xgb:
                n_estimators_xgb = st.sidebar.slider("n_estimators", 50, 500, 100, key="xgb_n_estimators_run") # Changed key
                learning_rate_xgb = st.sidebar.slider("learning_rate", 0.01, 0.5, 0.1, key="xgb_learning_rate_run") # Changed key
                model_params = {'n_estimators': n_estimators_xgb, 'learning_rate': learning_rate_xgb, 'random_state': 42, 'use_label_encoder': False, 'eval_metric': 'logloss' if task_type == 'Classification' else 'rmse'}
                model_instance = xgb.XGBClassifier(**model_params) if task_type == "Classification" else xgb.XGBRegressor(**model_params)
            else:
                st.warning("XGBoost not installed. Please install with `pip install xgboost`.")
                model_instance = None # Explicitly set to None if not available
        elif selected_model_name == "LightGBM":
            if lgb:
                n_estimators_lgb = st.sidebar.slider("n_estimators", 50, 500, 100, key="lgb_n_estimators_run") # Changed key
                learning_rate_lgb = st.sidebar.slider("learning_rate", 0.01, 0.5, 0.1, key="lgb_learning_rate_run") # Changed key
                model_params = {'n_estimators': n_estimators_lgb, 'learning_rate': learning_rate_lgb, 'random_state': 42}
                model_instance = lgb.LGBMClassifier(**model_params) if task_type == "Classification" else lgb.LGBMRegressor(**model_params)
            else:
                st.warning("LightGBM not installed. Please install with `pip install lightgbm`.")
                model_instance = None # Explicitly set to None if not available

        # Create a full pipeline with preprocessing and model
        if model_instance is not None:
            full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                            ('classifier_regressor', model_instance)])
        else:
            full_pipeline = None

        if full_pipeline is None:
            st.error("Selected model is not available or could not be initialized. Please check installations.")
            return

        # Data Splitting and Cross-Validation Options (re-read from sidebar as they are outside the button)
        test_size = st.sidebar.slider("Test Set Size (%)", 10, 50, 30, key="test_size_slider") / 100.0
        use_cross_val = st.sidebar.checkbox("Use K-Fold Cross-Validation", key="use_cv_checkbox") # This key is outside the button, so it's fine
        if use_cross_val:
            n_splits = st.sidebar.slider("Number of Folds (K)", 2, 10, 5, key="cv_n_splits") # This key is outside the button, so it's fine


        # Convert target to numeric if it's still object/category (after initial preprocessing)
        if y.dtype == 'object' or y.dtype.name == 'category':
            y = LabelEncoder().fit_transform(y.astype(str).fillna(y.mode()[0]))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        st.session_state.best_trained_model = None # Reset best model on new training run

        with st.spinner("Training and evaluating model..."):
            try:
                if use_cross_val:
                    cv_scores = cross_val_score(full_pipeline, X, y, cv=n_splits, scoring='accuracy' if task_type == 'Classification' else 'r2')
                    st.success(f"Cross-Validation Scores: {cv_scores}")
                    st.success(f"Mean CV Score: {np.mean(cv_scores):.3f} (Std: {np.std(cv_scores):.3f})")
                    # For CV, we train on full data for the 'best_trained_model' to be saved
                    full_pipeline.fit(X, y)
                    st.session_state.best_trained_model = full_pipeline
                    st.info("Model trained on full dataset for saving after cross-validation.")
                else:
                    full_pipeline.fit(X_train, y_train)
                    pred = full_pipeline.predict(X_test)
                    
                    st.session_state.best_trained_model = full_pipeline # Store the trained model

                    if task_type == "Classification":
                        score = accuracy_score(y_test, pred)
                        st.success(f"✅ Model Accuracy: {score:.3f}")
                        st.write("### Classification Metrics")
                        st.write(f"Precision: {precision_score(y_test, pred, average='weighted', zero_division=0):.3f}")
                        st.write(f"Recall: {recall_score(y_test, pred, average='weighted', zero_division=0):.3f}")
                        st.write(f"F1-Score: {f1_score(y_test, pred, average='weighted', zero_division=0):.3f}")
                        try:
                            auc_score = roc_auc_score(y_test, full_pipeline.predict_proba(X_test)[:, 1], multi_class='ovr')
                            st.write(f"ROC AUC: {auc_score:.3f}")
                        except Exception as e:
                            st.warning(f"Could not calculate ROC AUC (might need binary classification or specific `multi_class` for `predict_proba`): {e}")

                        st.write("### Confusion Matrix")
                        cm = confusion_matrix(y_test, pred)
                        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                                            labels=dict(x="Predicted", y="Actual", color="Count"),
                                            x=[str(i) for i in np.unique(y_test)],
                                            y=[str(i) for i in np.unique(y_test)],
                                            title='Confusion Matrix')
                        st.plotly_chart(fig_cm, use_container_width=True)

                        # ROC Curve (for binary classification)
                        if len(np.unique(y_test)) == 2:
                            try:
                                y_proba = full_pipeline.predict_proba(X_test)[:, 1]
                                fpr, tpr, _ = roc_curve(y_test, y_proba)
                                fig_roc = go.Figure()
                                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_proba)))
                                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line=dict(dash='dash')))
                                fig_roc.update_layout(title='Receiver Operating Characteristic (ROC) Curve',
                                                      xaxis_title='False Positive Rate',
                                                      yaxis_title='True Positive Rate')
                                st.plotly_chart(fig_roc, use_container_width=True)
                            except Exception as e:
                                st.warning(f"Could not plot ROC Curve: {e}")


                    else: # Regression
                        score = r2_score(y_test, pred)
                        st.success(f"✅ Model R2 Score: {score:.3f}")
                        st.write("### Regression Metrics")
                        st.write(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, pred):.3f}")
                        st.write(f"Mean Squared Error (MSE): {mean_squared_error(y_test, pred):.3f}")
                        st.write(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test, pred)):.3f}")

                        st.write("### Residual Plot")
                        residuals = y_test - pred
                        fig_res = px.scatter(x=pred, y=residuals, labels={'x': 'Predicted Values', 'y': 'Residuals'},
                                             title='Residual Plot')
                        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
                        st.plotly_chart(fig_res, use_container_width=True)


                st.write("### 🔍 Sample Predictions")
                # Ensure X_test is a DataFrame for display
                if not isinstance(X_test, pd.DataFrame):
                    # If preprocessor transformed it to numpy array, convert back for display
                    # This is a simplification; ideally, store feature names from preprocessor
                    st.info("Displaying sample predictions. Note: Input features might be transformed.")
                    # Attempt to get feature names after preprocessing if available
                    try:
                        # This works for ColumnTransformer with OneHotEncoder
                        # Need to get feature names from the preprocessor within the pipeline
                        processed_feature_names = full_pipeline.named_steps['preprocessor'].get_feature_names_out(X.columns)
                        result_df = pd.DataFrame(X_test, columns=processed_feature_names)
                    except Exception as e:
                        st.warning(f"Could not get processed feature names for display: {e}. Using generic names.")
                        result_df = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
                else:
                    result_df = X_test.copy()

                result_df["Actual"] = y_test.values
                result_df["Predicted"] = pred
                st.dataframe(result_df.head(10))

                st.write("### 📈 Prediction vs Actual Graph")
                # Convert to DataFrame for Plotly
                plot_data = pd.DataFrame({'Actual': y_test.values, 'Predicted': pred}).head(20)
                fig_pred_actual = px.line(plot_data, y=['Actual', 'Predicted'], title="Actual vs Predicted Values (First 20 Samples)",
                                          labels={'value': 'Value', 'index': 'Sample Index'})
                st.plotly_chart(fig_pred_actual, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred during model training/evaluation: {e}")

        st.markdown("---")
        st.markdown("## 🌀 Unsupervised Learning (Clustering)")
        with st.expander("⚙ KMeans Clustering"):
            # Ensure X_unsup is processed with the selected preprocessing options
            X_unsup_processed = preprocessor.fit_transform(X) # Apply the same preprocessing pipeline
            
            clusters = st.slider("🔢 Number of Clusters (K)", 2, 10, 3, key="kmeans_clusters")

            if st.button("📍 Run Clustering"):
                try:
                    model = KMeans(n_clusters=clusters, random_state=42, n_init='auto')
                    labels = model.fit_predict(X_unsup_processed)
                    sil_score = silhouette_score(X_unsup_processed, labels)

                    # Add cluster labels to the original (or filtered) DataFrame for display
                    clustered_df_display = df.copy()
                    clustered_df_display["Cluster"] = labels
                    st.success(f"🧩 Clustering Done! Silhouette Score: {sil_score:.2f}")
                    st.dataframe(clustered_df_display.head(10))

                    st.write("### 🖼 Cluster Graph (via PCA)")
                    pca = PCA(n_components=2)
                    reduced = pca.fit_transform(X_unsup_processed)
                    cluster_df_plot = pd.DataFrame(reduced, columns=["PC1", "PC2"])
                    cluster_df_plot["Cluster"] = labels.astype(str) # Convert to string for discrete colors in Plotly

                    fig_cluster = px.scatter(cluster_df_plot, x="PC1", y="PC2", color="Cluster", palette="Set2",
                                             title=f"KMeans Clustering with {clusters} Clusters (PCA Reduced)",
                                             labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"})
                    st.plotly_chart(fig_cluster, use_container_width=True)

                except Exception as e:
                    st.error(f"Error during clustering: {e}")

    st.markdown("---")
    st.markdown("## 💾 Load Saved Model for Prediction")
    if 'loaded_model' in st.session_state and st.session_state.loaded_model is not None:
        st.info(f"Loaded model: {type(st.session_state.loaded_model['classifier_regressor']).__name__}")
        st.write("Enter new data for prediction:")
        
        # Create input fields for each feature based on the training data's columns
        input_data = {}
        
        # Attempt to get original feature names from the preprocessor's transformers within the loaded model
        original_feature_names = []
        try:
            # This assumes the preprocessor in the pipeline was trained on the original X columns
            # and that it retains knowledge of these.
            # A more robust solution would be to save these names explicitly with the model.
            # For now, we'll try to reconstruct from the preprocessor's transformers.
            trained_preprocessor = st.session_state.loaded_model.named_steps['preprocessor']
            
            # Get feature names from numerical transformer
            # Check if 'num' transformer exists and has feature_names_in_
            if 'num' in trained_preprocessor.named_transformers_ and hasattr(trained_preprocessor.named_transformers_['num'], 'feature_names_in_'):
                original_feature_names.extend(trained_preprocessor.named_transformers_['num'].feature_names_in_.tolist())
            
            # Get feature names from categorical transformer (before OHE)
            # This is tricky because get_feature_names_out() gives OHE names.
            # We need the *original* categorical column names.
            # Assuming the `categorical_cols` list from the session state is consistent with the loaded model's training.
            if 'df' in st.session_state and st.session_state.df is not None:
                # Re-identify original categorical columns from the stored df
                original_X_cols = st.session_state.df.drop(columns=[target]).columns.tolist()
                original_categorical_cols = st.session_state.df.drop(columns=[target]).select_dtypes(include=['object', 'category']).columns.tolist()
                original_feature_names.extend(original_categorical_cols)
                
                # Filter out duplicates and ensure order if possible (complex)
                original_feature_names = list(dict.fromkeys(original_feature_names)) # Remove duplicates while preserving order
                
                for col in original_feature_names:
                    if col in df.columns and (df[col].dtype == 'object' or df[col].dtype.name == 'category'):
                        input_data[col] = st.text_input(f"Enter value for {col} (Categorical)", key=f"pred_input_{col}")
                    elif col in df.columns and df[col].dtype in [np.number, 'int64', 'float64']:
                        input_data[col] = st.number_input(f"Enter value for {col} (Numeric)", key=f"pred_input_{col}")
                    else:
                        st.warning(f"Could not infer type for feature '{col}'. Using text input.")
                        input_data[col] = st.text_input(f"Enter value for {col}", key=f"pred_input_{col}")

                if st.button("Make Prediction", key="make_prediction_btn"):
                    try:
                        input_df = pd.DataFrame([input_data])
                        # Ensure dtypes match the original DataFrame's dtypes for preprocessing consistency
                        for col in original_feature_names:
                            if col in df.columns:
                                if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                                    input_df[col] = input_df[col].astype(str)
                                elif df[col].dtype in [np.number, 'int64', 'float64']:
                                    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
                                # Handle cases where input might be empty string for numeric, convert to NaN
                                input_df[col] = input_df[col].replace('', np.nan)

                        prediction = st.session_state.loaded_model.predict(input_df)
                        st.success(f"Prediction: {prediction[0]}")
                    except Exception as e:
                        st.error(f"Error making prediction: {e}. Ensure input data matches expected format and types.")
                else:
                    st.warning("Please upload a dataset first to infer input features for prediction.")
        except Exception as e:
            st.warning(f"Could not automatically infer input features from loaded model's preprocessor: {e}. Manual input required.")
            # Fallback to generic text inputs if inference fails
            st.text_area("Enter input features as JSON (e.g., {'feature1': 10, 'feature2': 'A'})", key="manual_pred_input_json")
            if st.button("Make Prediction from JSON", key="make_prediction_json_btn"):
                try:
                    import json
                    manual_input = json.loads(st.session_state.manual_pred_input_json)
                    input_df = pd.DataFrame([manual_input])
                    prediction = st.session_state.loaded_model.predict(input_df)
                    st.success(f"Prediction: {prediction[0]}")
                except Exception as e:
                    st.error(f"Error parsing JSON or making prediction: {e}")
    else:
        st.info("No model loaded. Please train a model or upload a saved model to enable prediction.")


# --- Navigation Functions ---
def display_main_menu():
    st.title("Multi-Task Dashboard")
    st.write("Select a category to explore its tasks.")

    st.markdown('<div class="card btn-grid-lg">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Windows Tasks", key="windows_main_btn", use_container_width=True):
            st.session_state.current_view = "windows_sub_menu"
            st.session_state.selected_category = "Windows Tasks"
            # Ensure other sub-category states are cleared
            st.session_state.selected_sub_category = None
            st.session_state.selected_ml_sub_category = None
            st.rerun()
        if st.button("Docker Tasks", key="docker_main_btn", use_container_width=True):
            st.session_state.current_view = "docker_sub_menu"
            st.session_state.selected_category = "Docker Tasks"
            # Ensure other sub-category states are cleared
            st.session_state.selected_sub_category = None
            st.session_state.selected_ml_sub_category = None
            st.rerun()
    with col2:
        if st.button("Linux Tasks", key="linux_main_btn", use_container_width=True):
            st.session_state.current_view = "linux_sub_menu"
            st.session_state.selected_category = "Linux Tasks"
            # Ensure other sub-category states are cleared
            st.session_state.selected_sub_category = None
            st.session_state.selected_ml_sub_category = None
            st.rerun()
        if st.button("Machine Learning Tasks", key="ml_main_btn", use_container_width=True):
            st.session_state.current_view = "ml_sub_menu"
            st.session_state.selected_category = "Machine Learning Tasks"
            # Ensure other sub-category states are cleared
            st.session_state.selected_sub_category = None
            st.session_state.selected_ml_sub_category = None
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def display_windows_sub_menu():
    st.title(f"{st.session_state.selected_category} Sub-Categories")
    st.write("Select a sub-category to view specific tasks.")

    # Back button
    if st.button("⬅️ Back to Main Menu", key="back_to_main_win", help="Go back to the main category selection", use_container_width=True):
        st.session_state.current_view = "main_menu"
        st.session_state.selected_category = None
        st.session_state.selected_sub_category = None
        st.rerun()

    st.markdown("---")

    # Arrange buttons in rows
    st.markdown('<div class="card btn-grid">', unsafe_allow_html=True)
    col1_row1, col2_row1, col3_row1 = st.columns(3)
    with col1_row1:
        if st.button("Messaging & Communication", key="msg_comm_sub_btn", use_container_width=True):
            st.session_state.current_view = "windows_tasks_detail"
            st.session_state.selected_sub_category = "Messaging & Communication"
            st.rerun()
    with col2_row1:
        if st.button("File & Folder Operations", key="file_ops_sub_btn", use_container_width=True):
            st.session_state.current_view = "windows_tasks_detail"
            st.session_state.selected_sub_category = "File & Folder Operations"
            st.rerun()
    with col3_row1:
        if st.button("Application & System Management", key="app_sys_mgmt_sub_btn", use_container_width=True): # Renamed from Open Applications
            st.session_state.current_view = "windows_tasks_detail"
            st.session_state.selected_sub_category = "Open Applications" # Keep old name for routing to existing function
            st.rerun()

    col1_row2, col2_row2, col3_row2 = st.columns(3)
    with col1_row2:
        if st.button("Connectivity & Network", key="net_conn_sub_btn", use_container_width=True):
            st.session_state.current_view = "windows_tasks_detail"
            st.session_state.selected_sub_category = "Connectivity & Network"
            st.rerun()
    with col2_row2:
        if st.button("System Power Operations", key="power_ops_sub_btn", use_container_width=True):
            st.session_state.current_view = "windows_tasks_detail"
            st.session_state.selected_sub_category = "System Power Operations"
            st.rerun()
    with col3_row2:
        if st.button("Camera", key="camera_sub_btn", use_container_width=True):
            st.session_state.current_view = "windows_tasks_detail"
            st.session_state.selected_sub_category = "Camera"
            st.rerun()

    col1_row3, _, _ = st.columns(3) # New row for new sub-categories
    with col1_row3:
        if st.button("System Monitoring & Info", key="sys_mon_info_sub_btn", use_container_width=True):
            st.session_state.current_view = "windows_tasks_detail"
            st.session_state.selected_sub_category = "System Monitoring & Info"
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def display_linux_sub_menu():
    st.title("Linux Tasks Sub-Categories")
    st.write("Enter your SSH connection details for the RHEL9 machine:")

    # Initialize ssh_connected state if not present
    if 'ssh_connected' not in st.session_state:
        st.session_state.ssh_connected = False
    if 'ssh_host' not in st.session_state:
        st.session_state.ssh_host = ""
    if 'ssh_username' not in st.session_state:
        st.session_state.ssh_username = ""
    if 'ssh_password' not in st.session_state:
        st.session_state.ssh_password = ""

    # SSH Connection Inputs - now initialized with session state values
    ssh_host = st.text_input("RHEL9 IP Address/Hostname", key="ssh_host_input", value=st.session_state.ssh_host)
    ssh_username = st.text_input("Username", key="ssh_username_input", value=st.session_state.ssh_username)
    ssh_password = st.text_input("Password", type="password", key="ssh_password_input", value=st.session_state.ssh_password)

    if st.button("Connect to Linux Machine", key="connect_linux_btn", use_container_width=True):
        if ssh_host and ssh_username and ssh_password:
            # Test connection
            test_output, test_error = execute_ssh_command(ssh_host, ssh_username, ssh_password, "echo 'Connection Test Successful'")
            if not test_error:
                st.session_state.ssh_host = ssh_host
                st.session_state.ssh_username = ssh_username
                st.session_state.ssh_password = ssh_password
                st.session_state.ssh_connected = True
                st.success("SSH connection successful! Select a task category below.")
            else:
                st.session_state.ssh_connected = False
                st.error(f"SSH connection failed: {test_error}")
        else:
            st.error("Please provide all SSH connection details.")

    st.markdown("---")
    st.write("Select a sub-category to view specific tasks.")

    if st.button("⬅️ Back to Main Menu", key="back_to_main_linux", use_container_width=True):
        st.session_state.current_view = "main_menu"
        st.session_state.selected_category = None
        st.session_state.selected_sub_category = None
        st.session_state.ssh_connected = False # Reset connection status on main menu return
        st.rerun()

    st.markdown("---")

    # Linux Sub-Category Buttons (arranged in three rows)
    st.markdown('<div class="card btn-grid">', unsafe_allow_html=True)
    # Row 1: System Information, File System Management, Process Management
    col_l1_r1, col_l2_r1, col_l3_r1 = st.columns(3)
    with col_l1_r1:
        if st.button("System Information", key="linux_sys_info_sub_btn", disabled=not st.session_state.ssh_connected, use_container_width=True):
            st.session_state.current_view = "linux_tasks_detail"
            st.session_state.selected_sub_category = "System Information"
            st.rerun()
    with col_l2_r1:
        if st.button("File System Management", key="linux_file_sys_sub_btn", disabled=not st.session_state.ssh_connected, use_container_width=True):
            st.session_state.current_view = "linux_tasks_detail"
            st.session_state.selected_sub_category = "File System Management"
            st.rerun()
    with col_l3_r1:
        if st.button("Process Management", key="linux_proc_mgmt_sub_btn", disabled=not st.session_state.ssh_connected, use_container_width=True):
            st.session_state.current_view = "linux_tasks_detail"
            st.session_state.selected_sub_category = "Process Management"
            st.rerun()

    # Row 2: Network Management, User & Group Management, Package Management
    col_l1_r2, col_l2_r2, col_l3_r2 = st.columns(3)
    with col_l1_r2:
        if st.button("Network Management", key="linux_net_mgmt_sub_btn", disabled=not st.session_state.ssh_connected, use_container_width=True):
            st.session_state.current_view = "linux_tasks_detail"
            st.session_state.selected_sub_category = "Network Management"
            st.rerun()
    with col_l2_r2:
        if st.button("User & Group Management", key="linux_user_mgmt_sub_btn", disabled=not st.session_state.ssh_connected, use_container_width=True):
            st.session_state.current_view = "linux_tasks_detail"
            st.session_state.selected_sub_category = "User & Group Management"
            st.rerun()
    with col_l3_r2:
        if st.button("Package Management", key="linux_pkg_mgmt_sub_btn", disabled=not st.session_state.ssh_connected, use_container_width=True):
            st.session_state.current_view = "linux_tasks_detail"
            st.session_state.selected_sub_category = "Package Management"
            st.rerun()

    # Row 3: Service Management, Log & Cron Management, Firewall Management, SSH Key Management
    col_l1_r3, col_l2_r3, col_l3_r3 = st.columns(3)
    with col_l1_r3:
        if st.button("Service Management", key="linux_svc_mgmt_sub_btn", disabled=not st.session_state.ssh_connected, use_container_width=True):
            st.session_state.current_view = "linux_tasks_detail"
            st.session_state.selected_sub_category = "Service Management"
            st.rerun()
    with col_l2_r3:
        if st.button("Log & Cron Management", key="linux_log_cron_mgmt_sub_btn", disabled=not st.session_state.ssh_connected, use_container_width=True):
            st.session_state.current_view = "linux_tasks_detail"
            st.session_state.selected_sub_category = "Log & Cron Management"
            st.rerun()
    with col_l3_r3:
        if st.button("Firewall Management", key="linux_firewall_mgmt_sub_btn", disabled=not st.session_state.ssh_connected, use_container_width=True):
            st.session_state.current_view = "linux_tasks_detail"
            st.session_state.selected_sub_category = "Firewall Management"
            st.rerun()
    
    col_l1_r4, _, _ = st.columns(3)
    with col_l1_r4:
        if st.button("SSH Key Management", key="linux_ssh_key_mgmt_sub_btn", disabled=not st.session_state.ssh_connected, use_container_width=True):
            st.session_state.current_view = "linux_tasks_detail"
            st.session_state.selected_sub_category = "SSH Key Management"
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def display_docker_sub_menu():
    st.title("Docker Tasks Sub-Categories")
    st.write("Enter your SSH connection details for the RHEL9 machine where Docker is installed:")

    # Initialize ssh_connected state if not present
    if 'ssh_connected' not in st.session_state:
        st.session_state.ssh_connected = False
    if 'ssh_host' not in st.session_state:
        st.session_state.ssh_host = ""
    if 'ssh_username' not in st.session_state:
        st.session_state.ssh_username = ""
    if 'ssh_password' not in st.session_state:
        st.session_state.ssh_password = ""

    # SSH Connection Inputs - now initialized with session state values
    ssh_host = st.text_input("RHEL9 IP Address/Hostname", key="docker_ssh_host_input", value=st.session_state.ssh_host)
    ssh_username = st.text_input("Username", key="docker_ssh_username_input", value=st.session_state.ssh_username)
    ssh_password = st.text_input("Password", type="password", key="docker_ssh_password_input", value=st.session_state.ssh_password)

    if st.button("Connect to Linux Machine for Docker", key="connect_docker_linux_btn"):
        if ssh_host and ssh_username and ssh_password:
            # Test connection
            test_output, test_error = execute_ssh_command(ssh_host, ssh_username, ssh_password, "echo 'Docker Connection Test Successful'")
            if not test_error:
                st.session_state.ssh_host = ssh_host
                st.session_state.ssh_username = ssh_username
                st.session_state.ssh_password = ssh_password
                st.session_state.ssh_connected = True
                st.success("SSH connection successful for Docker tasks! Select a task category below.")
            else:
                st.session_state.ssh_connected = False
                st.error(f"SSH connection failed: {test_error}")
        else:
            st.error("Please provide all SSH connection details.")

    st.markdown("---")
    st.write("Select a sub-category to view specific Docker tasks.")

    if st.button("⬅️ Back to Main Menu", key="back_to_main_docker", use_container_width=True):
        st.session_state.current_view = "main_menu"
        st.session_state.selected_category = None
        st.session_state.selected_sub_category = None
        st.session_state.ssh_connected = False # Reset connection status on main menu return
        st.rerun()

    st.markdown("---")

    # Docker Sub-Category Buttons (arranged in rows)
    st.markdown('<div class="card btn-grid">', unsafe_allow_html=True)
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        if st.button("Container Management", key="docker_container_sub_btn", disabled=not st.session_state.ssh_connected, use_container_width=True):
            st.session_state.current_view = "docker_tasks_detail"
            st.session_state.selected_sub_category = "Container Management"
            st.rerun()
        if st.button("Image Management", key="docker_image_sub_btn", disabled=not st.session_state.ssh_connected, use_container_width=True):
            st.session_state.current_view = "docker_tasks_detail"
            st.session_state.selected_sub_category = "Image Management"
            st.rerun()
    with col_d2:
        if st.button("Network Management", key="docker_network_sub_btn", disabled=not st.session_state.ssh_connected, use_container_width=True):
            st.session_state.current_view = "docker_tasks_detail"
            st.session_state.selected_sub_category = "Network Management"
            st.rerun()
        if st.button("Volume Management", key="docker_volume_sub_btn", disabled=not st.session_state.ssh_connected, use_container_width=True):
            st.session_state.current_view = "docker_tasks_detail"
            st.session_state.selected_sub_category = "Volume Management"
            st.rerun()
    with col_d3:
        if st.button("System & Info", key="docker_system_sub_btn", disabled=not st.session_state.ssh_connected, use_container_width=True):
            st.session_state.current_view = "docker_tasks_detail"
            st.session_state.selected_sub_category = "System & Info"
            st.rerun()
    
    col_d4, col_d5, _ = st.columns(3)
    with col_d4:
        if st.button("Docker Compose", key="docker_compose_sub_btn", disabled=not st.session_state.ssh_connected, use_container_width=True):
            st.session_state.current_view = "docker_tasks_detail"
            st.session_state.selected_sub_category = "Docker Compose"
            st.rerun()
    with col_d5:
        if st.button("Docker Swarm", key="docker_swarm_sub_btn", disabled=not st.session_state.ssh_connected, use_container_width=True):
            st.session_state.current_view = "docker_tasks_detail"
            st.session_state.selected_sub_category = "Docker Swarm"
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def display_ml_sub_menu():
    st.title("Machine Learning Tasks Sub-Categories")
    st.write("Select a sub-category to manage your ML workflow.")

    if st.button("⬅️ Back to Main Menu", key="back_to_main_ml", use_container_width=True):
        st.session_state.current_view = "main_menu"
        st.session_state.selected_category = None
        st.session_state.selected_ml_sub_category = None # Clear ML sub-category when going back
        st.rerun()

    st.markdown("---")

    st.markdown('<div class="card btn-grid">', unsafe_allow_html=True)
    col_ml1, col_ml2, col_ml3 = st.columns(3)
    with col_ml1:
        if st.button("Data Upload & Preprocessing", key="ml_upload_filter_sub_btn", use_container_width=True):
            st.session_state.current_view = "ml_tasks_detail"
            st.session_state.selected_ml_sub_category = "Data Upload & Preprocessing"
            st.rerun()
    with col_ml2:
        if st.button("Data Visualization", key="ml_visualize_sub_btn", use_container_width=True):
            st.session_state.current_view = "ml_tasks_detail"
            st.session_state.selected_ml_sub_category = "Data Visualization"
            st.rerun()
    with col_ml3:
        if st.button("Model Training & Evaluation", key="ml_trainer_sub_btn", use_container_width=True):
            st.session_state.current_view = "ml_tasks_detail"
            st.session_state.selected_ml_sub_category = "Model Training & Evaluation"
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def display_ml_tasks_detail():
    st.title(f"{st.session_state.selected_category} - {st.session_state.selected_ml_sub_category}") # Corrected variable name

    # Back button to sub-menu
    if st.button(f"⬅️ Back to {st.session_state.selected_category} Sub-Categories", key="back_to_ml_sub"):
        st.session_state.current_view = "ml_sub_menu"
        st.session_state.selected_ml_sub_category = None # Clear ML sub-category when going back
        st.rerun()

    st.markdown("---")

    # Display tasks based on selected ML sub-category
    if st.session_state.selected_ml_sub_category == "Data Upload & Preprocessing":
        display_ml_upload_filter_tasks()
    elif st.session_state.selected_ml_sub_category == "Data Visualization":
        display_ml_visualize_tasks()
    elif st.session_state.selected_ml_sub_category == "Model Training & Evaluation": # Corrected variable name
        display_ml_trainer_tasks()


def main():
    """Main function to run the Streamlit application."""
    # Initialize session state for navigation
    if 'current_view' not in st.session_state:
        st.session_state.current_view = 'main_menu'
    
    # Initialize session state for categories and sub-categories
    if 'selected_category' not in st.session_state:
        st.session_state.selected_category = None
    if 'selected_sub_category' not in st.session_state: # For Windows, Linux, Docker sub-categories
        st.session_state.selected_sub_category = None
    if 'selected_ml_sub_category' not in st.session_state: # For ML sub-categories
        st.session_state.selected_ml_sub_category = None

    # Initialize session state for ML dataframes and models
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'filtered_df' not in st.session_state:
        st.session_state.filtered_df = None
    if 'best_trained_model' not in st.session_state:
        st.session_state.best_trained_model = None
    if 'loaded_model' not in st.session_state:
        st.session_state.loaded_model = None

    # Initialize SSH connection details (moved here for consistency)
    if 'ssh_connected' not in st.session_state:
        st.session_state.ssh_connected = False
    if 'ssh_host' not in st.session_state:
        st.session_state.ssh_host = ""
    if 'ssh_username' not in st.session_state:
        st.session_state.ssh_username = ""
    if 'ssh_password' not in st.session_state:
        st.session_state.ssh_password = ""


    if st.session_state.current_view == 'main_menu':
        display_main_menu()
    elif st.session_state.current_view == 'windows_sub_menu':
        display_windows_sub_menu()
    elif st.session_state.current_view == 'linux_sub_menu':
        display_linux_sub_menu()
    elif st.session_state.current_view == 'docker_sub_menu':
        display_docker_sub_menu()
    elif st.session_state.current_view == 'ml_sub_menu':
        display_ml_sub_menu()
    elif st.session_state.current_view == 'windows_tasks_detail':
        display_windows_tasks_detail()
    elif st.session_state.current_view == 'linux_tasks_detail':
        display_linux_tasks_detail()
    elif st.session_state.current_view == 'docker_tasks_detail':
        display_docker_tasks_detail()
    elif st.session_state.current_view == 'ml_tasks_detail':
        display_ml_tasks_detail()

if __name__ == "__main__":
    main()
