import subprocess
import os

def run_garmin_backup(backup_dir, username, password):
    # Hardcoded command to run the garmin-backup executable
    garmin_backup_path = r"C:\Users\michel.marien_icarew\Documents\GitHub\Machine-learning\.venv\Scripts\garmin-backup.exe"
    command = [
        garmin_backup_path,
        f"--backup-dir={backup_dir}",
        username
    ]
    
    # Set up environment variable for browser impersonation if needed
    os.environ['GARMINEXPORT_IMPERSONATE_BROWSER'] = 'ff117'  # Optionally change this value for different browser versions
    
    try:
        # Run the garmin-backup command, passing the password directly
        result = subprocess.run(command, input=password, text=True, check=True, capture_output=True)
        print("Backup completed successfully.")
        print(result.stdout)  # Print any output from the garmin-backup command
    except subprocess.CalledProcessError as e:
        print("Error occurred while running garmin-backup:")
        print(e.stderr)

# Hardcoded values
backup_directory = r"C:\Users\michel.marien_icarew\Documents\GitHub\Machine-learning\activities"
garmin_username = "kimidevliet@tutanota.com"

# Prompt the user for the password
password = input("Enter your Garmin password: ")

# Run the backup
run_garmin_backup(backup_directory, garmin_username, password)
