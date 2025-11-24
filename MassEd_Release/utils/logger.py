import csv
import time
import os

class DataLogger:
    """
    Logs system statistics to a CSV file.
    """
    def __init__(self, filename="session_log.csv"):
        self.filename = filename
        self.start_time = time.time()
        self.last_log_time = 0
        self.log_interval = 1.0 # Log every 1 second
        
        # Create file and write header if it doesn't exist
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Elapsed_Sec", "Total_Unique", "Current_Count", "Enter", "Exit", "Status"])

    def log(self, total_unique, current_count, enter_count, exit_count, status):
        """
        Log the current stats if the interval has passed.
        """
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            elapsed = round(current_time - self.start_time, 2)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
            
            with open(self.filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, elapsed, total_unique, current_count, enter_count, exit_count, status])
            
            self.last_log_time = current_time
