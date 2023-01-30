# Set log path
log_path="C:\GitProjects\AudioMNIST\logs\run_$(date +%Y%m%d).log"

# Show log path
echo "Saving Output Log To $log_path" >> "$log_path" 2>&1

# Show status
echo "Running Main Script..." >> "$log_path" 2>&1

# Run script
python "C:\GitProjects\AudioMNIST\main.py" >> "$log_path" 2>&1

# Show status
echo "Run Finished!" >> "$log_path" 2>&1
