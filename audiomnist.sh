# Set paths
path_repo="/C/GitProjects/audio-mnist"
path_log="/C/GitProjects/audio-mnist/logs/run_$(date +%Y%m%d).log"

# Show status
echo "Started at $(date +"%Y-%m-%d %T")" >> "$path_log" 2>&1

# Run script
python "$path_repo/audiomnist.py" >> "$path_log" 2>&1

# Show status
echo "Finished at $(date +"%Y-%m-%d %T")" >> "$path_log" 2>&1
