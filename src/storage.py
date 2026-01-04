import os
import shutil
import logging
from pathlib import Path

class StorageManager:
    def __init__(self, recordings_path, max_usage_percent=90):
        self.recordings_path = Path(recordings_path)
        self.max_usage_percent = max_usage_percent
        self._setup_logger()

    def _setup_logger(self):
        self.logger = logging.getLogger("StorageManager")
        self.logger.setLevel(logging.INFO)

    def clean_up(self):
        """
        Deletes the oldest files if disk usage exceeds the threshold.
        """
        while self._get_usage_percent() > self.max_usage_percent:
            oldest_file = self._get_oldest_file()
            if oldest_file:
                self.logger.info(f"Disk usage high. Deleting oldest file: {oldest_file}")
                os.remove(oldest_file)
                # Also delete associated metadata if it exists
                metadata_file = oldest_file.with_suffix('.json')
                if metadata_file.exists():
                    os.remove(metadata_file)
            else:
                self.logger.warning("Disk usage high but no files found to delete.")
                break

    def _get_usage_percent(self):
        total, used, free = shutil.disk_usage(self.recordings_path)
        return (used / total) * 100

    def _get_oldest_file(self):
        # List all mp4 files in the recordings directory
        files = list(self.recordings_path.glob("*.mp4"))
        if not files:
            return None
        
        # Sort by modification time
        files.sort(key=os.path.getmtime)
        return files[0]

if __name__ == "__main__":
    # Test
    import yaml
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    sm = StorageManager(cfg['video']['recordings_path'], cfg['storage']['max_usage_percent'])
    sm.clean_up()
    print("Storage cleanup check completed.")

