from pathlib import Path
import configparser
import threading
import time
from typing import Optional
from modules.common.logger import get_app_logger
import subprocess
import re

class ConfigManager:
    _instances = {} 
    _lock = threading.Lock()
    
    def __new__(cls, camera_name="cam1"):
        with cls._lock:
            if camera_name not in cls._instances:
                cls._instances[camera_name] = super().__new__(cls)
                cls._instances[camera_name]._initialized = False
            return cls._instances[camera_name]

    def __init__(self, camera_name="cam1"):        
        if self._initialized:
            return
        
        self.camera_name = camera_name
        self.logger = get_app_logger(camera_name, __name__)
        self.camera_index = self._extract_camera_index()
        self.streaming_port = 9001 + self.camera_index * 100
        self._init_paths()
        self._init_configs()
        self._start_watch_thread()
        self._initialized = True

    def _init_paths(self):
        self.config_dir = Path("configs") / self.camera_name
        self.config_path = self.config_dir /  "config.ini"
        self.visualize_path = self.config_dir / "vis.ini"
        
        if not self.config_path.exists():
            self.logger.info(f"Config file not found for {self.camera_name}, generating...")
            self._run_make_cfg()

        self.config_dir.mkdir(exist_ok=True)

    def _extract_camera_index(self) -> int:
        match = re.search(r"(\d+)$", self.camera_name)
        return int(match.group(1)) if match else 0
    
    def _run_make_cfg(self):
        script_path = Path("configs/make_cfg.sh").resolve()  
        if not script_path.exists():
            self.logger.error(f"Script {script_path} not found! Please check...")
            return

        try:
            result = subprocess.run(
                ["bash", str(script_path), self.camera_name, str(self.streaming_port)], 
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            if result.returncode == 0:
                self.logger.info(f"Successfully created config for {self.camera_name}")
            else:
                self.logger.error(f"Failed to create config: {result.stderr}")

        except Exception as e:
            self.logger.error(f"Error running make_cfg.sh: {e}")

    def _init_configs(self):
        self.config = configparser.ConfigParser()
        self.visualize_config = configparser.ConfigParser()
        self.mtimes = {
            self.config_path: 0,
            self.visualize_path: 0
        }
        self._load_all_configs()

    def _load_all_configs(self):
        try:
            self._load_config(self.config_path, self.config)
            self._load_config(self.visualize_path, self.visualize_config)
        except Exception as e:
            self.logger.error(f"Error loading configs: {e}")

    def _load_config(self, path: Path, config: configparser.ConfigParser):
        if path.exists():
            try:
                config.read(path)
                self.mtimes[path] = path.stat().st_mtime
                self.logger.debug(f"Loaded config: {path}")
            except Exception as e:
                self.logger.error(f"Error loading {path}: {e}")

    def _start_watch_thread(self):
        self.running = True
        self.thread = threading.Thread(target=self._watch_files, daemon=True)
        self.thread.start()
        self.logger.info("Config watch thread started")

    def _watch_files(self):
        while self.running:
            try:
                for path, old_mtime in self.mtimes.items():
                    if path.exists():
                        current_mtime = path.stat().st_mtime
                        if current_mtime > old_mtime:
                            config = self.config if path == self.config_path else self.visualize_config
                            self._load_config(path, config)
                            self.logger.info(f"🔄 Config updated: {path.name}")
                time.sleep(2)
            except Exception as e:
                self.logger.error(f"Error in watch thread: {e}")
                time.sleep(5)  

    def get_config(self) -> configparser.ConfigParser:
        return self.config

    def get_visualize_config(self) -> configparser.ConfigParser:
        return self.visualize_config

    def stop_watching(self):
        self.running = False
        self.thread.join()
        self.logger.info("Config watch thread stopped")

_config_managers = {}

def get_app_config(camera_name) -> configparser.ConfigParser:
    if camera_name not in _config_managers:
        _config_managers[camera_name] = ConfigManager(camera_name)
    return _config_managers[camera_name].get_config()

def get_vis_config(camera_name) -> configparser.ConfigParser:
    if camera_name not in _config_managers:
        _config_managers[camera_name] = ConfigManager(camera_name)
    return _config_managers[camera_name].get_visualize_config()

def get_path_vis_config(camera_name) -> str:
    if camera_name not in _config_managers:
        _config_managers[camera_name] = ConfigManager(camera_name)
    return _config_managers[camera_name].visualize_path
