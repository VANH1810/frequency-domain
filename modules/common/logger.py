import os
import threading
import time
import configparser
from typing import Dict
from loguru import logger
import sys
from pathlib import Path
import re 
import subprocess

class LoggerConfigMonitor(threading.Thread):
    def __init__(self, config_file: str, loggers: Dict[str, 'Logger']):
        super().__init__()
        self.config_file = config_file
        self.loggers = loggers
        self.last_modified = 0
        self.daemon = True
        self.config_lock = threading.Lock() 

    def run(self):
        while True:
            try:
                if not os.path.exists(self.config_file):
                    time.sleep(1)
                    continue
                
                current_modified = os.path.getmtime(self.config_file)
                if current_modified > self.last_modified:
                    self.last_modified = current_modified
                    self._reload_config()
                
                time.sleep(1)
            except Exception as e:
                print(f"Error monitoring config file: {e}")
    
    def _reload_config(self):
        config = configparser.ConfigParser()
        config.read(self.config_file)
        if "common" not in config:
            print(f"Missing 'common' section in {self.config_file}")
            return
        section = config["common"]
        debug_enabled = section.getboolean('debug', False)
        max_bytes = section.getint('max_bytes', 10 * 1024 * 1024)  
        backup_count = section.getint('backup_count', 5) 
        
        with self.config_lock:
            for _, logger_instance in list(self.loggers.items()):  
                logger_instance.update_config(debug_enabled, max_bytes, backup_count)

class Logger:
    _instances = {}
    _instances_lock = threading.Lock()  
    def __new__(cls, name="default", *args, **kwargs):
        if name not in cls._instances:            
            instance = super().__new__(cls)
            cls._instances[name] = instance
        return cls._instances[name]
    
    def __init__(self, name="default", 
                 camera_name="cam.cfg.default", 
                 log_file="logs/cam.cfg.default/app.log", 
                 config_file="./configs/cam.cfg.default/config.ini", 
                 max_bytes=10*1024*1024, backup_count=5):  
        if hasattr(self, 'log_file'):
            return  
        
        self.config_file = config_file
        self.camera_name = camera_name
        self.name = name
        self.log_file = log_file
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.recreated_log = False   
        self._load_init_config()
        self._setup_logger()
        threading.Timer(5.0, self._start_log_file_monitor).start()
    
        
        with Logger._instances_lock:
            if not hasattr(self.__class__, '_config_monitor'):
                self.__class__._config_monitor = LoggerConfigMonitor(config_file, self._instances)
                self.__class__._config_monitor.start()


    def _setup_logger(self):
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)        
        logger.remove()
        logger.add(
            self.log_file, 
            rotation=f"{self.max_bytes} B", 
            retention=self.backup_count, 
            level="DEBUG" if self.debug_enabled else "INFO",
            format="{time:YYYY-MM-DDTHH:mm:ss.SSSSSSZZ} - {level} - {name}:{function}:{line} - {message}", 
            enqueue=True
        )
        
        logger.add(
            sys.stdout,
            level="DEBUG" if self.debug_enabled else "INFO",
            format="{time:YYYY-MM-DDTHH:mm:ss.SSSSSSZZ} - {level} - {name}:{function}:{line} - {message}", 
            enqueue=True

        )
        
        
    def _load_init_config(self):
        if not os.path.exists(self.config_file):
            print(f"Config file not found for {self.camera_name}, generating...")
            self._run_make_cfg()
        
        config = configparser.ConfigParser()
        config.read(self.config_file)
        section = config["common"]
        self.debug_enabled = section.getboolean('debug', False)
        self.max_bytes = section.getint('max_bytes', 10* 1024 * 1024)
        self.backup_count = section.getint('backup_count', 5)
        
    def _extract_camera_index(self) -> int:
        match = re.search(r"(\d+)$", self.camera_name)
        return int(match.group(1)) if match else 0
        
    def _run_make_cfg(self):
        script_path = Path("configs/make_cfg.sh").resolve()  
        if not script_path.exists():
            print(f"Script {script_path} not found ! Please check ...")
            return
        self.camera_index = self._extract_camera_index()
        self.streaming_port = 9001 + self.camera_index * 100
        try:
            result = subprocess.run(
                ["bash", str(script_path), self.camera_name, str(self.streaming_port)], 
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            if result.returncode == 0:
                print(f"Successfully created config for {self.camera_name}")
            else:
                print(f"Failed to create config: {result.stderr}")

        except Exception as e:
            print(f"Error running make_cfg.sh: {e}")

    def _check_log_file(self):
        if not os.path.exists(self.log_file):
            print(f"File log '{self.log_file}' was deleted. Recreating...")
            if not self.recreated_log:
                self._setup_logger()    
                self.recreated_log = True
                
    def _start_log_file_monitor(self):
        self.recreated_log = False
        self._check_log_file()
        threading.Timer(5.0, self._start_log_file_monitor).start()
    
    def update_config(self, debug_enabled: bool, max_bytes: int, backup_count: int):
        self.debug_enabled = debug_enabled
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self._setup_logger() 
    
    def debug(self, message):
        logger.opt(depth=1).debug(message)
    
    def info(self, message):
        logger.opt(depth=1).info(message)
    
    def warning(self, message):
        logger.opt(depth=1).warning(message)
    
    def error(self, message):
        logger.opt(depth=1).error(message)
    
    def critical(self, message):
        logger.opt(depth=1).critical(message)



def get_app_logger(camera_name="cam1", name="app"):
    return Logger(name=name[:30].ljust(30), 
                  camera_name=camera_name, 
                  log_file=f"logs/{camera_name}/app.log", 
                  config_file=f"configs/{camera_name}/config.ini")
