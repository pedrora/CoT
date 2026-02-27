import cpuinfo, psutil, uuid

class Q1_Machine:
    def __init__(self):
        self.cpu_id = cpuinfo.get_cpu_info()['brand_raw']
        self.gpu_id = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        self.machine_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{self.cpu_id}_{self.gpu_id}"))
        
    def get_header(self):
        return f"{self.cpu_id[:8]}_{self.gpu_id[:8]}_{self.machine_id[:8]}"
    
    def sample(self):
        return {
            "cpu_usage": psutil.cpu_percent(),
            "ram_usage": psutil.virtual_memory().percent,
            "gpu_temp": pynvml.nvmlDeviceGetTemperature(self.handle, 0),
            "timestamp_ms": int((datetime.utcnow() - EPOCH).total_seconds() * 1000)
        }