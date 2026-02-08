import pynvml
import time

class Q1_Sensor:
    def __init__(self):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.temp_limit = 82  # °C - enter instinct mode

    def get_survival_pressure(self):
        temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
        load = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
        pressure = (temp / self.temp_limit) * (1.0 + (load / 100.0) * 0.2)
        return min(pressure, 1.0)

    def get_time_budget(self):
        pressure = self.get_survival_pressure()
        return 1.0 * (1.0 - pressure)  # seconds; shorter when hot/stressed
