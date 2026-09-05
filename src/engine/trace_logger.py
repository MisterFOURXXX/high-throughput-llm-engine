import os
import subprocess
import threading
import time

class TraceLogger:
    def __init__(self, log_dir: str = "./artifacts/nccl_traces"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.process = None

    def start_nvtx_trace(self, duration_sec: int = 10):
        """Start NVIDIA Nsight Systems trace using nvprof / nsys."""
        # Example using nsys (requires nsys installed)
        output_file = os.path.join(self.log_dir, f"trace_{int(time.time())}.nsys-rep")
        cmd = [
            "nsys", "profile",
            "--trace=cuda,nvtx,osrt",
            "-o", output_file,
            "--duration", str(duration_sec),
            "--force-overwrite",
            "python", "-c", "import time; time.sleep(10)"  # dummy target
        ]
        # In real usage, attach to the serving process.
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def stop_trace(self):
        if self.process:
            self.process.terminate()
            self.process.wait()