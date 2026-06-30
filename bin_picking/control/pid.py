from __future__ import annotations
import numpy as np
import cv2 as cv
import time
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from dataclasses import dataclass


@dataclass
class PID:
    """
    Discrete PID controller with anti-windup.

    Anti-windup: the integral term only accumulates when the unsaturated output
    is within the [0, 10] V output range, preventing windup against the hardware limits.

    Args:
        p:         Proportional gain.
        i:         Integral gain.
        d:         Derivative gain.
        dt:        Sample period in seconds. Set via ``set_control_rate()``.
        reference: Setpoint. Set via ``set_ref()``.
    """

    p: float
    i: float
    d: float
    dt: float = 0.0
    reference: float = 0.0
    _e: float = 0.0
    _e_prev: float = 0.0
    _integral: float = 0.0

    @property
    def error(self) -> float:
        """Current tracking error (reference − actual)."""
        return self._e

    def apply_control(self) -> float:
        """Compute and return the saturated control output [0, 10] V."""
        return np.clip((self.p * self._e
              + self.i * self._integral
              + self.d * (self._e - self._e_prev)), 0.0, 10.0)

    def update(self, actual: float) -> None:
        """
        Update the controller state with the latest measurement.

        Args:
            actual: Measured process variable (e.g. image brightness).
        """
        self._e_prev = self._e
        self._e = self.reference - actual

        output_unsat = self.p * self._e + self.i * self._integral + self.d * (self._e - self._e_prev)
        # Only integrate when unsaturated to prevent windup.
        if 0.0 < output_unsat < 10.0:
            self._integral += self._e * self.dt

    def set_ref(self, ref: float) -> None:
        """Set the controller setpoint."""
        self.reference = ref

    def set_control_rate(self, rate: float) -> None:
        """
        Set the control loop rate.

        Args:
            rate: Loop frequency in Hz. Converted to sample period dt = 1/rate.
        """
        self.dt = 1.0 / rate



class ControlLoop:
    """
    Closed-loop brightness controller for the ring light.

    Computes image brightness (67th-percentile grayscale value) and feeds it
    into a PID controller whose output drives the ring-light voltage (0–10 V).

    Usage::

        loop = ControlLoop()
        loop.add_controller(PID(p=0.1, i=0.01, d=0.0))
        loop.controller.set_ref(128.0)   # target brightness
        voltage = loop(frame)            # call per frame
    """

    controller: PID
    data: np.ndarray
    brightness: float
    ff: float

    def add_controller(self, c: PID) -> None:
        """Attach a PID instance as the active controller."""
        self.controller = c

    def __call__(self, data: np.ndarray) -> float:
        """
        Process one BGR frame and return the new control voltage.

        Args:
            data: H×W×3 uint8 BGR image.

        Returns:
            Control voltage in [0, 10] V.
        """
        self.data = data
        self._calculate_brightness()
        return self._apply()

    def _calculate_brightness(self):
        gray_scale = cv.cvtColor(self.data, cv.COLOR_BGR2GRAY).ravel()
        self.brightness = np.quantile(gray_scale, 67)

    def _apply(self):
        self.controller.update(self.brightness)
        return self.controller.apply_control()


    def auto_update(self, name, h, w, idx, stop_event, cb, start_event, finish, tol=0.1):
        '''
        Args:
            name: name of the shared memory.
            h,w: height and widht of the image.
            idx: (mp.Value) to index the array.
            stop_event: (mp.Event) to stop the process.
            cb: callback, to apply the control value.
        '''

        self._shm = SharedMemory(name=name, create=False)
        self._data = np.ndarray(shape=(3, h, w, 3), dtype=np.uint8,
                                buffer=self._shm.buf)
 
        last_fetch_time = time.perf_counter()
        last_idx = 0
        self.data = self._data[idx.value % 3, :].astype(np.uint8) # Fetch first index.
        voltage = 0.0
        settled = 0
 
        while not stop_event.is_set():

            if not start_event.is_set():

                while not start_event.wait(timeout=0.1):
                    if stop_event.is_set():
                        self._shm.close()
                        return
                    
                start = time.perf_counter()
 
            # Non-blocking rate limit.
            if (time.perf_counter() - last_fetch_time) < 0.02:
                continue
 
            if idx.value == last_idx:
                continue
            last_idx = idx.value

            self.data = self._data[(idx.value - 1) % 3, :].astype(np.uint8)
            voltage = self.__call__(self.data)
            cb(voltage)

            err = self.controller.error

            at_target  = abs(err) < tol
            stuck_low  = voltage <= 0.0  and err < 0.0
            stuck_high = voltage >= 10.0 and err > 0.0

            settled = settled + 1 if (at_target or stuck_low or stuck_high) else 0

            if settled >= 5 or (time.perf_counter() - start) > 2.0:
                finish.set()
                time.sleep(0.1)
 
        self._shm.close()




