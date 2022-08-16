from time import perf_counter
from typing import Optional

import datetime as dt

def add_time(t, hours=0, minutes=0):
    if len(t) == 8:
        fmt="%Y%m%d"
    elif len(t) == 10:
        fmt="%Y%m%d%H"
    elif len(t) == 12:
        fmt="%Y%m%d%H%M"

    t = dt.datetime.strptime(t, fmt)
    t = t + dt.timedelta(hours=hours, minutes=minutes)
    t = t.strftime("%Y%m%d%H%M")
    return t

def delta_time(t1, t2):
    if len(t1) == 10:
        fmt="%Y%m%d%H"
    elif len(t1) == 12:
        fmt="%Y%m%d%H%M"

    t1 = dt.datetime.strptime(t1, fmt)
    t2 = dt.datetime.strptime(t2, fmt)
    diff = t2 - t1
    if len(t1) == 10:
        diff = int(diff.total_seconds() / 3600)
    elif len(t1) == 12:
        diff = int(diff.total_seconds() / 60)
    return diff    


class Timer:
    """
    A timer which computes the time elapsed since the start/reset of the timer.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """
        Reset the timer.
        """
        self._start = perf_counter()
        self._paused: Optional[float] = None
        self._total_paused = 0
        self._count_start = 1

    def pause(self) -> None:
        """
        Pause the timer.
        """
        if self._paused is not None:
            raise ValueError("Trying to pause a Timer that is already paused!")
        self._paused = perf_counter()

    def is_paused(self) -> bool:
        """
        Returns:
            bool: whether the timer is currently paused
        """
        return self._paused is not None

    def resume(self) -> None:
        """
        Resume the timer.
        """
        if self._paused is None:
            raise ValueError("Trying to resume a Timer that is not paused!")
        self._total_paused += perf_counter() - self._paused  # pyre-ignore
        self._paused = None
        self._count_start += 1

    def seconds(self) -> float:
        """
        Returns:
            (float): the total number of seconds since the start/reset of the
                timer, excluding the time when the timer is paused.
        """
        if self._paused is not None:
            end_time: float = self._paused  # type: ignore
        else:
            end_time = perf_counter()
        return end_time - self._start - self._total_paused

    def avg_seconds(self) -> float:
        """
        Returns:
            (float): the average number of seconds between every start/reset and
            pause.
        """
        return self.seconds() / self._count_start
