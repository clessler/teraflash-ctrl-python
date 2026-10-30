"""
Copyright (c) 2026 University of Bern, Space Research & Planetary Sciences, Linus Leo Stöckli.

This work is licensed under the Creative Commons
Attribution-NonCommercial 4.0 International License.
To view a copy of this license, visit
https://creativecommons.org/licenses/by-nc/4.0/
"""

import time
from queue import Queue, Empty
import threading

import numpy as np
import platform
import socket
import subprocess
import logging

from math_utils import get_fft


class DataContainer:

    def __init__(self, n=1000):
        """
            Data container that stores the data sent by the instrument
        Args:
            n: length of dataset (depends on the selected range)
        """
        self.time = np.zeros(n)
        self.freq = np.zeros(n)

        self.signal_1 = np.zeros(n)
        self.fft_1_amp = np.zeros(n)
        self.fft_1_phase = np.zeros(n)

        self.signal_2 = np.zeros(n)
        self.fft_2_amp = np.zeros(n)
        self.fft_2_phase = np.zeros(n)

        # zero-line offsets (nA) subtracted from the latest frame: raw = signal + offset
        self.offset_1 = 0.0
        self.offset_2 = 0.0
        # latest decoded Frame, replaced as a whole so one read is never a mix of two frames
        self.frame = None


class Frame:

    def __init__(self, time, signal_1, signal_2, offset_1, offset_2):
        """
            One decoded frame as received, independent of the averaging in DataContainer
        Args:
            time: time axis (ps) in use when the frame was decoded
            signal_1, signal_2: traces (nA) with the zero line subtracted
            offset_1, offset_2: subtracted zero-line offsets (nA); raw = signal + offset
        """
        self.time = time
        self.signal_1 = signal_1
        self.signal_2 = signal_2
        self.offset_1 = offset_1
        self.offset_2 = offset_2


# TF5-Host's "0-line: auto" (manual 5.2.9, the standard setting) zeroes the signal at the
# beginning of the trace. In all 61 traces saved by TF5-Host in our demo data, the mean of
# the first 5 samples (0.25 ps) is zero to the file precision (<= 2e-7 nA) while the first
# sample alone is not (up to 0.11 nA), so we subtract that mean too. Inferred from the
# files; the exact TF5-Host rule is not documented.
ZERO_LINE_SAMPLES = 5


def decode_frame(payload: bytes, antenna_range: float):
    """
        Decodes a data payload (header removed) of big-endian int32 pairs
    Args:
        payload: the frame without its 52-byte header
        antenna_range: TIA range (nA)

    Returns: (signal_1, signal_2, offset_1, offset_2) in nA, with the zero line
        (mean of the first ZERO_LINE_SAMPLES samples) subtracted from each signal
    """
    types = np.dtype([("signal_1", np.int32), ("signal_2", np.int32)]).newbyteorder('>')
    arr = np.frombuffer(payload, dtype=types)
    scale_factor = 20.0 * 1000.0 / antenna_range
    raw_1 = arr['signal_1'] / scale_factor / 2 ** 16
    raw_2 = arr['signal_2'] / scale_factor / 2 ** 16
    offset_1 = float(np.mean(raw_1[:ZERO_LINE_SAMPLES]))
    offset_2 = float(np.mean(raw_2[:ZERO_LINE_SAMPLES]))
    return raw_1 - offset_1, raw_2 - offset_2, offset_1, offset_2


# global variables for thread communication
data = DataContainer()
n_avg = 0
status = ""


class TopticaSocket:
    def __init__(self,
                 ip: str,
                 running: threading.Event,
                 connected: threading.Event,
                 cmd_ack: threading.Event,
                 buffer_emptied: threading.Event,
                 range_changed: threading.Event,
                 acq_running: threading.Event,
                 avg_data: threading.Event):
        """
            TCP Socket struct, used for the communication between the server (Computer) and the client (instrument)
            with two TCP connections (one for configuration and one for data transmission)
        Args:
            ip: ip address
            running: threading event that signals if the application is running
            connected: threading event that signals if a client is connected
            cmd_ack: threading event that signals when a command is acknowledged
            range_changed: threading event that signals when the range has been changed in the configuration
        """
        self.send_header = b'\xcd\xef\x124x\x9a\xfe\xdc\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
        self.r_stat_header = b'\xcd\xef\x124x\x9a\xfe\xdc\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x02'
        self.r_dat_header = b'\xcd\xef\x124x\x9a\xfe\xdc\x00\x00\x00\x01\x00\t\x1a\xe6\x03\xe8\x00\x00\x04L\x00\x00\x0c\xcc\xcc\xcc\x00\x00\x16\x0b\x00\x00]\xd8\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x00'
        self.r_dat_header = b'\xcd\xef\x124x\x9a\xfe'
        self.full_data_header_len = 52
        self.read_header_len = 19
        self.data_header_len = 7

        self.avg_countdown = 0

        self.config_server_address = (ip, 6341)
        self.data_server_address = (ip, 6342)

        self.t_begin = 1000.00
        self.range = 50
        self.antenna_range = 1000.0

        self.running = running
        self.connected = connected
        self.cmd_ack = cmd_ack
        self.range_changed = range_changed
        self.buffer_emptied = buffer_emptied
        self.acq_running = acq_running
        self.avg_data = avg_data

        if not self.ping(ip):
            raise ConnectionError

    @staticmethod
    def ping(host: str):
        """
            determines whether a device is connected or not
        Args:
            host: IP address as a string

        Returns: True for connected, False for not connected

        """
        res = False
        ping_param = "-n" if platform.system().lower() == "windows" else "-c"
        command = ['ping', ping_param, '1', host]
        result = subprocess.call(command)
        if result == 0:
            res = True
        return res

    def _frame_command(self, cmd, log: bool = True):
        """
            builds the bytes sent for a command: send_header + length byte + ASCII command
        Args:
            cmd: (legacy length byte, command string) tuple as queued by teraflash.py
            log: log the sent command (off for the heartbeat, which runs every 0.25 s)

        Returns: the message bytes, or None if the command cannot be framed

        """
        (b, c) = cmd
        payload = c.encode()
        # the byte after the header is the command length: it equals len(command) for
        # every command that has been acked. compute it here instead of trusting the
        # per-command tables, which were wrong for e.g. "ACQUISITION : BEGIN 500.0"
        # (declared 26, sent 25), "TRANSMISSION : BLOCK" and "ACQUISITION : AVERAGE 10000"
        if not 1 <= len(payload) <= 255:
            # never raise here: an exception would end this thread and drop the
            # connection (which turns the laser off). the caller times out instead.
            logging.error(f"[TCP CONF] cannot frame command of length {len(payload)}: {c!r}")
            return None
        length_byte = bytes([len(payload)])
        if b != length_byte:
            logging.warning(f"[TCP CONF] corrected length byte for {c!r}: "
                            f"legacy {b!r}, sending {length_byte!r}")
        if log:
            logging.debug(f"[TCP CONF] sending: {(length_byte, c)}")
        return self.send_header + length_byte + payload

    def wait_for_answer(self, client: socket, length: int = 1024):
        """
            waits for the device to acknowledge the previously sent command
        Args:
            client: socket object
            length: length of the buffer

        Returns: True for valid response, False for error

        """
        while self.running.is_set():
            try:
                _data = client.recv(length)[self.read_header_len:]
            except socket.timeout:
                # no reply yet - loop back so `running` gets re-checked instead
                # of blocking on recv() forever if the device stops responding
                continue
            if _data:
                _data_decoded = _data.decode("utf-8", "ignore")
                if "OK" in _data_decoded:
                    logging.debug(f"[TCP CONF] received: {_data_decoded}")
                    return True
                elif "MON" in _data_decoded:
                    return True
                logging.debug(f"[TCP CONF] received: {_data_decoded}")
            else:
                logging.error(f"[TCP CONF] No valid reply from device: {_data.decode('utf-8', 'ignore')}")
                return False

    def run_conf_tcp(self, cmd_queue: Queue):
        """
            Config TCP thread on port 6341. This handles all the configurations for the device.
        Args:
            cmd_queue: Queue from main thread with all the interactive

        Returns:

        """
        global status
        global data
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # Set the SO_REUSEADDR option to allow reuse of the port
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # bind server to the address (only works when the address exists)
            s.bind(self.config_server_address)
            logging.info(f"[TCP CONF] Starting server at address {self.config_server_address}")
            # wait for connections
            s.listen()
            # accept connection from device
            client, addr = s.accept()
            # bounded recv/send so a silent device can never wedge this thread
            # forever - matches run_tcp_dat's existing pattern
            client.settimeout(2)
            logging.info(f"[TCP CONF] Connected by client with address {addr}")
            # now we are connected
            self.connected.set()

            try:
                while self.running.is_set():
                    try:
                        # check the queue for commands
                        try:
                            cmd = cmd_queue.get(block=False)
                            (b, c) = cmd
                            message = self._frame_command(cmd)
                            if message is None:
                                continue
                            # send command
                            client.send(message)
                            if c == "SYSTEM : TELL STATUS":
                                # if we request the status, save the response
                                status = client.recv(2048)[self.read_header_len:].decode("utf-8", "ignore")
                                self.cmd_ack.set()
                            elif "RANGE" in c:
                                # if we change the range, also change it for the data thread
                                parts = c.split(" ")
                                self.range = float(parts[-1])
                                data.time = np.linspace(self.t_begin, self.t_begin + self.range, 20 * int(self.range) + 1)
                                # wait for acknowledge
                                if not self.wait_for_answer(client):
                                    return
                                self.cmd_ack.set()
                            elif "BEGIN" in c:
                                # if we change the range, also change it for the data thread
                                parts = c.split(" ")
                                self.t_begin = float(parts[-1])
                                data.time = np.linspace(self.t_begin, self.t_begin + self.range, 20 * int(self.range) + 1)
                                # wait for acknowledge
                                if not self.wait_for_answer(client):
                                    return
                                self.cmd_ack.set()
                            else:
                                # wait for acknowledge
                                if not self.wait_for_answer(client):
                                    return
                                self.cmd_ack.set()
                        except Empty:
                            # just to a simple heartbeat
                            message = self._frame_command((b'\x12', "SYSTEM : MONITOR 1"), log=False)
                            client.send(message)
                            # wait for acknowledge
                            if not self.wait_for_answer(client):
                                return
                    except socket.timeout:
                        # device didn't respond to this cycle's recv/send within
                        # the socket timeout - loop back so `running` gets
                        # re-checked instead of this thread hanging forever
                        logging.debug("[TCP CONF] recv/send timed out this cycle, retrying")
                        continue
                    time.sleep(0.25)
            finally:
                # always release the device-facing socket, even on early return
                # or an escaping exception - otherwise the listening socket's
                # `with` block can't close either, and the port stays bound
                try:
                    client.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                client.close()

    def run_tcp_dat(self, cmd_queue: Queue, config_queue: Queue):
        """
            Data TCP thread on port 6342. This handles receives all the streamed data from the device and decodes it.
        """
        global data
        global n_avg
        antenna_range = 1000.0

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # Set the SO_REUSEADDR option to allow reuse of the port
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # bind server to the address (only works when the address exists)
            s.bind(self.data_server_address)
            logging.info(f"[TCP DAT] Starting server at address {self.data_server_address}")
            # wait for connections
            s.listen()
            # accept connection from device
            client, addr = s.accept()
            client.settimeout(2)

            logging.info(f"[TCP DAT] Connected by client with address {addr}")
            # now we are connected
            try:
                while self.running.is_set():
                    if self.range_changed.is_set():
                        # range has changed and needs to be adjusted
                        # need to empty the read buffer
                        while self.running.is_set():
                            logging.debug(f"emptying buffer...")
                            try:
                                client.recv(32100)
                            except socket.timeout:
                                break
                        self.buffer_emptied.set()
                        self.range_changed.clear()

                    # data always comes in the shape of 4 datasets each as 16bit ints with length of
                    # (20 * self.range + 1)
                    # the header is 52 8 bit ints and since we read 8 bit ints we need to multiply the data by 2
                    if self.range_changed.is_set():
                        continue
                    if not self.acq_running.is_set():
                        # no data received
                        time.sleep(1.0)
                        continue
                    try:
                        raw_data = client.recv(2 * 4 * (20 * int(self.range) + 1) + self.full_data_header_len)
                    except socket.timeout:
                        continue
                    if not raw_data:
                        # no data received
                        time.sleep(1.0)
                        continue

                    if raw_data[:self.data_header_len] != self.r_dat_header:
                        continue

                    if len(raw_data) != 2 * 4 * (20 * int(self.range) + 1) + self.full_data_header_len:
                        skip_this = False
                        while self.running.is_set():
                            try:
                                to_append = client.recv(
                                    2 * 4 * (20 * int(self.range) + 1) + self.full_data_header_len - len(raw_data))
                                raw_data += to_append
                                if len(raw_data) == 2 * 4 * (20 * int(self.range) + 1) + self.full_data_header_len:
                                    break
                                if len(raw_data) == 2 * 4 * (20 * int(self.range) + 1) + self.full_data_header_len:
                                    skip_this = True
                                    break
                            except socket.timeout:
                                skip_this = True
                                break
                        if skip_this:
                            continue

                    # TODO: the following code is not properly implemented yet, we need
                    # to check how we handle it, if the data comes not in the proper packet length

                    _data = raw_data[self.full_data_header_len:]

                    # check if header is at the beginning of the received payload
                    # if raw_data[:self.data_header_len] == self.r_dat_header:
                    #     # remove the header
                    #     _data = raw_data[self.data_header_len:]
                    # else:
                    #     _data = raw_data[self.data_header_len:]
                    #     # print("header not in the beginning")
                    #     while self.running.is_set():
                    #         logging.info("data does not have correct length...")
                    #
                    #         # check if the data is of the correct length
                    #         if len(_data) != 2 * 4 * (20 * self.range + 1):
                    #             _data = _data + client.recv(2 * 4 * (20 * int(self.range) + 1) - len(_data))
                    #         else:
                    #             break
                    try:
                        if not config_queue.empty():
                            antenna_range = config_queue.get()

                        # decode received payload, subtracting the zero line (see ZERO_LINE_SAMPLES)
                        signal_1, signal_2, offset_1, offset_2 = decode_frame(_data, antenna_range)
                        data.frame = Frame(data.time, signal_1, signal_2, offset_1, offset_2)
                        data.offset_1 = offset_1
                        data.offset_2 = offset_2

                        if self.avg_data.is_set():
                            # copies: the += below must not change the published frame
                            if data.signal_1.shape == signal_1.shape:
                                data.signal_1 += signal_1
                            else:
                                data.signal_1 = signal_1.copy()
                            if data.signal_2.shape == signal_2.shape:
                                data.signal_2 += signal_2
                            n_avg += 1
                        else:
                            data.signal_1 = signal_1
                            data.signal_2 = signal_2
                            n_avg = 0

                        # do fft of signal 1
                        pulse = data.signal_1
                        f, a, arg = get_fft(data.time, pulse)
                        data.freq = f
                        data.fft_1_amp = a / np.max(a)
                        data.fft_1_phase = arg

                        # do fft of signal 2
                        pulse = data.signal_2
                        f, a, arg = get_fft(data.time, pulse)
                        data.fft_2_amp = a / np.max(a)
                        data.fft_2_phase = arg

                        # update avg countdown
                        if self.avg_countdown > 0:
                            self.avg_countdown -= 1

                    except Exception as e:
                        logging.error(e)
                        logging.error(f"{len(data.signal_1)=}")
            finally:
                # always release the device-facing socket, even on early return
                # or an escaping exception - otherwise the listening socket's
                # `with` block can't close either, and the port stays bound
                try:
                    client.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                client.close()
