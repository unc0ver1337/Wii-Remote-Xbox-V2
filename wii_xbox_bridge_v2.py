"""MotionPlus + Nunchuk to Xbox 360. MotionPlus aiming with automatic gravity correction and bounded yaw."""
from __future__ import annotations

import argparse
import math
import statistics
import sys
import time


REPORT_MODE = 0x37
EXTENSION_OFFSET = 16


def extension_payload(report):
    """Only accept tagged MotionPlus passthrough packets from the selected mode."""
    if not report or report[0] != REPORT_MODE or len(report) < EXTENSION_OFFSET + 6:
        return None
    data = report[EXTENSION_OFFSET:EXTENSION_OFFSET+6]
    if data[5] & 1 or list(data) in ([0]*6, [255]*6):
        return None
    return data


def axis(value, center=128, deadzone=7):
    delta = value - center
    if abs(delta) <= deadzone:
        return 0
    span = 128 - deadzone if delta < 0 else 127 - deadzone
    return max(-32767, min(32767, int((delta - math.copysign(deadzone, delta)) * 32767 / span)))


def decode_gyro(data):
    """Return yaw, pitch, roll ADC readings and their independent slow flags."""
    return ((data[0] | (data[3] & 0xFC) << 6,
             data[2] | (data[5] & 0xFC) << 6,
             data[1] | (data[4] & 0xFC) << 6),
            (bool(data[3] & 2), bool(data[3] & 1), bool(data[4] & 2)))


def decode_nunchuk(data):
    # MotionPlus passthrough moves C/Z to bits 3/2; stick bytes stay intact.
    return data[0], data[1], not bool(data[5] & 8), not bool(data[5] & 4)


class GyroSignal:
    """Robust per-axis baseline and median filtering before angle integration."""
    def __init__(self):
        self.bias = None
        self.samples = []
        self.filtered = (0.0, 0.0, 0.0)
        self.slow = None

    def update(self, raw, slow, dt):
        if any(v < 1000 or v > 15000 for v in raw):
            return None
        if self.slow != slow:
            self.samples.clear()
            self.filtered = (0.0, 0.0, 0.0)
            self.slow = slow
        self.samples.append(tuple(raw))
        self.samples = self.samples[-5:]
        if len(self.samples) < 5:
            return (0.0, 0.0, 0.0)
        med = tuple(statistics.median(c) for c in zip(*self.samples[-3:]))
        stable = all(max(c)-min(c) <= 30 for c in zip(*self.samples))
        if self.bias is None:
            if all(slow) and stable:
                self.bias = med
            return (0.0, 0.0, 0.0)
        rates = tuple((v-b)/13.768 * (1 if flag else 2000/440)
                      for v, b, flag in zip(med, self.bias, slow))
        # No single malformed sample or mode transition may produce a large jump.
        if any(abs(v) > 720 for v in rates):
            return None
        if all(slow) and stable and all(abs(v) < 1.5 for v in rates):
            self.bias = tuple(b + .02*(v-b) for b, v in zip(self.bias, med))
        rates = tuple(0.0 if abs(v) < 1.0 else v for v in rates)
        alpha = 1-math.exp(-max(0, min(dt, .02))/.006)
        self.filtered = tuple(old+alpha*(new-old) for old, new in zip(self.filtered, rates))
        return self.filtered


def multiply(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return (w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
            w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v)


class Aim:
    def __init__(self, full_angle=15.0, deadzone=0.35):
        self.full_angle = full_angle
        self.deadzone = deadzone
        self.center()

    def center(self):
        self.q = (1.0, 0.0, 0.0, 0.0)
        self.gravity_reference = None

    def update(self, yaw, pitch, roll, dt):
        # Body axes: X right, Y forward along the remote, Z up.
        vector = tuple(math.radians(v) * dt for v in (pitch, roll, -yaw))
        angle = math.sqrt(sum(v*v for v in vector))
        if angle:
            f = math.sin(angle / 2) / angle
            self.q = multiply(self.q, (math.cos(angle / 2), *(v*f for v in vector)))
            norm = math.sqrt(sum(v*v for v in self.q))
            self.q = tuple(v/norm for v in self.q)

    def correct_orientation(self, accelerometer, dt):
        """Dolphin-style sensor fusion and bounded yaw, rather than a rest timer.

        This is an independent adaptation for an Xbox stick, not Dolphin code.
        """
        if 0 < dt <= .1 and accelerometer is not None:
            vector = tuple(float(v)-128 for v in accelerometer)
            length = math.sqrt(sum(v*v for v in vector))
            if 8 < length < 80:
                if self.gravity_reference is None:
                    self.gravity_reference = (tuple(v/length for v in vector), length)
                reference, original_length = self.gravity_reference
                if .8*original_length < length < 1.2*original_length:
                    body = (0.0, *(v/length for v in vector))
                    conjugate = (self.q[0], *(-v for v in self.q[1:]))
                    world = multiply(multiply(self.q, body), conjugate)[1:]
                    cross = (world[1]*reference[2]-world[2]*reference[1],
                             world[2]*reference[0]-world[0]*reference[2],
                             world[0]*reference[1]-world[1]*reference[0])
                    size = math.sqrt(sum(v*v for v in cross))
                    if size > 1e-8:
                        dot = max(-1, min(1, sum(a*b for a,b in zip(world,reference))))
                        angle = math.atan2(size, dot)*(1-math.exp(-dt/.25))
                        correction = (math.cos(angle/2), *(v/size*math.sin(angle/2) for v in cross))
                        self.q = multiply(correction, self.q)
        # Bound accumulated horizontal rotation: moving back from an edge reacts immediately.
        w,x,y,z = self.q
        yaw = math.atan2(2*(x*y-w*z), 1-2*(x*x+z*z))
        limit = math.radians(self.full_angle)
        target = max(-limit, min(limit, yaw))
        if abs(yaw-target) > 1e-9:
            delta = yaw-target
            self.q = multiply((math.cos(delta/2),0,0,math.sin(delta/2)), self.q)
        norm = math.sqrt(sum(v*v for v in self.q))
        self.q = tuple(v/norm for v in self.q)

    def stick(self):
        w, x, y, z = self.q
        # Rotate the forward vector (0,1,0) into the centered frame.
        fx, fy, fz = 2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)
        angles = (math.degrees(math.atan2(fx, fy)),
                  math.degrees(math.atan2(fz, math.hypot(fx, fy))))
        def scale(a):
            if abs(a) <= self.deadzone:
                return 0
            value = (abs(a)-self.deadzone)/(self.full_angle-self.deadzone)
            return int(math.copysign(min(1, value)*32767, a))
        return tuple(scale(a) for a in angles)


def send(remote, report):
    if remote.write(report) <= 0:
        raise RuntimeError('Bluetooth HID write failed')


def write_register(remote, address, value):
    # 0x16: memory/register write, 4-byte address, byte count, 16-byte payload.
    payload = [0x16, *address.to_bytes(4, 'big'), 1, value, *([0]*15)]
    send(remote, payload)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        report = remote.read(64, 100)
        if report and report[0] == 0x22 and len(report) >= 5 and report[3] == 0x16:
            if report[4]:
                raise RuntimeError(f'MotionPlus register {address:08X}: error {report[4]}')
            return
    raise RuntimeError(f'MotionPlus register {address:08X}: acknowledgement timed out')


def initialize(remote):
    # Reset an already-active MotionPlus, then activate Nunchuk passthrough.
    write_register(remote, 0x04A400F0, 0x55)
    time.sleep(0.1)
    write_register(remote, 0x04A400FB, 0x00)
    write_register(remote, 0x04A600F0, 0x55)
    time.sleep(0.1)
    write_register(remote, 0x04A600FE, 0x05)
    time.sleep(0.2)
    send(remote, [0x12, 0x04, REPORT_MODE])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-angle', type=float, default=15, help='Degrees for full stick deflection')
    parser.add_argument('--deadzone', type=float, default=0.35, help='Aim deadzone in degrees')
    parser.add_argument('--yaw-sign', type=int, choices=(-1, 1), default=1)
    parser.add_argument('--pitch-sign', type=int, choices=(-1, 1), default=1)
    args = parser.parse_args()
    if not (0 <= args.deadzone < args.full_angle <= 90):
        parser.error('Require 0 <= deadzone < full-angle <= 90')
    import hid
    import vgamepad as vg
    from vgamepad.win.vigem_commons import XUSB_BUTTON as XB

    devices = [d for d in hid.enumerate(0x057E, 0) if d['product_id'] in (0x0306, 0x0330)]
    if not devices:
        raise RuntimeError('No connected Wii Remote found. Pair it in Windows first.')
    remote = hid.device()
    pad = None
    opened = False
    try:
        remote.open_path(devices[0]['path'])
        opened = True
        remote.set_nonblocking(False)
        initialize(remote)
        signal = GyroSignal()
        send(remote, [0x12, 0x04, REPORT_MODE])
        aim = Aim(args.full_angle, args.deadzone)
        pad = vg.VX360Gamepad()
        print('Ready (v2.8 gravity correction + yaw clamp). Nunchuk stick -> left stick; MotionPlus -> right stick.')
        print('1/2 unassigned. Automatic gravity correction and gyro calibration. Ctrl+C to stop.', flush=True)
        mapping = {0x0800: XB.XUSB_GAMEPAD_Y,
                   0x0200: XB.XUSB_GAMEPAD_B,
                   0x0400: XB.XUSB_GAMEPAD_A,
                   0x0100: XB.XUSB_GAMEPAD_X,
                   0x0010: XB.XUSB_GAMEPAD_BACK,
                   0x1000: XB.XUSB_GAMEPAD_START,
                   0x0080: XB.XUSB_GAMEPAD_GUIDE,
                   0x0008: XB.XUSB_GAMEPAD_RIGHT_SHOULDER}
        last_packet = last_gyro = time.monotonic()
        nunchuk_time = 0.0
        nunchuk = (128, 128, False, False)
        seen_gyro = False
        report_counts = {}
        last_recovery = last_packet
        while True:
            report = remote.read(64, 100)
            now = time.monotonic()
            if now - last_gyro > 0.3:
                aim.center()
            if now - last_packet > 0.3:
                pad.reset()
                pad.update()
            if now - last_gyro > 3 and now - last_recovery > 3:
                last_recovery = now
                pad.reset()
                pad.update()
                aim.center()
                print(f'Waiting for MotionPlus; retrying stream. Reports: {report_counts}', flush=True)
                try:
                    initialize(remote)
                except (OSError, RuntimeError) as exc:
                    print(f'Stream recovery: {exc}', flush=True)
                last_gyro = time.monotonic()
                seen_gyro = False
                signal = GyroSignal()
            if report:
                report_counts[hex(report[0])] = report_counts.get(hex(report[0]), 0) + 1
                # Extension changes can reset the reporting mode after activation.
                if report[0] == 0x20 or report[0] in (0x30, 0x31):
                    send(remote, [0x12, 0x04, REPORT_MODE])
            data = extension_payload(report)
            if not report or report[0] != REPORT_MODE or len(report) < 3:
                continue
            last_packet = now
            buttons = ((report[1] << 8) | report[2]) & 0x1F9F
            if data is not None and data[5] & 2:
                if not data[4] & 1:
                    nunchuk = (128, 128, False, False)
                    nunchuk_time = 0
                raw, slow = decode_gyro(data)
                dt = now - last_gyro
                rates = signal.update(raw, slow, dt)
                if rates is not None:
                    last_gyro = now
                    first_gyro = not seen_gyro
                    seen_gyro = True
                    if first_gyro or dt > 0.1:
                        aim.center()
                    else:
                        aim.update(rates[0]*args.yaw_sign, rates[1]*args.pitch_sign,
                                   rates[2], min(dt, .02))
                        aim.correct_orientation(report[3:6] if len(report) >= 6 else None, dt)
            elif data is not None:
                nunchuk = decode_nunchuk(data)
                nunchuk_time = now
            pad.reset()
            for mask, target in mapping.items():
                if buttons & mask:
                    pad.press_button(target)
            pad.right_trigger(255 if buttons & 4 else 0)
            if now - nunchuk_time < 0.3:
                sx, sy, c, z = nunchuk
                pad.left_joystick(axis(sx), axis(sy))
                if c:
                    pad.press_button(XB.XUSB_GAMEPAD_LEFT_SHOULDER)
                pad.left_trigger(255 if z else 0)
            rx, ry = aim.stick()
            pad.right_joystick(-rx, -ry)
            pad.update()
    except KeyboardInterrupt:
        return 0
    finally:
        if pad is not None:
            pad.reset()
            pad.update()
        if opened:
            try:
                write_register(remote, 0x04A400F0, 0x55)
            except Exception:
                pass
            finally:
                remote.close()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'Bridge error: {exc}', file=sys.stderr)
        raise SystemExit(1)
