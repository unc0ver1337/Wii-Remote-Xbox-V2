import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('bridge', Path(__file__).with_name('wii_xbox_bridge_v2.py'))
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def gyro(yaw=8063, pitch=8063, roll=8063):
    return [yaw & 255, roll & 255, pitch & 255,
            ((yaw >> 8) << 2) | 3, ((roll >> 8) << 2) | 3,
            ((pitch >> 8) << 2) | 2]


class Tests(unittest.TestCase):
    def test_reject_ordinary_nunchuk_when_c_changes(self):
        # Captured legacy packet: C occupies the bit previously mistaken for gyro.
        for last in (0x27, 0x25, 0x8b, 0x89):
            data = [0x80, 0x80, 0xbc, 0x88, 0x94, last]
            self.assertIsNone(b.extension_payload([0x34, 0, 0, *data]))
            self.assertIsNone(b.extension_payload([0x37, 0, 0, *([0]*13), *data]))
        for last in (0, 4, 8, 12):
            data = [128, 128, 80, 90, 101, last]
            self.assertEqual(b.extension_payload([0x37, 0, 0, *([0]*13), *data]), data)

    def test_signal_rejects_spikes_and_centers(self):
        signal = b.GyroSignal()
        for _ in range(10):
            self.assertEqual(signal.update((8057,8227,8124), (True,)*3, .008), (0,0,0))
        self.assertIsNone(signal.update((0,16383,0), (False,)*3, .008))
        signal.update((10000,8227,8124), (True,)*3, .008)
        self.assertEqual(signal.filtered, (0,0,0))
        for _ in range(12):
            rates = signal.update((8500,8227,8124), (True,)*3, .008)
        self.assertGreater(rates[0], 0)
        # A new steady bias is learned automatically, without button input.
        signal = b.GyroSignal()
        for _ in range(10):
            signal.update((8100,8250,8150), (True,)*3, .008)
        old_bias = signal.bias
        for _ in range(250):
            signal.update((8105,8255,8155), (True,)*3, .008)
        self.assertGreater(signal.bias[0], old_bias[0])
        self.assertLess(abs(signal.bias[0]-8105), 1)

    def test_decode(self):
        self.assertEqual(b.decode_gyro(gyro(9000, 7500, 8200)),
                         ((9000, 7500, 8200), (True, True, True)))
        packet = gyro(); packet[3] &= ~2
        self.assertEqual(b.decode_gyro(packet)[1], (False, True, True))
        self.assertEqual(b.decode_nunchuk([255, 0, 0, 0, 1, 12]), (255, 0, False, False))
        self.assertEqual(b.decode_nunchuk([128, 128, 0, 0, 1, 0]), (128, 128, True, True))

    def test_gravity_correction_and_yaw_clamp(self):
        aim = b.Aim()
        aim.correct_orientation((128,128,154), .01)
        aim.update(0,10,0,1)
        before = abs(aim.stick()[1])
        for _ in range(200):
            aim.correct_orientation((128,128,154), .01)
        self.assertLess(abs(aim.stick()[1]), before)
        self.assertEqual(aim.stick()[1], 0)
        aim.update(100,0,0,1)
        aim.correct_orientation(None, .01)
        at_edge = aim.stick()[0]
        self.assertAlmostEqual(at_edge,32767,delta=1)
        aim.update(-5,0,0,1)
        aim.correct_orientation(None,.01)
        self.assertLess(aim.stick()[0], at_edge)

    def test_aim(self):
        for sign in (-1, 1):
            a = b.Aim()
            a.update(sign*30, 0, 0, 1)
            self.assertAlmostEqual(a.stick()[0], sign*32767, delta=1)
            a.update(-sign*30, 0, 0, 1)
            self.assertEqual(a.stick(), (0, 0))
            a.update(0, sign*30, 0, 1)
            self.assertAlmostEqual(a.stick()[1], sign*32767, delta=1)
            a.center()
            self.assertEqual(a.stick(), (0, 0))
        a = b.Aim(); a.update(0, 0, 90, 1)
        self.assertEqual(a.stick(), (0, 0))

    def test_nunchuk_normal(self):
        self.assertEqual((b.axis(0), b.axis(128), b.axis(255)), (-32767, 0, 32767))

    def test_register_ack(self):
        sent = []
        remote = types.SimpleNamespace(write=lambda p: sent.append(p) or len(p),
                                       read=lambda *a: [0x22, 0, 0, 0x16, 0])
        b.write_register(remote, 0x04A600FE, 5)
        self.assertEqual(sent[0][:7], [0x16, 4, 166, 0, 254, 1, 5])
        self.assertEqual(len(sent[0]), 22)
        remote.read = lambda *a: [0x22, 0, 0, 0x16, 7]
        with self.assertRaises(RuntimeError):
            b.write_register(remote, 0x04A600FE, 5)

    def test_stream_and_center_chord(self):
        frames = [(0, [255,255,0,0,1,0]), (0, gyro()), (0, gyro(12000)),
                  (1, gyro()), (0, gyro()), (0, gyro()),
                  (2, gyro()), (0, gyro()), (0x0800, gyro()),
                  (0x0200, gyro()), (0x0400, gyro()), (0x0100, gyro()),
                  (0x0010, gyro()), (0x1000, gyro()), (0x0080, gyro())]
        # A sustained C hold and missing gyro must not produce aiming or exit.
        frames.extend((0, [128,128,80,90,101,4]) for _ in range(400))
        class Remote:
            def open_path(self, p): pass
            def set_nonblocking(self, v): pass
            def close(self): pass
            def read(self, *args):
                if not frames: raise KeyboardInterrupt
                buttons, data = frames.pop(0)
                return [b.REPORT_MODE, buttons >> 8, buttons & 255, *([0]*13), *data]
        class Pad:
            def __init__(self): self.frames=[]
            def reset(self): self.state={'buttons':set(), 'left':(0,0),'right':(0,0)}
            def press_button(self, v): self.state['buttons'].add(v)
            def left_joystick(self, *v): self.state['left']=v
            def right_joystick(self, *v): self.state['right']=v
            def left_trigger(self,v): self.state['lt']=v
            def right_trigger(self,v): self.state['rt']=v
            def update(self): self.frames.append(self.state.copy())
        pad=Pad()
        buttons=types.SimpleNamespace(**{f'XUSB_GAMEPAD_{n}':n for n in
            ['RIGHT_SHOULDER','LEFT_SHOULDER','A','B','X','Y','START','BACK','GUIDE']})
        modules={'hid':types.SimpleNamespace(enumerate=lambda *a:[{'product_id':0x0330,'path':b'test'}],device=Remote),
                 'vgamepad':types.SimpleNamespace(VX360Gamepad=lambda:pad),
                 'vgamepad.win.vigem_commons':types.SimpleNamespace(XUSB_BUTTON=buttons)}
        now=[0]
        def clock(): now[0]+=.01; return now[0]
        with patch.dict(sys.modules, modules), patch.object(sys,'argv',['bridge']), \
             patch.object(b,'initialize'), patch.object(b,'write_register'), \
             patch.object(b,'GyroSignal', return_value=types.SimpleNamespace(update=lambda raw, slow, dt: tuple((v-8063)/13.768 for v in raw))), \
             patch.object(b,'send'), \
             patch.object(b.time,'monotonic',side_effect=clock):
            b.main()
        self.assertEqual(pad.frames[0]['left'], (32767,32767))
        self.assertLess(pad.frames[2]['right'][0], 0)
        for i in (3,6):
            self.assertFalse({'START','BACK'} & pad.frames[i]['buttons'])
        self.assertLess(pad.frames[3]['right'][0],0)
        for i, button in enumerate(('Y','B','A','X','BACK','START','GUIDE'), start=8):
            self.assertIn(button, pad.frames[i]['buttons'])
        self.assertEqual(pad.frames[-1]['left'], (0,0))
        self.assertIn('LEFT_SHOULDER', pad.frames[20]['buttons'])
        self.assertEqual(pad.frames[-1]['right'], (0,0))

if __name__ == '__main__':
    unittest.main()
