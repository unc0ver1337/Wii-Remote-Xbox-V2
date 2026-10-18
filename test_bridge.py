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
    def test_decode(self):
        self.assertEqual(b.decode_gyro(gyro(9000, 7500, 8200)),
                         ((9000, 7500, 8200), (True, True, True)))
        packet = gyro(); packet[3] &= ~2
        self.assertEqual(b.decode_gyro(packet)[1], (False, True, True))
        self.assertEqual(b.decode_nunchuk([255, 0, 0, 0, 1, 12]), (255, 0, False, False))
        self.assertEqual(b.decode_nunchuk([128, 128, 0, 0, 1, 0]), (128, 128, True, True))

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

    def test_factory_offsets_without_stationary_sampling(self):
        raw = b''.join(int(v*4).to_bytes(2, 'big') for v in (8000, 8100, 8200))
        reports = [[0x20, 0, 0], [0x21, 0, 0, 0x50, 0, 0x30, *raw]]
        sent = []
        remote = types.SimpleNamespace(write=lambda p: sent.append(p) or len(p),
                                       read=lambda *a: reports.pop(0))
        self.assertEqual(b.read_factory_bias(remote), (8000, 8200, 8100))
        self.assertIn([0x12, 0x04, 0x34], sent)
        remote.read = lambda *a: [0x21, 0, 0, 7, 0, 0x30]
        self.assertEqual(b.read_factory_bias(remote), (8063,)*3)

    def test_stream_and_center_chord(self):
        frames = [(0, [255,255,0,0,1,0]), (0, gyro()), (0, gyro(12000)),
                  (3, gyro(12000)), (2, gyro()), (0, gyro()),
                  (2, gyro()), (0, gyro())]
        class Remote:
            def open_path(self, p): pass
            def set_nonblocking(self, v): pass
            def close(self): pass
            def read(self, *args):
                if not frames: raise KeyboardInterrupt
                buttons, data = frames.pop(0)
                return [0x34, buttons >> 8, buttons & 255, *data]
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
            ['RIGHT_SHOULDER','LEFT_SHOULDER','X','Y','START','BACK']})
        modules={'hid':types.SimpleNamespace(enumerate=lambda *a:[{'product_id':0x0330,'path':b'test'}],device=Remote),
                 'vgamepad':types.SimpleNamespace(VX360Gamepad=lambda:pad),
                 'vgamepad.win.vigem_commons':types.SimpleNamespace(XUSB_BUTTON=buttons)}
        now=[0]
        def clock(): now[0]+=.01; return now[0]
        with patch.dict(sys.modules, modules), patch.object(sys,'argv',['bridge']), \
             patch.object(b,'initialize'), patch.object(b,'write_register'), \
             patch.object(b,'read_factory_bias',return_value=(8063,)*3), patch.object(b,'send'), \
             patch.object(b.time,'monotonic',side_effect=clock):
            b.main()
        self.assertEqual(pad.frames[0]['left'], (32767,32767))
        self.assertGreater(pad.frames[2]['right'][0], 0)
        for i in (3,4,5):
            self.assertEqual(pad.frames[i]['right'], (0,0))
            self.assertFalse({'START','BACK'} & pad.frames[i]['buttons'])
        self.assertIn('START', pad.frames[7]['buttons'])
        self.assertEqual(pad.frames[-1]['left'], (0,0))

if __name__ == '__main__':
    unittest.main()
