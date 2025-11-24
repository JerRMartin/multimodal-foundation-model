import XInput
import time


class XboxController:
    def __init__(self, controller_id=0):
        self.id = controller_id

        # Internal state
        self.state = XInput.get_state(self.id)
        self.buttons = {}
        self.left_stick = (0.0, 0.0)
        self.right_stick = (0.0, 0.0)
        self.left_trigger = 0.0
        self.right_trigger = 0.0

        print(f"Xbox Controller #{self.id} initialized.")

    # ---------------------------------------------------------
    # RUMBLE
    # ---------------------------------------------------------
    def rumble(self, left_strength=1.0, right_strength=1.0, duration=0.001):
        left = int(max(0, min(1, left_strength)) * 65535)
        right = int(max(0, min(1, right_strength)) * 65535)

        XInput.set_vibration(self.id, left, right)
        time.sleep(duration)
        XInput.set_vibration(self.id, 0, 0)

    # ---------------------------------------------------------
    # POLLING LOOP (call every frame)
    # ---------------------------------------------------------
    def poll(self):
        self.state = XInput.get_state(self.id)
        # self._process_buttons()
        # self._process_sticks()
        # self._process_triggers()

    # ---------------------------------------------------------
    # RAW INPUT PROCESSING
    # ---------------------------------------------------------
    # def _process_buttons(self):
    #     s = self.state

    #     self.buttons = {
            # "A": bool(s.Gamepad.wButtons & XInput.BUTTON_A),
            # "B": bool(s.Gamepad.wButtons & XInput.BUTTON_B),
            # "X": bool(s.Gamepad.wButtons & XInput.BUTTON_X),
            # "Y": bool(s.Gamepad.wButtons & XInput.BUTTON_Y),
            # "LB": bool(s.Gamepad.wButtons & XInput.BUTTON_LEFT_SHOULDER ),
            # "RB": bool(s.Gamepad.wButtons & XInput.BUTTON_RIGHT_SHOULDER),
            # "BACK": bool(s.Gamepad.wButtons & XInput.BUTTON_BACK),
            # "START": bool(s.Gamepad.wButtons & XInput.BUTTON_START),
            # "LS": bool(s.Gamepad.wButtons & XInput.STICK_LEFT),
            # "RS": bool(s.Gamepad.wButtons & XInput.STICK_RIGHT),
            # "DPAD_UP": bool(s.Gamepad.wButtons & XInput.BUTTON_DPAD_UP),
            # "DPAD_DOWN": bool(s.Gamepad.wButtons & XInput.BUTTON_DPAD_DOWN),
            # "DPAD_LEFT": bool(s.Gamepad.wButtons & XInput.BUTTON_DPAD_LEFT),
            # "DPAD_RIGHT": bool(s.Gamepad.wButtons & XInput.BUTTON_DPAD_RIGHT),
        # }

    # def _normalize(self, value, deadzone=7849, max_value=32767):
    #     if abs(value) < deadzone:
    #         return 0.0
    #     return value / max_value

    # def _process_sticks(self):
    #     s = self.state.Gamepad

    #     lx = self._normalize(s.sThumbLX)
    #     ly = self._normalize(s.sThumbLY)
    #     rx = self._normalize(s.sThumbRX)
    #     ry = self._normalize(s.sThumbRY)

    #     # Flip Y to match typical game coordinates
    #     self.left_stick = (lx, -ly)
    #     self.right_stick = (rx, -ry)

    # def _process_triggers(self):
    #     s = self.state.Gamepad
    #     self.left_trigger = s.bLeftTrigger / 255.0
    #     self.right_trigger = s.bRightTrigger / 255.0

    # ---------------------------------------------------------
    # PUBLIC METHODS
    # ---------------------------------------------------------
    # def get_button(self, name):
    #     return self.buttons.get(name, False)

    # def get_left_stick(self):
    #     return self.left_stick

    # def get_right_stick(self):
    #     return self.right_stick

    # def get_triggers(self):
    #     return (self.left_trigger, self.right_trigger)
