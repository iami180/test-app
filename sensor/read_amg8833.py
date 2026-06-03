"""Read an 8x8 thermal frame from an AMG8833 sensor (MicroPython / ESP32).

This runs on the microcontroller, not the training host. It streams CSV
lines over the serial console: 64 comma-separated temperatures (degC) per
frame, one frame per line. Capture them on the host with e.g.

    mpremote run sensor/read_amg8833.py > data/raw/run01.csv

Wiring (I2C):
    AMG8833 VIN -> 3V3
    AMG8833 GND -> GND
    AMG8833 SDA -> GPIO21 (default)
    AMG8833 SCL -> GPIO22 (default)

The AMG8833 returns 12-bit pixel values in 0.25 degC units.
"""

from machine import I2C, Pin  # type: ignore
import time
import struct

AMG8833_ADDR = 0x69      # 0x68 if AD_SELECT is tied low
PIXEL_BASE = 0x80        # first pixel low byte register
PCTL_REG = 0x00          # power control
RST_REG = 0x01           # reset


def _init(i2c: I2C) -> None:
    i2c.writeto_mem(AMG8833_ADDR, PCTL_REG, b"\x00")  # normal mode
    time.sleep_ms(50)
    i2c.writeto_mem(AMG8833_ADDR, RST_REG, b"\x3f")   # initial reset
    time.sleep_ms(50)


def read_frame(i2c: I2C) -> list:
    """Return 64 temperatures (degC) as a flat list in row-major order."""
    raw = i2c.readfrom_mem(AMG8833_ADDR, PIXEL_BASE, 128)  # 64 * 2 bytes
    temps = []
    for i in range(0, 128, 2):
        value = struct.unpack_from("<h", raw, i)[0]
        # 12-bit two's complement, 0.25 degC per LSB.
        if value & 0x800:
            value -= 0x1000
        temps.append(value * 0.25)
    return temps


def main() -> None:
    i2c = I2C(0, scl=Pin(22), sda=Pin(21), freq=100_000)
    _init(i2c)
    # AMG8833 refreshes at 10 Hz; sample a little slower to be safe.
    while True:
        frame = read_frame(i2c)
        print(",".join(f"{t:.2f}" for t in frame))
        time.sleep_ms(200)


if __name__ == "__main__":
    main()
