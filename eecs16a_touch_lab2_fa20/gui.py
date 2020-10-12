#!/usr/bin/python
from __future__ import unicode_literals
import sys
import argparse
import numpy as np
from tkinter import *
import time
import glob
import serial


########
# COMS #
########

BAUD_RATE = 115200

def serial_ports():
  """Lists serial ports

  Raises:
  EnvironmentError:
      On unsupported or unknown platforms
  Returns:
      A list of available serial ports
  """
  if sys.platform.startswith('win'):
      ports = ['COM' + str(i + 1) for i in range(256)]

  elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
      # this is to exclude your current terminal "/dev/tty"
      ports = glob.glob('/dev/tty[A-Za-z]*')

  elif sys.platform.startswith('darwin'):
      ports = glob.glob('/dev/tty.*')

  else:
      raise EnvironmentError('Unsupported platform')

  result = []
  for port in ports:
      try:
          s = serial.Serial(port)
          s.close()
          result.append(port)
      except (OSError, serial.SerialException):
          pass
  return result

def serial_write(ser, data):
  if sys.platform.startswith('win'):
    ser.write([data, ])
  else:
    ser.write(data)

def parse_serial(ser):
    """
    Extract the raw x and y values from Serial.
    Args:
        ser: the serial port object.
    Returns:
        tuple(int x, int y): the raw x and the raw y values read in by the Launchpad
    """
    try:
        data = ser.readline().decode().split()
        if len(data) == 6 and data[0] == 'X' and data[3] == 'Y':
            return (int(float(data[2])), int(float(data[5])))
    except Exception:
        ser.close()

def readSensorData(ser, get_loc):
    # Tell MSP to collect data
    serial_write(ser, b'6')
    # parse to readable format
    serial_data = parse_serial(ser)
    # convert raaw values to location
    if (serial_data[0] == -1 or serial_data[1] == -1):
        loc = (-1, -1)
    else:
        loc = get_loc(serial_data[0], serial_data[1])
    return "(" + str(loc[0]) + "," + str(loc[1]) + ")"

def get_ser_port():
    print("\nEE16A Touchscreen 2 Lab\n")

    print("Checking serial connections...")

    ports = serial_ports()
    if ports:
        print("Available serial ports:")
        for (i, p) in enumerate(ports):
            print("%d) %s" % (i + 1, p))
    else:
        print("No ports available. Check serial connection and try again.")
        print("Exiting...")
        sys.exit()

    selection = input("Select the port to use: ")
    ser = serial.Serial(ports[int(selection) - 1], BAUD_RATE, timeout = 150)
    return ser

def handshake(y, get_loc):
    ser = get_ser_port()
    print("\nPING MSP with data request.\n")
    i = 0
    for _ in range (0,y):
        x = str(readSensorData(ser, get_loc))
        print("Received " + x + " from MSP")
        if (x[0] == "("):
            i += 1;
        time.sleep(1)
    ser.close()
    print("\n", y, "packets requested,", i, "received,", (5-i)/y, "% packet loss.")

def stream(get_loc):
    ser = get_ser_port()
    time.sleep(0.5)
    print("\nStart Touching Points!\n")
    while True:
        try:
            time.sleep(0.1)
            x = str(readSensorData(ser, get_loc))
            if (x != "(-1,-1)"):
                print("Touched: " + x)
        except KeyboardInterrupt:
            ser.close()
            break

def launch_gui(get_loc):

    ser = get_ser_port()

    #######
    # GUI #
    #######

    on  = "#00ff00"
    off = "#a6a6a6"
    color = [off, off, off, off, off, off, off, off, off, off]
    locs = { "(0,0)":0, "(1,0)":1, "(2,0)":2, "(0,1)":3, "(1,1)":4, "(2,1)":5, "(0,2)":6, "(1,2)":7, "(2,2)":8, "(-1,-1)":9}

    # TKINTER ROOT
    root = Tk()
    root.minsize(200,50)

    # DRAW FRAMES
    toprow = Frame(root)
    toprow.pack(side=BOTTOM)

    midrow = Frame(root)
    midrow.pack(side=BOTTOM)

    botrow = Frame(root)
    botrow.pack(side=BOTTOM)

    # TOP ROW
    pixel00 = Frame(toprow)
    pixel00.pack(side=LEFT)
    button00 = Button(pixel00, text="0,0", fg="grey", bg=color[0], state=DISABLED)
    button00.pack()

    pixel01 = Frame(toprow)
    pixel01.pack(side=LEFT)
    button01 = Button(pixel01, text="1,0", fg="grey", bg=color[1], state=DISABLED)
    button01.pack()

    pixel02 = Frame(toprow)
    pixel02.pack(side=LEFT)
    button02 = Button(pixel02, text="2,0", fg="grey", bg=color[2], state=DISABLED)
    button02.pack()

    # MID ROW
    pixel10 = Frame(midrow)
    pixel10.pack(side=LEFT)
    button10 = Button(pixel10, text="0,1", fg="green", bg=color[3], state=DISABLED)
    button10.pack()

    pixel11 = Frame(midrow)
    pixel11.pack(side=LEFT)
    button11 = Button(pixel11, text="1,1", fg="green", bg=color[4], state=DISABLED)
    button11.pack()

    pixel12 = Frame(midrow)
    pixel12.pack(side=LEFT)
    button12 = Button(pixel12, text="2,1", fg="grey", bg=color[5], state=DISABLED)
    button12.pack()

    # BOT ROW
    pixel20 = Frame(botrow)
    pixel20.pack(side=LEFT)
    button20 = Button(pixel20, text="0,2", fg="grey", bg=color[6], state=DISABLED)
    button20.pack()

    pixel21 = Frame(botrow)
    pixel21.pack(side=LEFT)
    button21 = Button(pixel21, text="1,2", fg="grey", bg=color[7], state=DISABLED)
    button21.pack()

    pixel22 = Frame(botrow)
    pixel22.pack(side=LEFT)
    button22 = Button(pixel22, text="2,2", fg="grey", bg=color[8], state=DISABLED)
    button22.pack()

    i = 0
    while True:
        try:
            time.sleep(0.1)
            color = [off, off, off, off, off, off, off, off, off, off]
            index = locs[readSensorData(ser, get_loc)]
            color[index] = on
            button00.configure(bg=color[0])
            button01.configure(bg=color[1])
            button02.configure(bg=color[2])
            button10.configure(bg=color[3])
            button11.configure(bg=color[4])
            button12.configure(bg=color[5])
            button20.configure(bg=color[6])
            button21.configure(bg=color[7])
            button22.configure(bg=color[8])
            root.update()
            root.update_idletasks()
        except (KeyboardInterrupt, TclError):
            ser.close()
            break
