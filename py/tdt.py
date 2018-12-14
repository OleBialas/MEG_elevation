# -*- coding: utf-8 -*-
"""
eCreated on Mon Jun 12 12:20:32 2017
@author: ob56dap

TDT ActiveX black box
"""
import win32com.client


def initialize_processor(processor=None, connection=None, index=None, path=None):
    # Create object to handle TDT-functions
    try:
        RP = win32com.client.Dispatch('RPco.X')
    except win32com.client.pythoncom.com_error as e:
        print("Error:", e)
        return -1
    print("Successfully initialized TDT ActiveX interface")

    # Get the parameters
    if not processor:
        processor = input("what type of device do you want to connect to")
    if not connection:
        connection = input("is the device connected via 'USB' or 'GB'?")
    if not index:
        index = int(input("whats the index of the device (starting with 1)?"))
    if not path:
        path = input("full path of the circuit you want to load")

    # Connect to the device
    if processor == "RM1":
        if RP.ConnectRM1(connection, index):
            print("Connected to RM1")
    elif processor == "RP2":
        if RP.ConnectRP2(connection, index):
            print("Connected to RP2")
    elif processor == "RX8":
        if RP.ConnectRX8(connection, index):
            print("Connected to RX8")
    else:
        print("Error: unknown device!")
        return -1

    if not RP.ClearCOF():
        print("ClearCOF failed")
        return -1

    if RP.LoadCOF(path):
        print("Circuit {0} loaded".format(path))
    else:
        print("Failed to load {0}".format(path))
        return -1

    if RP.Run():
        print("Circuit running")
    else:
        print("Failed to run {0}".format(path))
        return -1
    return RP

def initialize_zbus(connection):

    try:
        ZB = win32com.client.Dispatch('ZBUS.x')
    except win32com.client.pythoncom.com_error as e:
        print("Error:", e)
        return -1
    print("Successfully initialized ZBus")

    if ZB.ConnectZBUS(connection):
        print("Connected to ZBUS")
    else:
        print("failed to connect to ZBUS")

    return ZB