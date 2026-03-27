import os
import PySpin
import cv2  # Replaced matplotlib.pyplot
import sys
import time

# Note: 'keyboard' module is less necessary now as cv2 has its own key handler
global continue_recording
continue_recording = True

def acquire_and_display_images(cam, nodemap, nodemap_tldevice):
    global continue_recording

    sNodemap = cam.GetTLStreamNodeMap()

    # Set buffer handling to NewestOnly to ensure we see the live feed without lag
    node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
    if PySpin.IsAvailable(node_bufferhandling_mode) and PySpin.IsWritable(node_bufferhandling_mode):
        node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
        node_bufferhandling_mode.SetIntValue(node_newestonly.GetValue())

    print('*** IMAGE ACQUISITION ***\n')
    try:
        # Set Acquisition Mode to Continuous
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        node_acquisition_mode.SetIntValue(node_acquisition_mode_continuous.GetValue())

        cam.BeginAcquisition()
        print('Acquiring images...')
        print('Press "q" or "Esc" in the window to stop.')

        # Create a named window that can be resized
        cv2.namedWindow('FLIR Camera Feed', cv2.WINDOW_NORMAL)

        while continue_recording:
            try:
                image_result = cam.GetNextImage(1000) # 1000ms timeout

                if image_result.IsIncomplete():
                    print('Image incomplete...')
                else:
                    # Convert PySpin image to numpy array for OpenCV
                    image_data = image_result.GetNDArray()

                    # Display the image
                    cv2.imshow('FLIR Camera Feed', image_data)

                    # cv2.waitKey(1) handles the window refresh and checks for key presses
                    # 1ms delay is standard for live feeds
                    key = cv2.waitKey(1)
                    if key == ord('q') or key == 27:  # 'q' or Esc key
                        print('Closing...')
                        continue_recording = False

                image_result.Release()

            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                break

        # Cleanup
        cv2.destroyAllWindows()
        cam.EndAcquisition()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return True

def run_single_camera(cam):
    try:
        cam.Init()
        nodemap = cam.GetNodeMap()
        nodemap_tldevice = cam.GetTLDeviceNodeMap()
        
        result = acquire_and_display_images(cam, nodemap, nodemap_tldevice)
        
        cam.DeInit()
        return result
    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

def main():
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    num_cameras = cam_list.GetSize()

    if num_cameras == 0:
        cam_list.Clear()
        system.ReleaseInstance()
        print('No cameras detected.')
        return False

    for i, cam in enumerate(cam_list):
        run_single_camera(cam)

    del cam
    cam_list.Clear()
    system.ReleaseInstance()
    return True

if __name__ == '__main__':
    main()