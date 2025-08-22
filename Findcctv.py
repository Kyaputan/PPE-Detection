import cv2
import time

def test_rtsp_connection(User, Password, ip):
    """Test RTSP connection with various methods"""
    
    base_url = f"rtsp://{User}:{Password}@{ip}"
    rtsp_urls = [
        # f"{base_url}/stream1",
        f"{base_url}:554/stream1", 
        f"{base_url}/live/stream1",
        f"{base_url}/h264",
        f"{base_url}/cam/realmonitor?channel=1&subtype=0",
        f"{base_url}/onvif1",
        f"{base_url}/Streaming/Channels/101",
        f"{base_url}/videoMain"
    ]
    
    print("Testing RTSP connections...")
    print("=" * 50)
    
    for i, url in enumerate(rtsp_urls, 1):
        print(f"\n{i}. Testing: {url}")
        
        # Test with different backends
        backends = [
            (cv2.CAP_FFMPEG, "FFMPEG"),
            (cv2.CAP_GSTREAMER, "GStreamer"), 
            (cv2.CAP_ANY, "Any")
        ]
        
        for backend, backend_name in backends:
            try:
                print(f"   Trying {backend_name} backend...")
                cap = cv2.VideoCapture(url, backend)
                
                # Set properties
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
                
                if cap.isOpened():
                    print(f"   ✓ Connected with {backend_name}!")
                    
                    # Try to read a frame
                    ret, frame = cap.read()
                    if ret:
                        print(f"   ✓ Successfully read frame: {frame.shape}")
                        
                        # Show properties
                        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        print(f"   Resolution: {int(width)}x{int(height)}")
                        print(f"   FPS: {fps}")
                        
                        # Test reading multiple frames
                        print("   Testing frame reading...")
                        success_count = 0
                        for j in range(10):
                            ret, frame = cap.read()
                            if ret:
                                success_count += 1
                            time.sleep(0.1)
                        
                        print(f"   Successfully read {success_count}/10 frames")
                        
                        cap.release()
                        return url, backend_name  
                    else:
                        print(f"   ✗ Connected but couldn't read frames")
                else:
                    print(f"   ✗ Failed to connect with {backend_name}")
                
                cap.release()
                
            except Exception as e:
                print(f"   ✗ Error with {backend_name}: {e}")
    
    print("\n" + "=" * 50)
    print("No successful RTSP connections found")
    return None, None

def test_with_vlc_method():
    """Test using VLC-like parameters"""
    print("\nTrying VLC-compatible method...")
    
    rtsp_url = "rtsp://comvision:9fktgh_J9H@10.41.171.89/stream1"
    
    # Create VideoCapture with specific options
    cap = cv2.VideoCapture()
    
    # Set options before opening (similar to VLC)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Try to open
    success = cap.open(rtsp_url, cv2.CAP_FFMPEG)
    
    if success:
        print("✓ VLC-method connection successful!")
        
        # Test reading
        for i in range(5):
            ret, frame = cap.read()
            if ret:
                print(f"✓ Frame {i+1} read successfully: {frame.shape}")
                cv2.imshow('RTSP Test', frame)
                cv2.waitKey(100)
            else:
                print(f"✗ Failed to read frame {i+1}")
        
        cv2.destroyAllWindows()
        cap.release()
        return True
    else:
        print("✗ VLC-method failed")
        cap.release()
        return False

if __name__ == "__main__":
    User = "root01"
    Password = "12345678"
    ip = "192.168.1.102"
    successful_url, backend = test_rtsp_connection(User, Password, ip)
    
    if successful_url:
        print(f"\n🎉 SUCCESS! Use this configuration:")
        print(f"URL: {successful_url}")
        print(f"Backend: {backend}")
        print("\nExample code:")
        print(f"cap = cv2.VideoCapture('{successful_url}', cv2.CAP_FFMPEG)")
    else:
        # Try VLC method
        vlc_success = test_with_vlc_method()
        
        if not vlc_success:
            print("\nTroubleshooting suggestions:")
            print("1. Check if the camera is accessible from your network")
            print("2. Verify username/password")
            print("3. Try different stream paths (/stream2, /live, etc.)")
            print("4. Check if OpenCV was compiled with FFMPEG support")
            print("5. Try installing additional codecs")
            
            # Network test
            import socket
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(3)
                result = sock.connect_ex((ip, 554))
                sock.close()
                if result == 0:
                    print("✓ Port 554 (RTSP) is accessible")
                else:
                    print("✗ Port 554 (RTSP) is not accessible")
            except:
                print("✗ Network connectivity issue")