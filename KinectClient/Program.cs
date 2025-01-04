using System;
using System.IO;
using System.Net.Sockets;
using Microsoft.Kinect;
using System.Windows.Media;
using System.Windows.Media.Imaging;

class KinectClient
{
    private const string ServerAddress = "127.0.0.1";
    private const int ServerPort = 5000;
    private Socket clientSocket;

    private KinectSensor kinectSensor;

    static void Main(string[] args)
    {
        KinectClient client = new KinectClient();
        client.Start();
    }

    public void Start()
    {
        InitializeKinect();
        InitializeKinect_ColorFrame();

        kinectSensor.AllFramesReady += ColorFrameReady;

        ConnectToServer();

        // Keep the application running
        Console.WriteLine("Press any key to exit...");
        Console.ReadKey();

        // Cleanup
        kinectSensor.Stop();
        clientSocket.Close();
    }

    private void InitializeKinect()
    {
        try
        {
            kinectSensor = KinectSensor.KinectSensors[0];
            kinectSensor.Start();
        }
        catch (Exception ex)
        {
            Console.WriteLine("Error initializing or starting Kinect: " + ex.Message);
            kinectSensor.Stop();
            clientSocket.Close();
            Environment.Exit(1);
        }
    }

    private void InitializeKinect_ColorFrame()
    {
        try 
        { 
            kinectSensor.ColorStream.Enable(ColorImageFormat.RgbResolution640x480Fps30); 
        }
        catch (Exception ex)
        {
            Console.WriteLine("Error enabling color stream: " + ex.Message);
            kinectSensor.Stop();
            clientSocket.Close();
            Environment.Exit(1);
        }
    }

    private void InitializeKinect_DepthFrame()
    {
        try
        {
            kinectSensor.DepthStream.Enable(DepthImageFormat.Resolution640x480Fps30);

        }
        catch (Exception ex)
        {
            Console.WriteLine("Error enabling depth stream: " + ex.Message);
            kinectSensor.Stop();
            clientSocket.Close();
            Environment.Exit(1);
        }
    }

    private void ConnectToServer()
    {
        try
        {
            clientSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
            clientSocket.Connect(ServerAddress, ServerPort);
            Console.WriteLine("Connected to server");
        }
        catch (Exception ex)
        {
            Console.WriteLine("Error connecting to server: " + ex.Message);
            kinectSensor.Stop();
            clientSocket.Close();
            Environment.Exit(1);
        }
    }

    private void ColorFrameReady(object sender, AllFramesReadyEventArgs e)
    {
        using (ColorImageFrame colorFrame = e.OpenColorImageFrame())
        {
            if (colorFrame != null)
            {
                // Process RGB frame
                byte[] rgbData = new byte[colorFrame.PixelDataLength];
                colorFrame.CopyPixelDataTo(rgbData);
                rgbData = CompressFrame(rgbData);
                SendFrame(1, rgbData);

            }
        }
    }

    private byte[] CompressFrame(byte[] frame)
    {
        // Compress the byte array before sending to the Python server
        using (MemoryStream ms = new MemoryStream())
        {
            BitmapSource bitmapSource = BitmapSource.Create(
                640, 480, 96, 96, PixelFormats.Bgr32, null, frame, 640 * 4);

            BitmapEncoder encoder = new JpegBitmapEncoder();
            encoder.Frames.Add(BitmapFrame.Create(bitmapSource));
            encoder.Save(ms);

            return ms.ToArray();
        }
    }

    private void DepthFrameReady(object sender, AllFramesReadyEventArgs e)
    {
        using (DepthImageFrame depthFrame = e.OpenDepthImageFrame())
        {
            if (depthFrame != null)
            {
                // Process Depth frame
                short[] depthData = new short[depthFrame.PixelDataLength];
                depthFrame.CopyPixelDataTo(depthData);
                byte[] depthBytes = new byte[depthData.Length * sizeof(short)];
                Buffer.BlockCopy(depthData, 0, depthBytes, 0, depthBytes.Length);
                SendFrame(2, depthBytes);
            }
        }
    }

    private void SendFrame(byte dataType, byte[] frameData)
    {
        try
        {
            // Create header
            byte[] header = new byte[5];
            header[0] = dataType;
            byte[] payloadSize = BitConverter.GetBytes(frameData.Length);
            if (BitConverter.IsLittleEndian)
            {
                Array.Reverse(payloadSize);
            }
            Array.Copy(payloadSize, 0, header, 1, 4);

            // Send header
            clientSocket.Send(header);

            // Send payload
            clientSocket.Send(frameData);
        }
        catch (Exception ex)
        {
            Console.WriteLine("Error sending frame: " + ex.Message);
            kinectSensor.Stop();
            clientSocket.Close();
            Environment.Exit(1);
        }
    }
}