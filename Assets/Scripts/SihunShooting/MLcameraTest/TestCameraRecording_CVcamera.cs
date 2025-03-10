using System;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.XR.MagicLeap;
using System.Collections;
using System.Collections.Generic;
using Newtonsoft.Json;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;
using UnityEngine.Android;
using System;
using System.IO;  // For file handling

/// <summary>
/// A script that enables and disables the RGB camera using the async methods.
/// </summary>
public class TestCameraRecording_CVcamera : MonoBehaviour
{
    /// <summary>
    /// Can be used by external scripts to query the status of the camera and see if the camera capture has been started.
    /// </summary>
    public bool IsCameraConnected => _captureCamera != null && _captureCamera.ConnectionEstablished;
    
    // [SerializeField][Tooltip("If true, the camera capture will start immediately.")]
    // public bool _startCameraCaptureOnStart = false;

    #region Capture Config

    private int _targetImageWidth = 1440;
    private int _targetImageHeight = 1080;

    [SerializeField]
    private int _videoImageWidth = 648;
    [SerializeField]
    private int _videoImageHeight = 720;
    private long _latestCaptureTime = 0;

    private MLCameraBase.Identifier _cameraIdentifier = MLCameraBase.Identifier.CV;
    private MLCameraBase.CaptureFrameRate _targetFrameRate = MLCameraBase.CaptureFrameRate._60FPS;
    private MLCameraBase.OutputFormat _outputFormat = MLCameraBase.OutputFormat.RGBA_8888;

    #endregion

    #region Magic Leap Camera Info
    //The connected Camera
    private MLCamera _captureCamera;
    // True if CaptureVideoStartAsync was called successfully
    private bool _isCapturingVideo = false;
    #endregion

    private bool? _cameraPermissionGranted;
    private bool _isCameraInitializationInProgress;
    private ConcurrentQueue<string> dataQueue;
    private string jsonFilePath;
    private const string baseFileName = "framedata_task";
    private const int WRITE_THRESHOLD = 50; // Number of entries before writing to disk
    private bool isRecording = false;
    private Task writeTask;
    private CancellationTokenSource cts;
    private float timetime;
    public class VideoFrameData
    {
        public long FrameNumber { get; set; }
        public double Timetime { get; set; }
        public double Timestamp { get; set; }
        public int image_width { get; set; }
        public int image_height { get; set; }
        public SerializableIntrinsicCalibrationParameters intrinsicParameters { get; set; }
        public SerializableMatrix4x4 cameraTransformMatrix { get; set; }

        public SerializableVector3 TopLeftPosition_3D { get; set; }
        public SerializableVector3 TopRightPosition_3D { get; set; }
        public SerializableVector3 BottomLeftPostion_3D { get; set; }
        public SerializableVector3 BottomRightPositon_3D { get; set; }
        public SerializableVector3 CenterPosition_3D { get; set; }

    }

   
    [Serializable]
    public class SerializableMatrix4x4
    {
        public float m00, m01, m02, m03;
        public float m10, m11, m12, m13;
        public float m20, m21, m22, m23;
        public float m30, m31, m32, m33;

        public SerializableMatrix4x4(Matrix4x4 matrix)
        {
            m00 = matrix.m00; m01 = matrix.m01; m02 = matrix.m02; m03 = matrix.m03;
            m10 = matrix.m10; m11 = matrix.m11; m12 = matrix.m12; m13 = matrix.m13;
            m20 = matrix.m20; m21 = matrix.m21; m22 = matrix.m22; m23 = matrix.m23;
            m30 = matrix.m30; m31 = matrix.m31; m32 = matrix.m32; m33 = matrix.m33;
        }

        public Matrix4x4 ToMatrix4x4()
        {
            return new Matrix4x4
            {
                m00 = m00, m01 = m01, m02 = m02, m03 = m03,
                m10 = m10, m11 = m11, m12 = m12, m13 = m13,
                m20 = m20, m21 = m21, m22 = m22, m23 = m23,
                m30 = m30, m31 = m31, m32 = m32, m33 = m33
            };
        }
    }

    [Serializable]
    public struct SerializableVector2
    {
        public float x { get; set; }
        public float y { get; set; }

        public SerializableVector2(Vector2 vector)
        {
            x = vector.x;
            y = vector.y;
        }

        public Vector2 ToVector2()
        {
            return new Vector2(x, y);
        }
    }
    [Serializable]
    public struct SerializableVector3
    {
        public float x { get; set; }
        public float y { get; set; }
        public float z { get; set; }

        public SerializableVector3(Vector3 vector)
        {
            x = vector.x;
            y = vector.y;
            z = vector.z;

        }

        public Vector3 ToVector3()
        {
            return new Vector3(x, y,z);
        }
    }

    [Serializable]
    public class SerializableIntrinsicCalibrationParameters
    {
        public uint Width { get; set; }
        public uint Height { get; set; }
        public SerializableVector2 FocalLength { get; set; }
        public SerializableVector2 PrincipalPoint { get; set; }
        public float FOV { get; set; }
        public double[] Distortion { get; set; }

        public SerializableIntrinsicCalibrationParameters(UnityEngine.XR.MagicLeap.MLCameraBase.IntrinsicCalibrationParameters source)
        {
            Width = source.Width;
            Height = source.Height;
            FocalLength = new SerializableVector2(source.FocalLength);
            PrincipalPoint = new SerializableVector2(source.PrincipalPoint);
            FOV = source.FOV;
            Distortion = source.Distortion;
        }
    }
    public bool visualize_camera;

    // We'll create one marker per corner/center.
    private GameObject markerTopLeft;
    private GameObject markerTopRight;
    private GameObject markerBottomLeft;
    private GameObject markerBottomRight;
    private GameObject markerCenter;
    private readonly MLPermissions.Callbacks _permissionCallbacks = new MLPermissions.Callbacks();
    // Add this public method to start the camera dynamically
    public void StartCameraFromExternalFlag()
    {
        if (!_isCapturingVideo && !_isCameraInitializationInProgress)
        {
            Debug.Log("External flag triggered camera start. Starting after a 2-second delay...");
            StartCoroutine(StartCameraWithDelay());
        }
        else
        {
            Debug.LogWarning("Camera is already capturing or initialization is in progress.");
        }
    }

    private IEnumerator StartCameraWithDelay()
    {
        yield return new WaitForSeconds(2f); // Wait for 2 seconds
        StartCameraCapture(_cameraIdentifier, _targetImageWidth, _targetImageHeight);
        Debug.Log("Camera capture started after 2-second delay.");
    }
    private void Awake()
    {
        _permissionCallbacks.OnPermissionGranted += OnPermissionGranted;
        _permissionCallbacks.OnPermissionDenied += OnPermissionDenied;
        _permissionCallbacks.OnPermissionDeniedAndDontAskAgain += OnPermissionDenied;
        _isCapturingVideo = false;
    }

    void Start()
    {
        if(visualize_camera)
        {

            // Create small red cubes (or random colors).
            markerTopLeft = CreateCubeMarker("Marker_TopLeft", Color.red);
            markerTopRight = CreateCubeMarker("Marker_TopRight", Color.blue);
            markerBottomLeft = CreateCubeMarker("Marker_BottomLeft", Color.green);
            markerBottomRight = CreateCubeMarker("Marker_BottomRight", Color.yellow);
            markerCenter = CreateCubeMarker("Marker_Center", Color.magenta);

            // Optionally rename for clarity in the Hierarchy
            markerTopLeft.name = "Marker_TopLeft";
            markerTopRight.name = "Marker_TopRight";
            markerBottomLeft.name = "Marker_BottomLeft";
            markerBottomRight.name = "Marker_BottomRight";
            markerCenter.name = "Marker_Center";


        }
    }
    private GameObject CreateCubeMarker(string name, Color color)
    {
        // Create a primitive cube object in the scene
        GameObject cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
        var renderer = cube.GetComponent<Renderer>();
        renderer.material = new Material(Shader.Find("Unlit/Color"));
        cube.name = name;

        // Make it small so it looks more like a point marker
        cube.transform.localScale = Vector3.one * 0.02f;

        // Optionally set its color
        if (renderer != null)
        {
            renderer.material.color = color;
        }

        return cube;
    }
    /// <summary>
    /// Starts the Camera capture with the target settings.
    /// </summary>
    /// <param name="cameraIdentifier">Which camera to use. (Main or CV)</param>
    /// <param name="width">The width of the video stream.</param>
    /// <param name="height">The height of the video stream.</param>
    /// <param name="onCameraCaptureStarted">An action callback that returns true if the video capture started successfully.</param>
    public void StartCameraCapture(MLCameraBase.Identifier cameraIdentifier = MLCameraBase.Identifier.CV, int width = 640, int height = 480, Action<bool> onCameraCaptureStarted = null)
    {
        if (_isCameraInitializationInProgress)
        {
            Debug.LogError("Camera Initialization is already in progress.");
            onCameraCaptureStarted?.Invoke(false);
            return;
        }

        this._cameraIdentifier = cameraIdentifier;
        _targetImageWidth = width;
        _targetImageHeight = height;
        TryEnableMLCamera(onCameraCaptureStarted);
    }
    private void OnEnable()
    {
        MarkerDetection.OnMarkerDetectionDestroyed += StartCameraFromExternalFlag;
        dataQueue = new ConcurrentQueue<string>();

    }    
    private void OnDisable()
    {
        Debug.Log("[Camera] Disabling camera system...");
        
        // Stop camera capture if active
        if (_isCapturingVideo)
        {
            _ = DisconnectCameraAsync();
        }

        // Unsubscribe from MarkerDetection event
        MarkerDetection.OnMarkerDetectionDestroyed -= StartCameraFromExternalFlag;

        // Stop recording if active
        if (isRecording)
        {
            StopRecording();
        }
    }


    private void OnPermissionGranted(string permission)
    {
        if (permission == MLPermission.Camera)
        {
            _cameraPermissionGranted = true;
            Debug.Log($"Granted {permission}.");
        }
    }

    private void OnPermissionDenied(string permission)
    {
        if (permission == MLPermission.Camera)
        {
            _cameraPermissionGranted = false;
            Debug.LogError($"{permission} denied, camera capture won't function.");
        }
    }

    private async void TryEnableMLCamera(Action<bool> onCameraCaptureStarted = null)
    {
        // If the camera initialization is already in progress, return immediately
        if (_isCameraInitializationInProgress)
        {
            onCameraCaptureStarted?.Invoke(false);
            return;
        }

        _isCameraInitializationInProgress = true;

        _cameraPermissionGranted = null;
        Debug.Log("Requesting Camera permission.");
        MLPermissions.RequestPermission(MLPermission.Camera, _permissionCallbacks);

        while (!_cameraPermissionGranted.HasValue)
        {
            // Wait until we have permission to use the camera
            await Task.Delay(TimeSpan.FromSeconds(1.0f));
        }

        if (MLPermissions.CheckPermission(MLPermission.Camera).IsOk || _cameraPermissionGranted.GetValueOrDefault(false))
        {
            Debug.Log("Initializing camera.");
            bool isCameraAvailable = await WaitForCameraAvailabilityAsync();

            if (isCameraAvailable)
            {
                await ConnectAndConfigureCameraAsync();
            }
        }

        _isCameraInitializationInProgress = false;
        onCameraCaptureStarted?.Invoke(_isCapturingVideo);
    }

    /// <summary>
    /// Connects the MLCamera component and instantiates a new instance
    /// if it was never created.
    /// </summary>
    private async Task<bool> WaitForCameraAvailabilityAsync()
    {
        bool cameraDeviceAvailable = false;
        int maxAttempts = 10;
        int attempts = 0;
   
        while (!cameraDeviceAvailable && attempts < maxAttempts)
        {
            MLResult result =
                MLCameraBase.GetDeviceAvailabilityStatus(_cameraIdentifier, out cameraDeviceAvailable);

            if (result.IsOk == false && cameraDeviceAvailable == false)
            {
                // Wait until the camera device is available
                await Task.Delay(TimeSpan.FromSeconds(1.0f));
            }
            attempts++;
        }

        return cameraDeviceAvailable;
    }

    private async Task<bool> ConnectAndConfigureCameraAsync()
    {
        Debug.Log("Starting Camera Capture.");

        MLCameraBase.ConnectContext context = CreateCameraContext();

        _captureCamera = await MLCamera.CreateAndConnectAsync(context);
        if (_captureCamera == null)
        {
            Debug.LogError("Could not create or connect to a valid camera. Stopping Capture.");
            return false;
        }

        Debug.Log("Camera Connected.");

        bool hasImageStreamCapabilities = GetStreamCapabilityWBestFit(out MLCameraBase.StreamCapability streamCapability);
        if (!hasImageStreamCapabilities)
        {
            Debug.LogError("Could not start capture. No valid Image Streams available. Disconnecting Camera.");
            await DisconnectCameraAsync();
            return false;
        }

        Debug.Log("Preparing camera configuration.");

        // Try to configure the camera based on our target configuration values
        MLCameraBase.CaptureConfig captureConfig = CreateCaptureConfig(streamCapability);
        var prepareResult = _captureCamera.PrepareCapture(captureConfig, out MLCameraBase.Metadata _);
        if (!MLResult.DidNativeCallSucceed(prepareResult.Result, nameof(_captureCamera.PrepareCapture)))
        {
            Debug.LogError($"Could not prepare capture. Result: {prepareResult.Result} .  Disconnecting Camera.");
            await DisconnectCameraAsync();
            return false;
        }

        Debug.Log("Starting Video Capture.");

        bool captureStarted = await StartVideoCaptureAsync();
        if (!captureStarted)
        {
            Debug.LogError("Could not start capture. Disconnecting Camera.");
            await DisconnectCameraAsync();
            return false;
        }

        return _isCapturingVideo;
    }

    private MLCameraBase.ConnectContext CreateCameraContext()
    {
        var context = MLCameraBase.ConnectContext.Create();
        context.CamId = _cameraIdentifier;
        context.Flags = MLCameraBase.ConnectFlag.CamOnly;
        return context;
    }

    private MLCameraBase.CaptureConfig CreateCaptureConfig(MLCameraBase.StreamCapability streamCapability)
    {
        var captureConfig = new MLCameraBase.CaptureConfig();
        captureConfig.CaptureFrameRate = _targetFrameRate;
        captureConfig.StreamConfigs = new MLCameraBase.CaptureStreamConfig[1];
        captureConfig.StreamConfigs[0] = MLCameraBase.CaptureStreamConfig.Create(streamCapability, _outputFormat);
        return captureConfig;
    }

    private async Task<bool> StartVideoCaptureAsync()
    {
        // Trigger auto exposure and white balance
        await _captureCamera.PreCaptureAEAWBAsync();

        var startCapture = await _captureCamera.CaptureVideoStartAsync();
        _isCapturingVideo = MLResult.DidNativeCallSucceed(startCapture.Result, nameof(_captureCamera.CaptureVideoStart));

        if (!_isCapturingVideo)
        {
            Debug.LogError($"Could not start camera capture. Result : {startCapture.Result}");
            return false;
        }

        _captureCamera.OnRawVideoFrameAvailable += OnCaptureRawVideoFrameAvailable;
        return true;
    }

    public async Task DisconnectCameraAsync()
    {
        if (_captureCamera != null)
        {
            if (_isCapturingVideo)
            {
                await _captureCamera.CaptureVideoStopAsync();
                _captureCamera.OnRawVideoFrameAvailable -= OnCaptureRawVideoFrameAvailable;
            }

            await _captureCamera.DisconnectAsync();
            Debug.LogError($"Disconnect CVCamera");

            _captureCamera = null;
        }

        _isCapturingVideo = false;
    }

    /// <summary>
    /// Gets the Image stream capabilities.
    /// </summary>
    /// <returns>True if MLCamera returned at least one stream capability.</returns>
    private bool GetStreamCapabilityWBestFit(out MLCameraBase.StreamCapability streamCapability)
    {
        streamCapability = default;

        if (_captureCamera == null)
        {
            Debug.Log("Could not get Stream capabilities Info. No Camera Connected");
            return false;
        }

        MLCameraBase.StreamCapability[] streamCapabilities =
            MLCameraBase.GetImageStreamCapabilitiesForCamera(_captureCamera, MLCameraBase.CaptureType.Video);

        if (streamCapabilities.Length <= 0) 
            return false;


        if (MLCameraBase.TryGetBestFitStreamCapabilityFromCollection(streamCapabilities, _targetImageWidth,
                _targetImageHeight, MLCameraBase.CaptureType.Video,
                out streamCapability))
        {
            Debug.Log($"Stream: {streamCapability} selected with best fit.");
            return true;
        }

        Debug.Log($"No best fit found. Stream: {streamCapabilities[0]} selected by default.");
        streamCapability = streamCapabilities[0];
        return true;
    }



    public void StartRecording(int taskindex)
    {
        if (isRecording)
        {
            Debug.LogWarning("Recording is already active.");
            return;
        }

        jsonFilePath = SharedInfomanager.Instance.GenerateUniqueFilePath(baseFileName, taskindex, "json");
        Debug.Log($"Recording started. Saving to file: {jsonFilePath}");

        dataQueue.Clear();

        cts = new CancellationTokenSource();
        writeTask = Task.Run(async () =>
        {
            while (!cts.Token.IsCancellationRequested)
            {
                Debug.Log($"[WriterTask] dataQueue.Count = {dataQueue.Count}, threshold = {WRITE_THRESHOLD}");

                if (dataQueue.Count >= WRITE_THRESHOLD)
                {
                    await WriteToDiskAsync(cts.Token); // Pass the cancellation token here
                }
                await Task.Delay(200, cts.Token); // Pass the cancellation token here as well
            }

            await WriteToDiskAsync(cts.Token); // Final flush with cancellation token
        }, cts.Token);
        isRecording = true;

    }


    public void StopRecording()
    {
        if (!isRecording)
        {
            Debug.LogWarning("No active recording to stop.");
            return;
        }

        isRecording = false;

        // Signal the task to stop
        if (cts != null)
        {
            cts.Cancel();
            try
            {
                writeTask?.Wait();
            }
            catch (AggregateException ae)
            {
                foreach (var ex in ae.InnerExceptions)
                {
                    if (ex is TaskCanceledException)
                    {
                        Debug.Log("Write task was canceled.");
                    }
                    else
                    {
                        Debug.LogError($"Unexpected error: {ex.Message}");
                    }
                }
            }
            finally
            {
                cts.Dispose();
                cts = null;
            }
        }

        // Flush any remaining data
        if (!dataQueue.IsEmpty)
        {
            Debug.Log("Flushing remaining data in the queue to disk...");
            try
            {
                WriteRemainingDataToDisk();
            }
            catch (Exception ex)
            {
                Debug.LogError($"Error flushing remaining data: {ex.Message}");
            }
        }

        Debug.Log($"Recording stopped. Data saved to file: {jsonFilePath}");
    }
    private void OnCaptureRawVideoFrameAvailable(MLCameraBase.CameraOutput cameraOutput,
        MLCameraBase.ResultExtras resultExtras,
        MLCameraBase.Metadata metadata)
    {
        if (!isRecording)
            return;
        timetime= Time.time;
        Debug.Log("Frame callback triggered.");
        var result = MLTime.ConvertMLTimeToSystemTime(resultExtras.VCamTimestamp, out long time);
        _latestCaptureTime = time / 1000000;

        if (MLCVCamera.GetFramePose(resultExtras.VCamTimestamp, out Matrix4x4 cameraTransform).IsOk && result.IsOk)
        {
            uint width = cameraOutput.Planes[0].Width;
            uint height = cameraOutput.Planes[0].Height;

            Vector2 topLeftPixel = new Vector2(0, 0);
            Vector2 topRightPixel = new Vector2(width, 0);
            Vector2 bottomLeftPixel = new Vector2(0, height);
            Vector2 bottomRightPixel = new Vector2(width, height);
            Vector2 centerPixel = new Vector2(width / 2f, height / 2f);

            Vector3 TopLeftposition_3D = CameraUtilities.CastRayFromScreenToWorldPoint(resultExtras.Intrinsics.Value, cameraTransform,topLeftPixel);
            Vector3 TopRightposition_3D = CameraUtilities.CastRayFromScreenToWorldPoint(resultExtras.Intrinsics.Value, cameraTransform, topRightPixel);
            Vector3 BottomLeftposition_3D = CameraUtilities.CastRayFromScreenToWorldPoint(resultExtras.Intrinsics.Value, cameraTransform, bottomLeftPixel);
            Vector3 BottomRightposition_3D = CameraUtilities.CastRayFromScreenToWorldPoint(resultExtras.Intrinsics.Value, cameraTransform, bottomRightPixel);
            Vector3 Centerposition_3D = CameraUtilities.CastRayFromScreenToWorldPoint(resultExtras.Intrinsics.Value, cameraTransform, centerPixel);

            // Update your five marker positions:
            // markerTopLeft.transform.position = TopLeftposition_3D;
            // markerTopRight.transform.position = TopRightposition_3D;
            // markerBottomLeft.transform.position = BottomLeftposition_3D;
            // markerBottomRight.transform.position = BottomRightposition_3D;
            // markerCenter.transform.position = Centerposition_3D;
            // Debug.Log($"TopLeftposition_3D: {TopLeftposition_3D}");
            // Debug.Log($"TopRightposition_3D: {TopRightposition_3D}");
            // Debug.Log($"BottomLeftposition_3D: {BottomLeftposition_3D}");
            // Debug.Log($"BottomRightposition_3D: {BottomRightposition_3D}");
            // Debug.Log($"Centerposition_3D: {Centerposition_3D}");
            // Debug.Log($"CameraPosition: {Camera.main.transform.position}");
            // save log
            var videoFrameData = new VideoFrameData
            {
                FrameNumber = resultExtras.FrameNumber,
                Timetime = timetime,
                Timestamp = _latestCaptureTime,
                image_width = _videoImageWidth,
                image_height = _videoImageHeight,
                intrinsicParameters = new SerializableIntrinsicCalibrationParameters(resultExtras.Intrinsics.Value),
                cameraTransformMatrix = new SerializableMatrix4x4(cameraTransform),
                TopLeftPosition_3D = new SerializableVector3(TopLeftposition_3D),
                TopRightPosition_3D = new SerializableVector3(TopRightposition_3D),
                BottomLeftPostion_3D = new SerializableVector3(BottomLeftposition_3D),
                BottomRightPositon_3D = new SerializableVector3(BottomRightposition_3D),
                CenterPosition_3D = new SerializableVector3(Centerposition_3D)
            };
            // transmit to the eye tracker
            SharedInfomanager.Instance.CameraDataQueue.Enqueue(new SharedInfomanager.CameraData
            {
                Timestamp = _latestCaptureTime,
                image_width = _videoImageWidth,
                image_height = _videoImageHeight,
                intrinsicParameters = resultExtras.Intrinsics.Value,
                cameraTransformMatrix = cameraTransform,
                TopLeftPosition_3D=TopLeftposition_3D,
                TopRightPosition_3D=TopRightposition_3D,
                BottomLeftPostion_3D=BottomLeftposition_3D,
                BottomRightPositon_3D=BottomRightposition_3D,
                CenterPosition_3D=Centerposition_3D
            });

            Debug.Log($"FrameNumber: {videoFrameData.FrameNumber}");

            // Serialize to JSON and enqueue
            string jsonData = JsonConvert.SerializeObject(videoFrameData, Formatting.Indented);
            dataQueue.Enqueue(jsonData);

            // Write to disk if the queue reaches the threshold
            if (dataQueue.Count >= WRITE_THRESHOLD)
            {
                _ = WriteToDiskAsync(cts.Token); // Pass the cancellation token
            }
        }
    }
    private async Task WriteToDiskAsync(CancellationToken token)
        {
            if (dataQueue.IsEmpty)
                return;

            List<string> batch = new List<string>();
            while (dataQueue.TryDequeue(out string entry))
            {
                token.ThrowIfCancellationRequested(); // Explicitly observe cancellation
                batch.Add(entry);
            }

            try
            {
                // Lock to prevent simultaneous access to the file
                lock (jsonFilePath)
                {
                    using (StreamWriter writer = new StreamWriter(jsonFilePath, append: true))
                    {
                        foreach (var entry in batch)
                        {
                            token.ThrowIfCancellationRequested(); // Check for cancellation during write
                            writer.WriteLine(entry);
                        }
                    }
                }

                Debug.Log($"Wrote {batch.Count} entries to {jsonFilePath}.");
            }
            catch (OperationCanceledException)
            {
                Debug.LogWarning("Write operation was canceled.");
            }
            catch (IOException e)
            {
                Debug.LogError($"Error writing to file: {e.Message}");
            }
        }
        
    private void WriteRemainingDataToDisk()
    {
        if (dataQueue.IsEmpty)
            return;

        List<string> batch = new List<string>();
        while (dataQueue.TryDequeue(out string entry))
        {
            batch.Add(entry);
        }

        try
        {
            // Lock to prevent simultaneous file access
            lock (jsonFilePath)
            {
                using (StreamWriter writer = new StreamWriter(jsonFilePath, append: true))
                {
                    foreach (var entry in batch)
                    {
                        writer.WriteLine(entry);
                    }
                }
            }

            Debug.Log($"Flushed {batch.Count} remaining entries to {jsonFilePath}.");
        }
        catch (IOException e)
        {
            Debug.LogError($"Error writing remaining data to file: {e.Message}");
        }
    }
}

