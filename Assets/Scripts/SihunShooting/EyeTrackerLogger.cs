using UnityEngine;
using UnityEngine.XR.OpenXR;
using MagicLeap.OpenXR.Features.EyeTracker;
using MagicLeap.Android;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.IO;  // For file handling
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;
using UnityEngine.Android;
using System;
using UnityEngine.InputSystem;
using System.Collections;
using UnityEngine.XR.MagicLeap;

using UnityEngine.XR.OpenXR.NativeTypes;


public class EyeTrackerLogger : MonoBehaviour
{
    // Eye gaze 
    [SerializeField] private InputActionReference positionActionReference;
    [SerializeField] private InputActionReference rotationActionReference;
    [SerializeField] private InputActionReference headrotationActionReference;
    [SerializeField]
    private InputAction positionInputAction =
        new InputAction(binding: "<MagicLeapController>/pointerPosition", expectedControlType: "Vector3");

    [SerializeField]
    private InputAction rotationInputAction =
        new InputAction(binding: "<MagicLeapController>/pointerRotation", expectedControlType: "Quaternion");
   
    [SerializeField] private float offsetDistance = 1.0f;
    [SerializeField] private bool isVisualizing = false;


    public GameObject objectToMove;

    
    
    
    
    
    //Eye tracking 
    private MagicLeapEyeTrackerFeature eyeTrackerFeature;
    private bool eyeTrackPermissionGranted = false;
    private bool pupilSizePermissionGranted = false;
    private bool writePermissionGranted = false;
    private string jsonFilePath;
    private string baseFileName = "eyetracking_task";
    
    


    private long prevGazeTime = 0;
    private long prevCameraTime = 0;
    private Vector3 fixationPoint;
   private long timeDifferenceThreshold = 18; // 18ms

    private SharedInfomanager.CameraData cameraData;

    
    private ConcurrentQueue<string> dataQueue;
    private Task writeTask;
    private CancellationTokenSource cts;

    private const int WRITE_THRESHOLD = 50; // Number of entries before writing to disk
    private bool isRecording = false;

    private void OnEnable()
    {
        positionActionReference?.action.Enable();
        rotationActionReference?.action.Enable();
        headrotationActionReference?.action.Enable();

    if (isVisualizing && objectToMove)
    {
        objectToMove.SetActive(false);

        Material newMaterial = new Material(Shader.Find("Universal Render Pipeline/Lit"));
        newMaterial.SetColor("_BaseColor", Color.red);
        objectToMove.GetComponent<Renderer>().material = newMaterial;
        objectToMove.transform.localScale = new Vector3(0.025f, 0.025f, 0.025f);

        Debug.Log("Visualization cube created and tinted red.");
        }
        dataQueue = new ConcurrentQueue<string>();
    }

    private void OnDisable()
    {
        positionActionReference?.action.Disable();
        rotationActionReference?.action.Disable();
        headrotationActionReference?.action.Disable();

        if (objectToMove != null)
        {
            Destroy(objectToMove);
        }

        StopRecording();
    }

    void Start()
    {

        dataQueue = new ConcurrentQueue<string>();

        Permissions.RequestPermissions(new string[]
        {
            Permissions.EyeTracking,
            Permissions.PupilSize
        }, OnPermissionGranted, OnPermissionDenied);

        if (!Permission.HasUserAuthorizedPermission(Permission.ExternalStorageWrite))
        {
            Debug.Log("Requesting write permission...");
            Permission.RequestUserPermission(Permission.ExternalStorageWrite);
        }
        else
        {
            writePermissionGranted = true;
            Debug.Log("Write permission granted.");
        }


    }


    private void OnPermissionGranted(string permission)
    {
        if (permission == Permissions.EyeTracking)
        {
            eyeTrackPermissionGranted = true;
            Debug.Log("Eye Tracking permission granted.");
        }

        if (permission == Permissions.PupilSize)
        {
            pupilSizePermissionGranted = true;
            Debug.Log("Pupil Size permission granted.");
        }

        // Check if all required permissions are granted
        if (eyeTrackPermissionGranted && pupilSizePermissionGranted)
        {
            InitializeEyeTracker();
        }
    }

    private void OnPermissionDenied(string permission)
    {
        Debug.LogError($"{permission} denied. Eye tracking data will not be available.");
    }

    private void InitializeEyeTracker()
    {
        // Initialize the Eye Tracker feature
        eyeTrackerFeature = OpenXRSettings.Instance.GetFeature<MagicLeapEyeTrackerFeature>();

        if (eyeTrackerFeature != null && eyeTrackerFeature.enabled)
        {
            eyeTrackerFeature.CreateEyeTracker();
            Debug.Log("Eye Tracker initialized.");
        }
        else
        {
            Debug.LogError("Failed to initialize the Eye Tracker. Ensure the feature is enabled in OpenXR settings.");
        }
    }

    void FixedUpdate()
    {

        if (!isRecording || !eyeTrackPermissionGranted || !pupilSizePermissionGranted || !writePermissionGranted)
            return;


        // Check if the eye tracker feature is available and enabled
        if (eyeTrackerFeature == null || !eyeTrackerFeature.enabled)
        {
            Debug.LogError("Update skipped: Eye tracker feature is null or not enabled.");
            return;
        }

        // If all conditions are met, log that data will be recorded
        Debug.Log("All conditions met. Logging eye-tracking data.");

        // Log data
        LogEyeTrackingData();


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

    public void StopRecording()
    {
        if (!isRecording)
        {
            Debug.LogWarning("No active recording to stop.");
            return;
        }
        // Enable input actions
        positionInputAction.Dispose();
        rotationInputAction.Dispose();
        
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
    
    public void StartRecording(int taskindex)
    {
        if (isRecording)
        {
            Debug.LogWarning("Recording is already active.");
            return;
        }

        // Enable input actions
        positionInputAction.Enable();
        rotationInputAction.Enable();
        
        
        jsonFilePath = SharedInfomanager.Instance.GenerateUniqueFilePath(baseFileName, taskindex, "json");
        Debug.Log($"Recording started. Saving to file: {jsonFilePath}");

        dataQueue.Clear();

        cts = new CancellationTokenSource();
        writeTask = Task.Run(async () =>
        {
            while (!cts.Token.IsCancellationRequested)
            {
                if (dataQueue.Count >= WRITE_THRESHOLD)
                {
                    await WriteToDiskAsync(cts.Token); // Pass the cancellation token here
                }
                await Task.Delay(200, cts.Token); // Pass the cancellation token here as well
            }

            await WriteToDiskAsync(cts.Token); // Final flush with cancellation token
        }, cts.Token);
        isRecording = true;

        if (isVisualizing && objectToMove )
        {
        objectToMove.SetActive(true);
        StartCoroutine(DisableFixationObjectAfterSeconds(5f)); // <-- Start the 5s timer here

        }
    }
    private IEnumerator DisableFixationObjectAfterSeconds(float seconds)
    {
        // Wait for the specified time
        yield return new WaitForSeconds(seconds);

        // Disable the object, if it still exists
        if (objectToMove != null)
        {
            objectToMove.SetActive(false);
            Debug.Log("Fixation object disabled after " + seconds + " seconds.");
        }
    }
        
    
    private void LogEyeTrackingData()
    {


        // Eye  Tracking

        EyeTrackerData eyeTrackerData = eyeTrackerFeature.GetEyeTrackerData();
        float currentinTimestamp=Time.time;

        // Retrieve all eye-tracking data
        long currentTime = eyeTrackerData.GazeBehaviorData.Time / 1000000;

        if (prevGazeTime == currentTime)
        {
            Debug.Log("Duplicate gaze time. Skipping frame: " + currentTime);
            return;
        }
        prevGazeTime = currentTime;

        var eyeTrackingDataDict = new Dictionary<string, object>();

        var geometricDataList = new List<Dictionary<string, object>>();
        foreach (var geometricData in eyeTrackerData.GeometricData)
        {
            var geometricDict = new Dictionary<string, object>
            {
                { "Eye", geometricData.Eye.ToString() },
                { "Time", geometricData.Time / 1000000},
                { "Valid", geometricData.Valid },
                { "EyeOpenness", geometricData.EyeOpenness },
                { "EyeInSkullPosition", new Dictionary<string, float>
                    {
                        { "x", geometricData.EyeInSkullPosition.x },
                        { "y", geometricData.EyeInSkullPosition.y }
                    }
                }
            };
            geometricDataList.Add(geometricDict);
            // Log each geometric data entry
            
        }
        eyeTrackingDataDict["GeometricData"] = geometricDataList;

        var pupilDataList = new List<Dictionary<string, object>>();
        foreach (var pupilData in eyeTrackerData.PupilData)
        {
            var pupilDict = new Dictionary<string, object>
            {
                { "Eye", pupilData.Eye.ToString() },
                { "Time", pupilData.Time/ 1000000 },
                { "Valid", pupilData.Valid },
                { "PupilDiameter", pupilData.PupilDiameter }
            };
            pupilDataList.Add(pupilDict);
        }
        eyeTrackingDataDict["PupilData"] = pupilDataList;

        GazeBehavior gazeBehavior = eyeTrackerData.GazeBehaviorData;
        var gazeBehaviorDict = new Dictionary<string, object>
        {
            { "GazeBehaviorType", gazeBehavior.GazeBehaviorType.ToString() },
            { "Time", gazeBehavior.Time/ 1000000 },
            { "Valid", gazeBehavior.Valid },
            { "OnsetTime", gazeBehavior.OnsetTime/ 1000000 },
            { "Duration", gazeBehavior.Duration },
            { "MetaData", new Dictionary<string, object>
                {
                    { "Valid", gazeBehavior.MetaData.Valid },
                    { "Amplitude", gazeBehavior.MetaData.Amplitude },
                    { "Direction", gazeBehavior.MetaData.Direction },
                    { "Velocity", gazeBehavior.MetaData.Velocity }
                }
            }
        };
        eyeTrackingDataDict["GazeBehaviorData"] = gazeBehaviorDict;

        StaticData staticData = eyeTrackerData.StaticData;
        var staticDataDict = new Dictionary<string, object>
        {
            { "EyeWidthMax", staticData.EyeWidthMax },
            { "EyeHeightMax", staticData.EyeHeightMax }
        };
        eyeTrackingDataDict["StaticData"] = staticDataDict;

        // 

        // 
        // eye gaze from eyetrackerdata 
        bool hasData = eyeTrackerData.PosesData.Result == XrResult.Success;

        Vector3 centerEyePosition = eyeTrackerData.PosesData.GazePose.Pose.position;
        Quaternion centerEyeRotation = eyeTrackerData.PosesData.GazePose.Pose.rotation;
        Quaternion headRotation = headrotationActionReference?.action.ReadValue<Quaternion>()?? Quaternion.identity;
        if (hasData)
        {
            fixationPoint = eyeTrackerData.PosesData.FixationPose.Pose.position;
        }
        else
        {
            fixationPoint = Vector3.zero;
        }
      
        eyeTrackingDataDict["Startvideo"] = SharedInfomanager.Instance.startVideo;

        var eyeGazeData_from = new Dictionary<string, object>
        {
            { "Timestamp", currentinTimestamp},

            {
                "EyeGazePosition", new Dictionary<string, float>
                {
                    { "x", centerEyePosition[0] },
                    { "y", centerEyePosition[1] },
                    { "z", centerEyePosition[2] }
                }
            },
            {
                "HeadRotation", new Dictionary<string, float>
                {
                    { "x", headRotation[0] },
                    { "y", headRotation[1] },
                    { "z", headRotation[2] },
                    { "w", headRotation[3] }
                }
            },
            {
                "EyeGazeRotation", new Dictionary<string, float>
                {
                    { "x", centerEyeRotation[0] },
                    { "y", centerEyeRotation[1] },
                    { "z", centerEyeRotation[2] },
                    { "w", centerEyeRotation[3] }
                }
            },
            {
                "fixationPoint", new Dictionary<string, float>
                {
                    { "x", fixationPoint[0] },
                    { "y", fixationPoint[1] },
                    { "z", fixationPoint[2] }
                }
            }
        };

        if (fixationPoint==Vector3.zero){
        fixationPoint = centerEyePosition + centerEyeRotation * Vector3.forward * offsetDistance;
        }
        
        (long cameraTime, Vector2 projected2DGazePoint, Vector3 topleft3D,Vector3 topright3D,Vector3 bottomleft3D,Vector3 bottomright3D,Vector3 center3D) = ComputeProjected2DGazePoint(fixationPoint, currentTime);


        // 2. Later, when you have the values for cameraTime and projected2DGazePoint:
        eyeGazeData_from["cameraTime"] = cameraTime;

        eyeGazeData_from["projected2DGazePoint"] = new Dictionary<string, float>
        {
            { "x", projected2DGazePoint[0] },
            { "y", projected2DGazePoint[1] }
        };
       eyeGazeData_from["topleft3D"] = new Dictionary<string, float>
        {
            { "x", topleft3D[0] },
            { "y", topleft3D[1] },
            { "z", topleft3D[2] },
        };
       eyeGazeData_from["topright3D"] = new Dictionary<string, float>
        {
            { "x", topright3D[0] },
            { "y", topright3D[1] },
            { "z", topright3D[2] },
        };
       eyeGazeData_from["bottomleft3D"] = new Dictionary<string, float>
        {
            { "x", bottomleft3D[0] },
            { "y", bottomleft3D[1] },
            { "z", bottomleft3D[2] },
        };
       eyeGazeData_from["bottomright3D"] = new Dictionary<string, float>
        {
            { "x", bottomright3D[0] },
            { "y", bottomright3D[1] },
            { "z", bottomright3D[2] },
        };
       eyeGazeData_from["center3D"] = new Dictionary<string, float>
        {
            { "x", center3D[0] },
            { "y", center3D[1] },
            { "z", center3D[2] },
        };
        

        eyeTrackingDataDict["GazeData_from"] = eyeGazeData_from;


        // Eye Gaze data from read
        Vector3 eyeGazePosition = positionActionReference?.action.ReadValue<Vector3>() ?? Vector3.zero;
        Quaternion eyeGazeRotation = rotationActionReference?.action.ReadValue<Quaternion>() ?? Quaternion.identity;
        Vector3 gazeDirection = eyeGazeRotation * Vector3.forward;

        var eyeGazeData_read = new Dictionary<string, object>
        {
            { "Timestamp", Time.time },
            { "EyeGazePosition", new Dictionary<string, float>
                {
                    { "x", eyeGazePosition.x },
                    { "y", eyeGazePosition.y },
                    { "z", eyeGazePosition.z }
                }
            },
            { "EyeGazeRotation", new Dictionary<string, float>
                {
                    { "x", eyeGazeRotation.x },
                    { "y", eyeGazeRotation.y },
                    { "z", eyeGazeRotation.z },
                    { "w", eyeGazeRotation.w }
                }
            },
            { "GazeDirection", new Dictionary<string, float>
                {
                    { "x", gazeDirection.x },
                    { "y", gazeDirection.y },
                    { "z", gazeDirection.z }
                }
            }
        };

        eyeTrackingDataDict["GazeData_read"] = eyeGazeData_read;



        // controller data
        Vector3 controllerPosition = positionInputAction.ReadValue<Vector3>();
        Quaternion controllerrotation = rotationInputAction.ReadValue<Quaternion>();
        Vector3 controllerdirection = controllerrotation * Vector3.forward;
        var controllerdata = new Dictionary<string, object>
        {
            { "Timestamp", Time.time },
            { "controllerPosition", new Dictionary<string, float>
                {
                    { "x", controllerPosition.x },
                    { "y", controllerPosition.y },
                    { "z", controllerPosition.z }
                }
            },
            { "controllerrotation", new Dictionary<string, float>
                {
                    { "x", controllerrotation.x },
                    { "y", controllerrotation.y },
                    { "z", controllerrotation.z },
                    { "w", controllerrotation.w }
                }
            },
            { "controllerdirection", new Dictionary<string, float>
                {
                    { "x", controllerdirection.x },
                    { "y", controllerdirection.y },
                    { "z", controllerdirection.z }
                }
            }
        };
        eyeTrackingDataDict["controller_data"] = controllerdata;


        Debug.Log($"eye tracking: {eyeTrackingDataDict}");

        string jsonData = JsonConvert.SerializeObject(eyeTrackingDataDict, Formatting.None);
        dataQueue.Enqueue(jsonData);
        if (dataQueue.Count >= WRITE_THRESHOLD)
        {
            _ = WriteToDiskAsync(cts.Token); // Pass the cancellation token
        }

        if (isVisualizing && objectToMove != null)
        {
            objectToMove.transform.position = fixationPoint;
        }
    }



    private void OnDestroy()
    {
        if (isRecording)
        {
            StopRecording();
        }

        if (eyeTrackerFeature != null && eyeTrackerFeature.enabled)
        {
            eyeTrackerFeature.DestroyEyeTracker();
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

    public static Vector2 WorldPointToPixel(Vector3 worldPoint, int width, int height, MLCameraBase.IntrinsicCalibrationParameters parameters, Matrix4x4 cameraTransformationMatrix)
        {

            // Step 1: Convert the world space point to camera space
            Vector3 cameraSpacePoint = cameraTransformationMatrix.inverse.MultiplyPoint(worldPoint);

            // Step 2: Project the camera space point onto the normalized image plane
            Vector2 normalizedImagePoint = new Vector2(cameraSpacePoint.x / cameraSpacePoint.z, cameraSpacePoint.y / cameraSpacePoint.z);
            
            // Step 3: Adjust for FOV
            float verticalFOVRad = parameters.FOV * Mathf.Deg2Rad;
            float aspectRatio = width / (float)height;
            float horizontalFOVRad = 2 * Mathf.Atan(Mathf.Tan(verticalFOVRad / 2) * aspectRatio);
            // float horizontalFOVRad = 2 * Mathf.Atan(Mathf.Tan(verticalFOVRad / 2));

            normalizedImagePoint.x /= Mathf.Tan(horizontalFOVRad / 2);
            normalizedImagePoint.y /= Mathf.Tan(verticalFOVRad / 2);

            // Step 4: Convert normalized image coordinates to pixel coordinates
            // Vector2 pixelPosition = new Vector2(
            //     normalizedImagePoint.x * width + parameters.PrincipalPoint.x,
            //     normalizedImagePoint.y * height + parameters.PrincipalPoint.y
            // );
            Vector2 pixelPosition = new Vector2(
                normalizedImagePoint.x * width + width / 2,
                normalizedImagePoint.y * height + height / 2
            );

            // Debug.Log("Pixel Position: " + pixelPosition);
            return pixelPosition;
        }
    private (long cameraTime,         // 1
         Vector2 projected2DPoint,  // 2
         Vector3 topLeft3D,         // 3
         Vector3 topRight3D,        // 4
         Vector3 bottomLeft3D,      // 5
         Vector3 bottomRight3D,     // 6
         Vector3 center3D           // 7
        ) 
    ComputeProjected2DGazePoint(Vector3 _3dGazePoint, long currentGazeTime)
    {
        // Get the camera data
        if (SharedInfomanager.Instance.CameraDataQueue.Count == 0)
        {
            return (0, Vector2.zero, Vector3.zero, Vector3.zero,Vector3.zero,Vector3.zero,Vector3.zero);
        }


        while (SharedInfomanager.Instance.CameraDataQueue.TryPeek(out var newCameraData))
        {
            // Debug.Log("Syncing camera data with gaze data. Gaze Time: " + currentGazeTime + ", Camera Time: " + cameraData.Timestamp);
            long timeDiff = newCameraData.Timestamp - currentGazeTime;
            if (timeDiff < 0 || prevCameraTime == 0)
            { // pretty old, keep updating until we get a recent one
                SharedInfomanager.Instance.CameraDataQueue.TryDequeue(out cameraData);
                prevCameraTime = cameraData.Timestamp;
            }
            else
            {
                if (timeDiff >= Math.Abs(currentGazeTime - prevCameraTime)) break;
                SharedInfomanager.Instance.CameraDataQueue.TryDequeue(out cameraData);
                prevCameraTime = cameraData.Timestamp;
                break;
            }
        }
        // Debug.Log("Synced camera data with gaze data. Gaze Time: " + currentGazeTime + ", Camera Time: " + cameraData.Timestamp);

        if (prevCameraTime - currentGazeTime > timeDifferenceThreshold || currentGazeTime - prevCameraTime > timeDifferenceThreshold)
        {
            // debugLogger.Log("Camera data is too old or too new. Skipping frame.");
            return (0, Vector2.zero,Vector3.zero,Vector3.zero,Vector3.zero,Vector3.zero,Vector3.zero);
        }

        // compute the 2D gaze point for future use. Note that this is on the original size
        Vector2 _2dGazePoint = WorldToCameraPixel(_3dGazePoint, cameraData.image_width, cameraData.image_height, cameraData.intrinsicParameters, cameraData.cameraTransformMatrix);
        // Vector2 _2dGazePoint = CameraExtensions.ConvertWorldPointToScreen(_3dGazePoint, cameraData.intrinsicParameters, cameraData.cameraTransformMatrix);
        
        // debugLogger.Log("Computing 2D gaze point, image width: " + cameraData.image_width + ", image height: " + cameraData.image_height);
        return (cameraData.Timestamp, _2dGazePoint, cameraData.TopLeftPosition_3D,cameraData.TopRightPosition_3D,cameraData.BottomLeftPostion_3D,cameraData.BottomRightPositon_3D,cameraData.CenterPosition_3D);

    }

    public static Vector2 WorldToCameraPixel(
        Vector3 worldPoint, int width, int height, MLCameraBase.IntrinsicCalibrationParameters parameters, Matrix4x4 cameraTransformationMatrix)
    {
   // ─────────────────────────────────────────────
    // Step 1: Convert the world space point to camera space.
    // ─────────────────────────────────────────────
    Matrix4x4 invMatrix = cameraTransformationMatrix.inverse;
    Vector3 camSpacePoint = invMatrix.MultiplyPoint(worldPoint);

    // If Z <= 0, the point is behind the camera or extremely close.
    // We can return an invalid coordinate (e.g., -1,-1) to indicate that.
    if (camSpacePoint.z <= 1e-6f)
    {
        return new Vector2(-1f, -1f);
    }

    // ─────────────────────────────────────────────
    // Step 2: Project the camera space point onto
    //         the normalized pinhole image plane.
    //         x_norm = X / Z,  y_norm = Y / Z
    // ─────────────────────────────────────────────
    float xNorm = camSpacePoint.x / camSpacePoint.z;
    float yNorm = camSpacePoint.y / camSpacePoint.z;

    // ─────────────────────────────────────────────
    // Step 3: Apply lens distortion (radial + tangential).
    //         If you want an *undistorted* pixel, skip this step.
    // ─────────────────────────────────────────────
    // Distortion array = [k1, k2, p1, p2, k3]
    double[] distCoeffs = parameters.Distortion;
    float k1 = (float)distCoeffs[0];
    float k2 = (float)distCoeffs[1];
    float p1 = (float)distCoeffs[2];
    float p2 = (float)distCoeffs[3];
    float k3 = (float)distCoeffs[4];

    float r2 = xNorm * xNorm + yNorm * yNorm;
    float r4 = r2 * r2;
    float r6 = r2 * r4;

    // Radial factor
    float radial = 1.0f + k1 * r2 + k2 * r4 + k3 * r6;

    // Distorted coordinates
    float xDist = xNorm * radial + 2f * p1 * xNorm * yNorm + p2 * (r2 + 2f * xNorm * xNorm);
    float yDist = yNorm * radial + p1 * (r2 + 2f * yNorm * yNorm) + 2f * p2 * xNorm * yNorm;

    // ─────────────────────────────────────────────
    // Step 4: Convert from normalized coords to pixel coords
    //         using focal length + principal point from intrinsics.
    // ─────────────────────────────────────────────
    float fx = parameters.FocalLength.x;  // focal length in X dimension
    float fy = parameters.FocalLength.y;  // focal length in Y dimension
    float cx = parameters.PrincipalPoint.x;
    float cy = parameters.PrincipalPoint.y;

    // Pixel coordinates (distorted)
    float pixelX = fx * xDist + cx;
    float pixelY = fy * yDist + cy;

    // Depending on how principal point and image coordinates are defined
    // you may or may not need "height - cy". Magic Leap intrinsics typically
    // measure 'cy' from the top, so we don't invert Y here.

    // Optionally, clamp or check if pixel is in [0, width]x[0, height]
    // or just return it directly:
    return new Vector2(pixelX, pixelY);
    }

}