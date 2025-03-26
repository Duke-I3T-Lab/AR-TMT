using System.Collections.Generic;
using UnityEngine;
using System.IO;  // For file handling
using System.Collections;
using UnityEngine.XR.MagicLeap;
using MagicLeap.Examples;
using System.Linq;
using System.Threading.Tasks;
using System;

public class SharedInfomanager : MonoBehaviour
{
    public static SharedInfomanager Instance { get; private set; } // Singleton instance


    // Marker data
    public Pose MarkerPosition { get; private set; }
    public float MarkerSize { get; private set; }
    public Vector3 Markerdirection { get; private set; }

    public bool IsMarkerDetected { get; private set; }
    
    // Sequence of targets to be hit
    public List<object> TargetHitSequence { get; private set;  }
    private int currentSequenceIndex = 0;

    // Predefined sequences for five generations
    private readonly List<List<object>> predefinedTargetSequences = new List<List<object>>
    {
        // 1. TMT-A(Baseline)
        GenerateSequence(1, 25), 

        // 2. TMT-B
        new List<object> { 1, "A", 2, "B", 3, "C", 4, "D", 5, "E", 6, "F", 7, "G", 8, "H", 9, "I", 10, "J", 11, "K", 12, "L", 13 },

        // 3. Baseline(+neutral)
        GenerateSequence(1, 25),

        // 4. Top-down
        GenerateSequence(1, 25),

        // 5. Bottom-up
        GenerateSequence(1, 25),
        
        // 6. TMT-A in Walls
        GenerateSequence(1, 25),

        // 7. TMT-B in Walls
        new List<object> { 1, "A", 2, "B", 3, "C", 4, "D", 5, "E", 6, "F", 7, "G", 8, "H", 9, "I", 10, "J", 11, "K", 12, "L", 13 },



    };
    // List to hold 3D target and distractor locations with labels
    public List<LocationData> TargetLocations { get; private set; } = new List<LocationData>();
    public List<LocationData> DistractorLocations { get; private set; } = new List<LocationData>();

    [System.Serializable]
    public struct LocationData
    {
        public float X;
        public float Y;
        public float Z;
        public string Label;

        public LocationData(Vector3 position, string label)
        {
            X = position.x;
            Y = position.y;
            Z = position.z;
            Label = label;
        }

        public Vector3 GetVector3()
        {
            return new Vector3(X, Y, Z);
        }
    }

    public List<ShootingData> shootingdata { get; private set; } = new List<ShootingData>();
    public struct ShootingData
    {
        public string type;
        public string label;
        public float distance;
        public float time;
        public string result; 

        public ShootingData(string type, string label, float distance, float time, string result)
        {
            this.type = type;
            this.label = label;
            this.distance = distance;
            this.time = time;
            this.result = result;
        }
    }

    // Updated GenerateSequence method to return List<object>
    private static List<object> GenerateSequence(int start, int end)
    {
        List<object> sequence = new List<object>();
        for (int i = start; i <= end; i++)
        {
            sequence.Add(i);
        }
        return sequence;
    }
    // Performance metric
    public float CompletionTime { get; private set; }
    public float Current_time { get; private set; }

    public int N_hitdistractor { get; private set; }
    public int N_hitworngorder { get; private set; }
    public int N_hitmiss { get; private set; }
    private string jsonFilePath;
    public int userFolderCounter { get; private set; }  // Tracks folder counters for each user
    public bool IsTaskActive { get; private set; } = false;
    public bool IsMotorTestActive { get; private set; } = false;


    public TargetGenerator targetGenerator; // Reference to TargetGenerator script
    public EyeTrackerLogger eyeTrackerLogger;
    public TestCameraRecording_MainCamera MainCamera;

    public TestCameraRecording_CVcamera CVCamera;
    public SelectionNoticeHandler selectionnoticeUI;

    public float Time_startrecording { get; private set; }
    public float Time_endrecording { get; private set; }
    public float Time_startask { get; private set; }
    public float Time_endtask { get; private set; }


    public Queue<CameraData> CameraDataQueue = new Queue<CameraData>();
    public struct CameraData
    {
        public long Timestamp;
        public int image_width;
        public int image_height;
        public MLCameraBase.IntrinsicCalibrationParameters intrinsicParameters;
        public Matrix4x4 cameraTransformMatrix;

        public Vector3 TopLeftPosition_3D;
        public Vector3 TopRightPosition_3D;
        public Vector3 BottomLeftPostion_3D;
        public Vector3 BottomRightPositon_3D;
        public Vector3 CenterPosition_3D;


    }

    public int currentGeneration { get; set; }

    public int wall_tmtA = 6;
    public int wall_tmtB = 7;
    public static List<WallData> SavedWalls { get; private set; } = new List<WallData>();

    public static void SaveWalls(List<WallData> walls)
    {
        SavedWalls = new List<WallData>(walls);
    }
    public class WallData
    {
        public Vector3 Center { get; }
        public Vector3 Normal { get; }
        public float Width { get; }
        public float Height { get; }

        public WallData(Vector3 center, Vector3 normal, float width, float height)
        {
            Center = center;
            Normal = normal;
            Width = width;
            Height = height;
        }

    }
    public bool wall_calibrated { get; private set; } = false;

    // Method to set marker data
    public void wallcalibration(bool calibrated)
    {
        wall_calibrated = calibrated;
    }    
    
    public int startVideo = 0;

    public QuestionnaireControl questionnairecontrol;

    public bool automaticupload;
    [SerializeField] private DataUploader uploader;
    [SerializeField] private string serverUrl = "http://192.168.1.23:5000/upload";


    // Thresholds for relocating the canvas
    public float desiredDistance = 1f;         // desired distance (in meters) from the camera
    public float angleThreshold = 30f;           // if the angle between camera forward and canvas > 30 degrees, relocate
    public float distanceThreshold = 0.3f;       // allowable deviation from desiredDistance


    private void Awake()
    {
        // Ensure only one instance exists
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;
        DontDestroyOnLoad(gameObject); // Optional, keeps the instance across scenes
    }



    // Method to set marker data
    public void SetMarkerData(Pose position, float size, Vector3 direction)
    {
        MarkerPosition = position;
        MarkerSize = size;
        Markerdirection = direction;
        IsMarkerDetected = true;
        Debug.Log($"Marker data updated. Position: {position}, Size: {size}");
    }
    // Reset marker detection state (optional)
    public void ClearMarkerData()
    {
        IsMarkerDetected = false;
        MarkerPosition = Pose.identity;
        MarkerSize = 0f;
    }
    // Method to set marker data
    public void SetStartrecordingtime(float time)
    {
        Time_startrecording = time;
    }
    public void SetEndrecordingtime(float time)
    {
        Time_endrecording = time;
    }

    public void SetStarttasktime(float time)
    {
        Time_startask = time;
    }
    public void SetEndtasktime(float time)
    {
        Time_endtask= time;
    }

    // Sequence

    public void SetTargetHitSequenceByIndex(int index)
    {
        if (index >= 0 && index < predefinedTargetSequences.Count)
        {
            TargetHitSequence = new List<object>(predefinedTargetSequences[index]);
            currentSequenceIndex = 0;
            Debug.Log($"Target hit sequence set: {string.Join(", ", TargetHitSequence)}");
        }
        else
        {
            Debug.LogError("Invalid  sequence index.");
        }
    }
    // Get the next target number in the sequence
    public object GetNextExpectedTarget()
    {
        // Log the current state for debugging
        Debug.Log($"[GetNextExpectedTarget] Current index: {currentSequenceIndex}, Target sequence count: {TargetHitSequence.Count}");

        // Check if the current sequence index is within the valid range
        if (currentSequenceIndex >= 0 && currentSequenceIndex < TargetHitSequence.Count)
        {
            return TargetHitSequence[currentSequenceIndex];
        }

        // Log an error if the sequence is complete or the index is invalid
        Debug.LogError($"[GetNextExpectedTarget] Invalid index: {currentSequenceIndex}. Sequence may be complete or index is out of range.");
        return null; // Return null to indicate an error or sequence completion
    }
    // Move to the next target in the sequence
    public void AdvanceToNextTarget()
    {
        currentSequenceIndex++;
    }
    public void ClearSequence()
    {
        TargetHitSequence?.Clear();
        currentSequenceIndex = 0;
    }




    // Perofrmance measurement 
    public void Incrementhitdistractor()
    {
        N_hitdistractor++;
        Debug.Log($"increment hit distractor. {N_hitdistractor}");
    }
    public void IncrementMissHit()
    {
        N_hitmiss++;
        Debug.Log($"increment Miss hit count. {N_hitmiss}");
    }

    public void IncrementWrongorder()
    {
        N_hitworngorder++;
        Debug.Log($"increment wrong order. {N_hitworngorder}");
    }



    public void ClearPerformanceData()
    {
        Current_time = Time.time;
        N_hitdistractor = 0;
        N_hitmiss = 0;
        N_hitworngorder = 0;
    }
    public void ClearLocations()
    {
        TargetLocations.Clear();
        DistractorLocations.Clear();
    }
    public void Clearshootingdata()
    {
        shootingdata.Clear();
    }

    // Function to add a single location with a label to either targets or distractors
    public void AddLocation(Vector3 location, string label, bool isDistractor)
    {
        LocationData newLocation = new LocationData(location, label);
        if (isDistractor)
        {
            DistractorLocations.Add(newLocation);
        }
        else
        {
            TargetLocations.Add(newLocation);
        }
    }

    public string GenerateUniqueFilePath(string baseFileName, int taskIndex, string extension)
    {
        string directory = Application.persistentDataPath;

        string userFolderName = $"User{userFolderCounter.ToString("D3")}";
        string userFolderPath = Path.Combine(directory, userFolderName);

        // Ensure the user folder exists
        if (!Directory.Exists(userFolderPath))
        {
            Directory.CreateDirectory(userFolderPath);
            Debug.Log($"Created folder: {userFolderPath}");
        }

        // Generate the file path for the task
        string filePath = Path.Combine(userFolderPath, $"{baseFileName}{taskIndex}.{extension}");

        return filePath;
    }

    // Performance Data
    public void SavePerformanceData(int taskIndex)
    {
        CompletionTime = Time.time -Current_time;
        // Create a dictionary to store the final performance data
        var serializedTargets = TargetLocations.Select(loc => new { X = loc.X, Y = loc.Y, Z = loc.Z, Label = loc.Label }).ToList();
        var serializedDistractors = DistractorLocations.Select(loc => new { X = loc.X, Y = loc.Y, Z = loc.Z, Label = loc.Label }).ToList();
        
        // Serialize shooting data
        var serializedShootingData = shootingdata.Select(data => new
        {
            Type = data.type,
            Label = data.label,
            Distance = data.distance,
            Time = data.time,
            Result = data.result
        }).ToList();

        var performanceData = new Dictionary<string, object>
        {
            { "CompletionTime", CompletionTime },
            { "NumberOfHittingDistractors", N_hitdistractor },
            { "NumberOfMissHits", N_hitmiss },
            { "NumberOfWrongOrderHits", N_hitworngorder },
            { "Timestamp", System.DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss") }, // Add a timestamp for when the test ended
            { "Time_StartRecording", Time_startrecording },
            { "Time_StopRecording", Time_endrecording },
            { "Time_StartTask", Time_startask },
            { "Time_StopTask", Time_endtask },
            { "Locations_targets", TargetLocations },
            { "Locations_distractors", DistractorLocations },
            { "ShootingData", serializedShootingData } 

        };
        // Generate a file path for the result
 
        
        jsonFilePath = GenerateUniqueFilePath("Performancedata_task", taskIndex, "json");

        try
        {
            // Serialize the performance data to JSON and save it to a file
            string jsonData = Newtonsoft.Json.JsonConvert.SerializeObject(performanceData, Newtonsoft.Json.Formatting.Indented);
            File.WriteAllText(jsonFilePath, jsonData);

            Debug.Log($"Test results saved to: {jsonFilePath}");
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"Failed to save test results. Error: {ex.Message}");
        }
    }


    public void StartTask()
    {
        IsTaskActive = true;
        Debug.Log("Task started. IsTaskActive set to true.");
    }
    public void EndTask()
    {
        IsTaskActive = false;
        Debug.Log("Task ended. IsTaskActive set to false.");
    }



    public void StartMotorSpeedTask()
    {
        IsMotorTestActive = true;
        Debug.Log("Task started. IsTaskActive set to true.");
    }


    public void EndMotorSpeedTask()
    {
        IsMotorTestActive = false;
        Debug.Log("Task ended. IsTaskActive set to false.");
    }

    public void InitializeUserFolderCounter()
    {
        string directory = Application.persistentDataPath;

        // Ensure the directory exists
        if (!Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        // Get all directories that match the "UserXXX" pattern
        string[] userFolders = Directory.GetDirectories(directory, "User???");
        
        int maxFolderNumber = 0;

        foreach (string folder in userFolders)
        {
            string folderName = Path.GetFileName(folder); // Extract the folder name
            if (folderName.StartsWith("User"))
            {
                string numberPart = folderName.Substring(4); // Extract the numeric part
                if (int.TryParse(numberPart, out int folderNumber))
                {
                    maxFolderNumber = Mathf.Max(maxFolderNumber, folderNumber); // Update the max number
                }
            }
        }

        // Set the counter to the next folder number
        userFolderCounter = maxFolderNumber + 1;
        Debug.Log($"Initialized userFolderCounters to: {userFolderCounter:D3}");
        string userFolderName = $"User{userFolderCounter.ToString("D3")}";
        string userFolderPath = Path.Combine(directory, userFolderName);

        // Ensure the user folder exists
        if (!Directory.Exists(userFolderPath))
        {
            Directory.CreateDirectory(userFolderPath);
            Debug.Log($"Created folder: {userFolderPath}");
        }

    }



    public void StartTaskWithDelay()
    {
        StartCoroutine(TaskCoroutine_start());
    }

    private IEnumerator TaskCoroutine_start()
    {
        eyeTrackerLogger.StartRecording(currentGeneration);
        CVCamera.StartRecording(currentGeneration);
        yield return new WaitForSeconds(1f); 

        MainCamera.StartVideoCapture(currentGeneration);



        yield return new WaitForSeconds(2f); 
        // Wait until the video capture has started (isCapturingVideo == true)
        while (!MainCamera.isCapturingVideo)
        {
            yield return null; // Wait until the flag becomes true
        }
        Debug.Log("Starting recording and generating targets...");
        if (currentGeneration==wall_tmtA || currentGeneration==wall_tmtB)
        {
            targetGenerator.GenerateTargets(true);
        }        
        else
        {
            targetGenerator.GenerateTargets(false);
        }
        StartTask();
        // start task notification
        startVideo = 2;

        Debug.Log("Task started!");
    }
    public void FinishTask()
    {
        // end task notification
        startVideo = 1;

        StartCoroutine(eye_egocentricvideo_store());



        SavePerformanceData(currentGeneration);
        SharedInfomanager.Instance.EndTask();
        targetGenerator.InitializeTargetGeneration();

        StartCoroutine(HandleQuestionnaireAndNextSteps());
    }

    private IEnumerator eye_egocentricvideo_store()
    {
        MainCamera.StopRecording();
        yield return new WaitForSeconds(1f); // Wait for 2 seconds

        CVCamera.StopRecording();
        eyeTrackerLogger.StopRecording();

    }

    private IEnumerator HandleQuestionnaireAndNextSteps()
    {
        yield return new WaitForSeconds(1f); // Wait for 2 seconds



        // 1. Start the questionnaire.
        questionnairecontrol.StartSurvey();

        // 2. Wait until it’s finished.
        //    While the user is answering the questionnaire, 
        //    we yield each frame until 'IsFinished' is true.
        while (!questionnairecontrol.IsFinished)
        {
            yield return null;
        }

        yield return new WaitForSeconds(1f); // Wait for 2 seconds

        // 3. Once finished, continue your logic
        if (automaticupload)
        {
            if (uploader == null)
            {
                Debug.LogError("Uploader is null!");
            }
            else{
            Debug.Log("Data Transmission Triggered");
            string userFolderPath=Path.Combine(Application.persistentDataPath, $"User{userFolderCounter.ToString("D3")}");
            
            string path_eyetracking=Path.Combine(userFolderPath, $"eyetracking_task{currentGeneration}.json");
            string path_cameraframe=Path.Combine(userFolderPath, $"framedata_task{currentGeneration}.json");
            string path_egocentric=Path.Combine(userFolderPath, $"egocentric_vdieo{currentGeneration}.mp4");
            string path_performancedata=Path.Combine(userFolderPath, $"Performancedata_task{currentGeneration}.json");
            string path_survey=Path.Combine(userFolderPath, $"Survey_task{currentGeneration}.csv");
            
            uploader.UploadData(path_eyetracking,serverUrl);
            uploader.UploadData(path_cameraframe,serverUrl);
            uploader.UploadData(path_egocentric,serverUrl);
            uploader.UploadData(path_performancedata,serverUrl);
            uploader.UploadData(path_survey,serverUrl);
            }
        }
        // 4. Then, start the next step (or description):
        selectionnoticeUI.selection_noticegeneration();
    }

    public void initializeUIposition(GameObject gameobject, float scale)
    {

        // Position the canvas desiredDistance in front of the camera
        Transform camTransform = Camera.main.transform;
        
        // 1. Get camera position (head-level) and “flatten” camera forward
        Vector3 camPos = camTransform.position;
        Vector3 horizontalForward = Vector3.ProjectOnPlane(camTransform.forward, Vector3.up).normalized;

        // 2. Position the notice so that its center is at the user’s head height
        Vector3 newPos = new Vector3(camPos.x, camPos.y, camPos.z) + horizontalForward * desiredDistance;
        gameobject.transform.position = newPos;

        // 3. Rotate so that it faces the camera on a purely horizontal plane
        Vector3 lookDir = newPos - camPos;    // direction from camera to UI
        lookDir.y = 0f;                       // ignore camera pitch
        gameobject.transform.rotation = Quaternion.LookRotation(lookDir.normalized, Vector3.up);

        gameobject.transform.localScale = Vector3.one * scale; // adjust scale as needed

    }



    public void UpdateUIposition(GameObject gameobject)
    {
        // Only reposition if the canvas is active
        if (!gameobject.activeSelf) return;

        Transform camTransform = Camera.main.transform;
        Vector3 camPos = camTransform.position;
        Vector3 camForward = camTransform.forward;

        // Compute direction and distance from the camera to the canvas
        Vector3 directionToCanvas = gameobject.transform.position - camPos;
        float currentDistance = directionToCanvas.magnitude;
        directionToCanvas.Normalize();

        // Compute the angle between camera's forward direction and the direction to the canvas
        float angle = Vector3.Angle(camForward, directionToCanvas);

        // Check if the canvas is out of view (angle too large) or at the wrong distance
        if (angle > angleThreshold || Mathf.Abs(currentDistance - desiredDistance) > distanceThreshold)
        {
            // 1) Flatten camera forward on the horizontal plane (ignore tilt)
            Vector3 horizontalForward = Vector3.ProjectOnPlane(camForward, Vector3.up).normalized;

            // 2) Position the notice at the camera's Y-level, desiredDistance away
            Vector3 newPos = camPos + horizontalForward * desiredDistance;
            gameobject.transform.position = newPos;

            // 3) Rotate the notice so it faces the camera horizontally, staying upright
            Vector3 lookDir = newPos - camPos;
            lookDir.y = 0f; // ignore pitch
            gameobject.transform.rotation = Quaternion.LookRotation(lookDir.normalized, Vector3.up);
        }
    }
}
