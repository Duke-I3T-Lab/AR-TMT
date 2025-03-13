using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;
using Unity.XR.CoreUtils;
using UnityEngine.XR.OpenXR;
using MagicLeap.OpenXR.Features.MarkerUnderstanding;
using System.IO;  // For file handling
using System.Collections;
using UnityEngine.XR.MagicLeap;
using MagicLeap.Examples;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.XR;
using UnityEngine.XR.Interaction.Toolkit;

public class MotorSpeedTest : MonoBehaviour
{
    public static MotorSpeedTest Instance { get; private set; } // Singleton instance

    public GameObject motorspeedtargetPrefab;  // Assign a prefab for the target
    private GameObject currentTarget;
    private int hitCount = 0;

    [Tooltip("total try")]
    public int totalHits ; // The number of times the user needs to hit the target
    public PlaceObjectAvoidingMesh placementHelper; // Reference to the PlaceObjectAvoidingMesh script
    [Tooltip("Set the XR Origin so that the marker appears relative to headset's origin. If null, the script will try to find the component automatically.")]
    public XROrigin XROrigin;
    public SelectionNoticeHandler selectionnoticeUI;
    public EyeTrackerLogger eyeTrackerLogger;
    public TestCameraRecording_MainCamera MainCamera;
    private string jsonFilePath;
    public List<float> completiontime_list { get; private set; } = new List<float>();
    public List<float> distance_list { get; private set; } = new List<float>();
    public float StartTime { get; private set; }
    public TestCameraRecording_CVcamera CVCamera;

    [SerializeField]
    private InputAction triggerInputAction =
        new InputAction(binding: "<XRController>/trigger", expectedControlType: "Button");
    
    [SerializeField]
    private InputAction positionInputAction =
        new InputAction(binding: "<MagicLeapController>/pointerPosition", expectedControlType: "Vector3");

    [SerializeField]
    private InputAction rotationInputAction =
        new InputAction(binding: "<MagicLeapController>/pointerRotation", expectedControlType: "Quaternion");

    private bool hasTriggered = false;
    public AudioClip soundClip;
    private AudioSource audioSource;
    public bool automaticupload;
    [SerializeField] private DataUploader uploader;
    [SerializeField] private string serverUrl = "http://192.168.1.23:5000/upload";

    void Start()
    {
        positionInputAction.Enable();
        rotationInputAction.Enable();
        triggerInputAction.Enable();

        triggerInputAction.performed += ActionOnPerformed;
        triggerInputAction.canceled += ActionOnCanceled;


            audioSource = GetComponent<AudioSource>();
        // If none exists, add one dynamically.
        if (audioSource == null)
        {
            audioSource = gameObject.AddComponent<AudioSource>();
        }
    }
    private void OnDestroy()
    {
        // Dispose input actions
        triggerInputAction.Dispose();
        positionInputAction.Dispose();
        rotationInputAction.Dispose();
    }
     private void ActionOnPerformed(InputAction.CallbackContext obj)
    {
        if (!SharedInfomanager.Instance.IsMotorTestActive)
        {
            Debug.Log("Task is not active. Ignoring start button press.");
            return;
        }

        if (!hasTriggered)
        {
            hasTriggered = true;

            // Get the pointer position and rotation
            Vector3 rayOrigin = positionInputAction.ReadValue<Vector3>();
            Quaternion rayRotation = rotationInputAction.ReadValue<Quaternion>();
            Vector3 rayDirection = rayRotation * Vector3.forward;

            Debug.Log($"Trigger pressed! Ray origin: {rayOrigin}, Ray direction: {rayDirection}");
            
            // Make sound effect
            audioSource.PlayOneShot(soundClip,1.0f);
            
            // Perform a raycast from the pointer position
            if (Physics.Raycast(rayOrigin, rayDirection, out RaycastHit hit, 10f))
            {
                Debug.Log($"Raycast hit: {hit.collider.gameObject.name}");

                if (hit.collider != null && hit.collider.CompareTag("MotorTarget"))
                {
                    Vector3 targetCenter = hit.collider.bounds.center;
                    Vector3 hitPoint = hit.point;
                    float distanceToCenter = Vector3.Distance(targetCenter, hitPoint);
                    float completion_time =Time.time- StartTime;
                    
                    Debug.Log($"Target hit! Distance from center: {distanceToCenter} meters");
                    completiontime_list.Add(completion_time);
                    distance_list.Add(distanceToCenter);                
                    
                    OnTargetHit();

                    Debug.Log($" Destroying {hit.collider.gameObject.name}");
                    Destroy(hit.collider.gameObject); // Destroy the actual hit object

                    if (currentTarget == hit.collider.gameObject)
                    {
                        Debug.Log($" Resetting currentTarget to null");
                        currentTarget = null; //  Reset after destruction
                    }
                }
            }
        }
    }
    private void ActionOnCanceled(InputAction.CallbackContext obj)
    {
        hasTriggered = false; // Reset trigger state when button is released
    }
    public void StartMotorSpeedTest()
    {

        // fix the locations 
        Random.InitState(12345);

        // Debug.Log("Press motorspeed test start button");
        SharedInfomanager.Instance.StartMotorSpeedTask();
        hitCount = 0;
        GenerateTarget();
    }

    private void GenerateTarget()
    {
        if (currentTarget != null)
        {
            Debug.Log($"Destroying old target: {currentTarget.name}");

            Destroy(currentTarget); // Remove the old target before spawning a new one
        }
        Pose markerPose = SharedInfomanager.Instance.MarkerPosition;
        Vector3 targetSize = GetTargetSize(motorspeedtargetPrefab);
        float markerSize = SharedInfomanager.Instance.MarkerSize;
        Vector3 userPos = XROrigin.Camera.transform.position;

        // Generate a random position within the given range
        Vector3 randomposition = placementHelper.GetValidPosition_fixed(userPos, markerPose.position, targetSize, 0.3f, 30f, 1f, 3f);

        // Instantiate the new target at the calculated position
        currentTarget = Instantiate(motorspeedtargetPrefab, randomposition, Quaternion.identity);
        currentTarget.transform.localScale = new Vector3(markerSize, currentTarget.transform.localScale.y, markerSize);
        currentTarget.transform.LookAt(XROrigin.Camera.transform.position);
        currentTarget.transform.Rotate(90, 0, 0);
        currentTarget.SetActive(true);

        StartTime=Time.time;
    }

    public void OnTargetHit()
    {
        hitCount++;

        if (hitCount < totalHits)
        {
            GenerateTarget(); // Spawn a new target
        }
        else
        {
            Debug.Log("Motor speed test completed!");
            SaveData();
            SharedInfomanager.Instance.EndMotorSpeedTask();
            StartCoroutine(DelayedNoticeGeneration());
        }
    }
    private IEnumerator DelayedNoticeGeneration()
    {
        yield return new WaitForSeconds(2f); // Wait for 2 seconds

        // Next task description
        selectionnoticeUI.selection_noticegeneration();
    }
    public void SaveData()
    {
        var performanceData = new Dictionary<string, object>
        {
            { "Distance_list", distance_list.ToArray() },  // Convert list to array for JSON serialization
            { "Completion_time_list", completiontime_list.ToArray() }
        };

        // Generate a file path for the result
 
        
        jsonFilePath = SharedInfomanager.Instance.GenerateUniqueFilePath("Performancedata_task", 0, "json");

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



        if (automaticupload)
        {
            if (uploader == null)
            {
                Debug.LogError("Uploader is null!!");
            }
            else{
            Debug.Log("Data Transmission Triggered");
            string userFolderPath=Path.Combine(Application.persistentDataPath, $"User{SharedInfomanager.Instance.userFolderCounter.ToString("D3")}");
            
            string path_performancedata=Path.Combine(userFolderPath, $"Performancedata_task0.json");            
            uploader.UploadData(path_performancedata,serverUrl);

            }
        }




    }
    private Vector3 GetTargetSize(GameObject targetPrefab)
    {
        Renderer renderer = targetPrefab.GetComponentInChildren<Renderer>();
        if (renderer != null)
        {
            return renderer.bounds.size;
        }
        else
        {
            Debug.LogError("Renderer not found on TargetPrefab. Using default size.");
            return new Vector3(0.2f, 0.2f, 0.2f); // Fallback size
        }
    }
    

}