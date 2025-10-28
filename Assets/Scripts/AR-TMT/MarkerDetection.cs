using Unity.XR.CoreUtils;
using UnityEngine;
using UnityEngine.XR.OpenXR;
using MagicLeap.OpenXR.Features.MarkerUnderstanding;
using System;

public class MarkerDetection : MonoBehaviour
{
    [Tooltip("Set the XR Origin so that the marker appears relative to headset's origin. If null, the script will try to find the component automatically.")]
    public XROrigin XROrigin ;

    [Tooltip("Specify the Aruco marker ID to detect.")]
    public int TargetArucoID = 1; // Set this to the ID of the marker you want to detect

    public ArucoType ArucoType = ArucoType.Dictionary_4x4_50;
    public MarkerDetectorProfile DetectorProfile = MarkerDetectorProfile.Speed;

    private MarkerDetectorSettings _detectorSettings;
    private MagicLeapMarkerUnderstandingFeature _markerFeature;
    private MarkerDetector _markerDetector;
    public SelectionNoticeHandler SelectionNoticeHandler;
    // Reference to your AR camera (the one the user sees)

    public static event Action OnMarkerDetectionDestroyed; // Event to signal destruction
    private void OnValidate()
    {
        // Automatically find the XROrigin component if it's present in the scene
        if (XROrigin == null)
        {
            XROrigin = FindAnyObjectByType<XROrigin>();
        }
    }

    private void Start()
    {
        _markerFeature = OpenXRSettings.Instance.GetFeature<MagicLeapMarkerUnderstandingFeature>();

        if (_markerFeature == null || !_markerFeature.enabled)
        {
            Debug.LogError("The Magic Leap 2 Marker Understanding OpenXR Feature is missing or disabled. Disabling script.");
            return;
        }

        if (XROrigin == null)
        {
            Debug.LogError("No XR Origin found; marker tracking will not work. Disabling script.");
            return;
        }


        // Create the Marker Detector Settings
        _detectorSettings = new MarkerDetectorSettings
        {
            MarkerDetectorProfile = DetectorProfile,
            MarkerType = MarkerType.Aruco,
            ArucoSettings = new ArucoSettings
            {
                ArucoType = ArucoType,
                EstimateArucoLength = false,
                ArucoLength = 0.11f
                
            }
        };

        // Create the Marker Detector with the settings above
        _markerDetector = _markerFeature.CreateMarkerDetector(_detectorSettings);



        
    }

    private void OnDestroy()
    {
        
        _markerFeature.DestroyAllMarkerDetectors();
        
    }
    private bool markerProcessed = false; // Flag to track if the marker has been processed

    private void Update()
    {
        if (markerProcessed)
        {
            // Skip further detection if the marker has already been processed
            return;
        }

        // Check if _markerFeature and _markerDetector are initialized
        if (_markerFeature == null || _markerDetector == null)
        {
            Debug.LogError("_markerFeature or _markerDetector is null. Ensure the feature is enabled and the detector is created.");
            return;
        }




        Debug.Log($"Marker Detector Status: {_markerDetector.Status} ");

        // Proceed with the marker detection update if initialized
        
        _markerFeature.UpdateMarkerDetectors();

        if (_markerDetector.Status == MarkerDetectorStatus.Ready)
        {
            foreach (var data in _markerDetector.Data)
            {
                if (data.MarkerPose.HasValue && data.MarkerNumber == (ulong)TargetArucoID)
                {
                    var position = data.MarkerPose.Value.position;
                    var rotation = data.MarkerPose.Value.rotation;

                    if (data.MarkerNumber == 0 || position == Vector3.zero || rotation == Quaternion.identity)
                    {
                        continue; // Skip processing this marker
                    }

                    Debug.Log($"Detected Marker ID: {data.MarkerNumber}, Pose: {data.MarkerPose}");
                    DetectMarkerAndInitialize(data.MarkerPose.Value, data.MarkerLength);
                    return;
                }
            }
        }
    }
    
    private void DetectMarkerAndInitialize(Pose markerPose, float markerSize)
    {
        if (markerProcessed)
        {
            return; // Prevent re-processing the marker
        }

        markerProcessed = true; // Set the flag to prevent multiple calls
        Vector3 userPos = XROrigin.Camera.transform.position;
        Vector3 centerDirection = markerPose.position - userPos;

        SharedInfomanager.Instance.SetMarkerData(markerPose, markerSize, centerDirection);
        Debug.Log("Marker detected. Storing marker data.");



        //   Show Notice UI
        if (SelectionNoticeHandler.SelectionUI != null)
        {
            // Activate the Notice UI parent
            SelectionNoticeHandler.selection_noticegeneration();
            Debug.Log($"Notice UI displayed at position: {SelectionNoticeHandler.SelectionUI.transform.position}");

        }

        SharedInfomanager.Instance.InitializeUserFolderCounter();

        OnMarkerDetectionDestroyed?.Invoke();

        // Destroy this component
        Destroy(this);
        
        Debug.Log("Script has been disabled after detecting the marker.");
    }
}