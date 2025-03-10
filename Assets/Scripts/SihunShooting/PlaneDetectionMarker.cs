using System.Collections.Generic;
using Unity.XR.CoreUtils;
using UnityEngine;
using UnityEngine.XR.OpenXR;
using MagicLeap.OpenXR.Features.MarkerUnderstanding;
using UnityEngine.XR.Management;
using System;

public class PlaneDetectionMarker : MonoBehaviour
{
    [Header("Marker Understanding Feature")]
    public XROrigin XROrigin;
    public ArucoType ArucoType = ArucoType.Dictionary_4x4_50;
    public MarkerDetectorProfile DetectorProfile = MarkerDetectorProfile.Accuracy;
    public float DefaultMarkerLength = 0.03f;

    [Header("Marker Visualization")]
    public GameObject MarkerPrefab; // ✅ Cube for marker visualization

    [Header("Marker IDs")]
    public int TopMarkerID = 1;   // ✅ Identical ID for all upper markers
    public int BottomMarkerID = 2; // ✅ Identical ID for all lower markers

    private MagicLeapMarkerUnderstandingFeature _markerFeature;
    private MarkerDetector _markerDetector;
    private MarkerDetectorSettings _detectorSettings;

    private Dictionary<ulong, List<Pose>> storedMarkerInstances = new Dictionary<ulong, List<Pose>>(); // ✅ Store multiple instances per marker ID
    private Dictionary<ulong, List<GameObject>> markerVisuals = new Dictionary<ulong, List<GameObject>>(); // ✅ Store visual cubes
    private List<GameObject> instantiatedWalls = new List<GameObject>();

    private const float duplicateThreshold = 0.2f; // ✅ 20cm threshold to prevent duplicate markers
    private const float MarkerSize = 0.05f; // ✅ Size of cube visualization
    private bool detectionEnabled = false; // ✅ Control flag for marker detection

    private void Start()
    {

    }




    
    public void StartPlaneDetection()
    {
        if (detectionEnabled)
        {
            Debug.LogWarning("[Marker Detection] Already started.");
            return;
        }

        _markerFeature = OpenXRSettings.Instance.GetFeature<MagicLeapMarkerUnderstandingFeature>();
        if (_markerFeature == null || !_markerFeature.enabled)
        {
            Debug.LogError("MagicLeapMarkerUnderstandingFeature is missing or disabled.");
            enabled = false;
            return;
        }

        if (XROrigin == null)
        {
            Debug.LogError("No XR Origin found; marker tracking will not work.");
            enabled = false;
            return;
        }

        _detectorSettings = new MarkerDetectorSettings
        {
            MarkerDetectorProfile = DetectorProfile,
            MarkerType = MarkerType.Aruco,
            ArucoSettings = new ArucoSettings
            {
                ArucoType = ArucoType,
                EstimateArucoLength = false,
                ArucoLength = DefaultMarkerLength
            }
        };

        _markerDetector = _markerFeature.CreateMarkerDetector(_detectorSettings);
        detectionEnabled = true; // ✅ Enable detection

        Debug.Log("Marker Detector created.");
    }
    private void Update()
    {
        if (!detectionEnabled) return; 

        if (_markerFeature == null || _markerDetector == null)
            return;

        _markerFeature.UpdateMarkerDetectors();

        if (_markerDetector.Status != MarkerDetectorStatus.Ready)
            return;

        bool markerUpdated = false;

        foreach (var data in _markerDetector.Data)
        {
            if (!data.MarkerNumber.HasValue || !data.MarkerPose.HasValue)
                continue;

            ulong markerId = data.MarkerNumber.Value;
            Pose newPose = data.MarkerPose.Value;

            // ✅ Initialize storage for new marker IDs if not already present
            if (!storedMarkerInstances.ContainsKey(markerId))
            {
                storedMarkerInstances[markerId] = new List<Pose>();
                markerVisuals[markerId] = new List<GameObject>();
            }

            bool updated = false;

            // ✅ Check existing markers to update or detect new ones
            for (int i = 0; i < storedMarkerInstances[markerId].Count; i++)
            {
                Pose storedPose = storedMarkerInstances[markerId][i];
                float distance = Vector3.Distance(storedPose.position, newPose.position);

                // ✅ If the marker is within correction range, update its position
                if (distance < duplicateThreshold)
                {
                    Debug.Log($"[Marker Update] Marker {markerId} updated at index {i}");

                    // 🔥 Directly update marker position and rotation
                    storedMarkerInstances[markerId][i] = newPose;

                    if (markerVisuals[markerId].Count > i)
                    {
                        markerVisuals[markerId][i].transform.position = newPose.position;
                        markerVisuals[markerId][i].transform.rotation = newPose.rotation;
                    }

                    updated = true;
                    markerUpdated = true;
                    break;
                }
            }

            // ✅ If the marker is at a new location and we have space, add it as a new instance
            if (!updated && storedMarkerInstances[markerId].Count < 4)
            {
                storedMarkerInstances[markerId].Add(newPose);
                Debug.Log($"[Marker Detected] New marker {markerId} detected at {newPose.position}");

                // ✅ Instantiate new visual marker
                GameObject newMarker = Instantiate(MarkerPrefab, newPose.position, newPose.rotation);
                newMarker.transform.localScale = Vector3.one * MarkerSize;
                markerVisuals[markerId].Add(newMarker);

                markerUpdated = true;
            }
        }

        // ✅ Update marker visuals **immediately** when a marker is detected or updated
        if (markerUpdated)
        {
            UpdateMarkerVisuals();
        }

        // ✅ Ensure **all 4 top and 4 bottom markers** are detected before proceeding
        if (storedMarkerInstances.ContainsKey((ulong)TopMarkerID) &&
            storedMarkerInstances.ContainsKey((ulong)BottomMarkerID) &&
            storedMarkerInstances[(ulong)TopMarkerID].Count == 4 &&
            storedMarkerInstances[(ulong)BottomMarkerID].Count == 4)
        {
            UpdateWallsFromMarkers();
            detectionEnabled = false; // ✅ Stop further updates after detection
        }
    }

    private void UpdateWallsFromMarkers()
    {
        if (!storedMarkerInstances.ContainsKey((ulong)TopMarkerID) ||
            !storedMarkerInstances.ContainsKey((ulong)BottomMarkerID))
        {
            return;
        }

        List<Pose> topMarkers = storedMarkerInstances[(ulong)TopMarkerID];
        List<Pose> bottomMarkers = storedMarkerInstances[(ulong)BottomMarkerID];

        Debug.Log($"[Wall Debug] Top markers detected: {topMarkers.Count}, Bottom markers detected: {bottomMarkers.Count}");

        // ✅ Ensure exactly 4 markers are present for both types
        if (topMarkers.Count < 4 || bottomMarkers.Count < 4)
        {
            Debug.Log("Waiting for all 4 top and 4 bottom markers...");
            return;
        }

        List<Tuple<Pose, Pose>> validWallPairs = FindValidWallPairs(topMarkers, bottomMarkers);

        Debug.Log($"[Wall Debug] Found {validWallPairs.Count} valid wall pairs.");

        if (validWallPairs.Count < 4)
        {
            Debug.LogWarning($"Only {validWallPairs.Count} walls detected. Waiting for all 4.");
            return;
        }
        
        List<SharedInfomanager.WallData> wallDataList = new List<SharedInfomanager.WallData>();

        for (int i = 0; i < validWallPairs.Count; i++)
        {
            Pose topMarker = validWallPairs[i].Item1;
            Pose bottomMarker = validWallPairs[i].Item2;
            SharedInfomanager.WallData newWallData = BuildWallDataFromMarkers(topMarker, bottomMarker);
            wallDataList.Add(newWallData);
            
            // If an existing wall exists, destroy it and create a new one
            if (i < instantiatedWalls.Count)
            {
                if (instantiatedWalls[i] != null)
                {
                    Destroy(instantiatedWalls[i]);
                }
                GameObject newWall = InstantiateWall(newWallData);
                instantiatedWalls[i] = newWall;
            }
            else
            {
                GameObject newWall = InstantiateWall(newWallData);
                instantiatedWalls.Add(newWall);
            }
        }
        // ✅ Save the detected walls using SharedInfoManager
        SharedInfomanager.SaveWalls(wallDataList);        

        // ✅ Print the saved walls
        Debug.Log("[Wall Debug] Saved Walls:");
        foreach (var wall in SharedInfomanager.SavedWalls)
        {
            Debug.Log($"Wall - Center: {wall.Center}, Normal: {wall.Normal}, Width: {wall.Width}, Height: {wall.Height}");
        }


        // ✅ Stop further marker detection
        _markerFeature.DestroyAllMarkerDetectors();
        _markerDetector = null;
        enabled = false; // Disable script to prevent further updates
        // Destroy(this);

    }

    private void UpdateMarkerVisuals()
    {
        foreach (var entry in storedMarkerInstances)
        {
            ulong markerId = entry.Key;
            List<Pose> markerPoses = entry.Value;

            if (!markerVisuals.ContainsKey(markerId))
            {
                markerVisuals[markerId] = new List<GameObject>();
            }

            for (int i = 0; i < markerPoses.Count; i++)
            {
                Pose markerPose = markerPoses[i];

                if (i >= markerVisuals[markerId].Count)
                {
                    GameObject markerCube = Instantiate(MarkerPrefab);
                    markerCube.transform.localScale = Vector3.one * MarkerSize;
                    markerVisuals[markerId].Add(markerCube);
                }

                markerVisuals[markerId][i].transform.position = markerPose.position;
                markerVisuals[markerId][i].transform.rotation = markerPose.rotation;

                Renderer renderer = markerVisuals[markerId][i].GetComponent<Renderer>();
                if (renderer != null)
                {
                    renderer.material.color = (markerId == (ulong)TopMarkerID) ? Color.blue : Color.green;
                }
            }
        }
    }

    private List<Tuple<Pose, Pose>> FindValidWallPairs(List<Pose> topMarkers, List<Pose> bottomMarkers)
    {
        List<Tuple<Pose, Pose>> validWallPairs = new List<Tuple<Pose, Pose>>();
        HashSet<Pose> usedBottomMarkers = new HashSet<Pose>();

        const float expectedWidth = 0.52f;  // 🔹 50cm expected width
        const float expectedHeight = 1.7f; // 🔹 170cm expected height
        const float positionTolerance = 0.2f; // 🔹 Allow slight deviation (5cm)

        foreach (var topMarker in topMarkers)
        {
            Pose bestMatch = default;
            float bestScore = float.MaxValue;

            foreach (var bottomMarker in bottomMarkers)
            {
                if (usedBottomMarkers.Contains(bottomMarker))
                    continue; // Prevent reusing the same bottom marker

                // 🔹 Compute displacement between markers
                Vector3 displacement = bottomMarker.position - topMarker.position;

                // ✅ **Define a local frame based on marker orientations**
                Vector3 up = Vector3.up; // Always Y-axis for height
                Vector3 forward = (topMarker.rotation * Vector3.forward).normalized;
                Vector3 right = Vector3.Cross(up, forward).normalized; // Perpendicular to forward

                // ✅ **Project displacement onto this local frame**
                float projectedWidth = Mathf.Abs(Vector3.Dot(displacement, right));
                float projectedHeight = Mathf.Abs(Vector3.Dot(displacement, up));

                // ✅ **Check if width & height are within tolerance**
                bool isWidthCorrect = Mathf.Abs(projectedWidth - expectedWidth) < positionTolerance;
                bool isHeightCorrect = Mathf.Abs(projectedHeight - expectedHeight) < positionTolerance;

                if (!isWidthCorrect || !isHeightCorrect)
                    continue; // Skip misaligned markers

                // ✅ **Select the best match based on total distance**
                float distance = displacement.magnitude;
                if (distance < bestScore)
                {
                    bestMatch = bottomMarker;
                    bestScore = distance;
                }
            }

            if (bestMatch != default)
            {
                validWallPairs.Add(new Tuple<Pose, Pose>(topMarker, bestMatch));
                usedBottomMarkers.Add(bestMatch);
            }
            else
            {
                Debug.LogWarning($"[Wall Debug] No bottom marker found for top marker at {topMarker.position}!");
            }
        }

        return validWallPairs;
    }


    private SharedInfomanager.WallData BuildWallDataFromMarkers(Pose topPose, Pose bottomPose)
    {
        // Compute the vector from bottom marker to top marker
        Vector3 markerVector = topPose.position - bottomPose.position;

        // Compute midpoint correctly (midpoint between top and bottom marker)
        Vector3 midpoint = bottomPose.position + (markerVector * 0.5f);

        // Ensure width and height match the expected physical dimensions
        float width = 0.53f; // 50 cm ± 5cm tolerance
        float height = 1.7f; // 170 cm ± 5cm tolerance
    
        // Compute a normal from positions: perpendicular to the vertical marker vector.
        Vector3 normalFromPositions = Vector3.Cross(markerVector.normalized, Vector3.up).normalized;
        
        // Compute a normal from the marker's orientation.
        // Assuming the marker's forward (local Z) points perpendicularly from the wall.
        Vector3 markerNormal = topPose.rotation * Vector3.back;
        // markerNormal.y = 0f; // Force horizontal normal.
        markerNormal.Normalize();
        Vector3 blendedNormal = (normalFromPositions + markerNormal).normalized;

        Debug.Log($"[Wall Debug] Midpoint: {midpoint}, Width: {width}, Height: {height}, Normal Normal: {normalFromPositions}");

        return new SharedInfomanager.WallData(midpoint, normalFromPositions, width, height);
    }

    /// <summary>
    /// Instantiates a wireframe wall using LineRenderers.
    /// The wireframe is generated as a box with dimensions defined by wallData.
    /// </summary>
    private GameObject InstantiateWall(SharedInfomanager.WallData wallData)
    {
    // Create an empty parent GameObject for the wireframe wall.
    GameObject wireframeWall = new GameObject("WireframeWall");
    wireframeWall.transform.position = wallData.Center;
    // Set the wall's rotation so that its local XY plane aligns with the wall plane.
    Quaternion rotation = Quaternion.LookRotation(-wallData.Normal, Vector3.up);
    wireframeWall.transform.rotation = rotation;

    // Add a LineRenderer to draw the rectangle.
    LineRenderer lr = wireframeWall.AddComponent<LineRenderer>();
    lr.useWorldSpace = false; // Use local space so the defined positions are relative to the wireframeWall.
    lr.loop = true;
    lr.positionCount = 4; // Four corners define the rectangle.
    lr.startWidth = 0.005f;
    lr.endWidth = 0.005f;
    lr.material = new Material(Shader.Find("Sprites/Default"));
    lr.startColor = Color.red;
    lr.endColor = Color.red;

    // Calculate the four corners in local space (the rectangle lies on the XY plane, with Z = 0).
    Vector3 bottomLeft = new Vector3(-wallData.Width / 2f, -wallData.Height / 2f, 0f);
    Vector3 bottomRight = new Vector3(wallData.Width / 2f, -wallData.Height / 2f, 0f);
    Vector3 topRight = new Vector3(wallData.Width / 2f, wallData.Height / 2f, 0f);
    Vector3 topLeft = new Vector3(-wallData.Width / 2f, wallData.Height / 2f, 0f);

    // Set the positions of the LineRenderer (they are defined in the wall's local space).
    lr.SetPosition(0, bottomLeft);
    lr.SetPosition(1, bottomRight);
    lr.SetPosition(2, topRight);
    lr.SetPosition(3, topLeft);

    return wireframeWall;
    }


    public void DestroyGeneratedMarkersAndWalls()
    {
        Debug.Log("[Cleanup] Destroying all detected walls and markers...");

        // ✅ Destroy all instantiated walls
        foreach (GameObject wall in instantiatedWalls)
        {
            if (wall != null)
            {
                Destroy(wall);
            }
        }
        instantiatedWalls.Clear();

        // ✅ Destroy all instantiated marker cubes
        foreach (var markerList in markerVisuals.Values)
        {
            foreach (GameObject marker in markerList)
            {
                if (marker != null)
                {
                    Destroy(marker);
                }
            }
        }
        markerVisuals.Clear();

        // ✅ Clear all stored marker data
        storedMarkerInstances.Clear();

        Debug.Log("[Cleanup] All walls and markers have been removed.");
    }

    public void OnRescanButtonClicked()
    {
        // 1) Clean up all previously detected markers and walls
        DestroyGeneratedMarkersAndWalls();
        Debug.Log("OnRescanButtonClicked triggered");
        
        // 2) KEEP the existing marker detector alive; do NOT destroy it.
        //    (Comment out or remove these lines)
        // if (_markerFeature != null)
        // {
        //     _markerFeature.DestroyAllMarkerDetectors();
        //     _markerDetector = null;
        // }

        // 3) Optionally, keep this script enabled and detectionEnabled set to true 
        //    so the marker detection continues running.
        //    (Comment out or remove if you do NOT want to re-init detection.)
        // this.enabled = true;
        // detectionEnabled = false;
        // StartPlaneDetection();
    }

}
