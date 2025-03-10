using System.Collections;
using System.Collections.Generic;

using UnityEngine;
using UnityEngine.XR;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.MagicLeap;
using UnityEngine.XR.Management;
using UnityEngine.XR.OpenXR;
using MagicLeap.Android;
using MagicLeap.OpenXR.Features.Meshing;

public class PlaceObjectAvoidingMesh : MonoBehaviour
{


    [SerializeField]
    private ARMeshManager meshManager;
    private MagicLeapMeshingFeature meshingFeature;


    // Search for the Mesh Manager and assign it automatically if it was not set in the inspector    // Search for the Mesh Manager and assign it automatically if it was not set in the inspector
    private void OnValidate()
    {
        if (meshManager == null)
        {
            meshManager = FindObjectOfType<ARMeshManager>();
        }
    }

    IEnumerator Start()
    {
        // Check if the ARMeshManager component is assigned
        if (meshManager == null)
        {
            Debug.LogError("No ARMeshManager component found. Disabling script.");
            enabled = false;
            yield break;
        }

        // Disable the mesh manager until permissions are granted
        meshManager.enabled = false;

        yield return new WaitUntil(IsMeshingSubsystemLoaded);

        // Access the Magic Leap Meshing Feature
        meshingFeature = OpenXRSettings.Instance.GetFeature<MagicLeapMeshingFeature>();
        if (meshingFeature == null || !meshingFeature.enabled)
        {
            Debug.LogError("MagicLeapMeshingFeature was not enabled. Disabling script.");
            enabled = false;
            yield break;
        }

        // Request permissions
        Permissions.RequestPermission(Permissions.SpatialMapping, OnPermissionGranted, OnPermissionDenied, OnPermissionDenied);
    }

        private void OnPermissionGranted(string permission)
    {
        meshManager.enabled = true;
    }

    private void OnPermissionDenied(string permission)
    {
        Debug.LogError($"Permission {Permissions.SpatialMapping} denied. Disabling script.");
        enabled = false;
    }

    public Vector3 GenerateLocalPosition(float y_range, float degree, float r_min, float r_max)
    {
        Vector3 localForward = Vector3.forward;
        const int maxAttempts = 100;
        for (int i = 0; i < maxAttempts; i++)
        {
            // Random direction in horizontal plane:
            Vector3 randomOffset = Random.insideUnitSphere * r_max;
            randomOffset.y = 0f;
            randomOffset.Normalize();

            // Check angle from localForward
            float angle = Vector3.Angle(localForward, randomOffset);
            if (angle > degree / 2f)
                continue;

            // Distance within [r_min, r_max]
            float distance = Random.Range(r_min, r_max);
            Vector3 candidateLocalPos = randomOffset * distance;

            // Y in [-y_range, +y_range]
            float chosenY = Random.Range(-y_range, y_range);
            candidateLocalPos.y = chosenY;

            // If you have no environment checks here, we simply return it:
            // or do a local "IsPositionClear" if you want
            return candidateLocalPos;
        }

        Debug.LogWarning("No valid local position found after attempts.");
        return Vector3.zero;
    }
       public Vector3 GetValidPosition_fixed(Vector3 userPosition, Vector3 markerpos, Vector3 targetSize, float y_range, float degree, float r_min, float r_max)
    {
        // 1. Generate a candidate in local space (user at (0,0,0), forward = (0,0,1)).
        Vector3 localpos = GenerateLocalPosition(y_range,degree,r_min,r_max);

        // 3. Compute the forward direction from the user to the marker (origin).
        Vector3 centerDirection = markerpos - userPosition;
        centerDirection.y = 0f;  // Project onto horizontal plane.
        centerDirection.Normalize();

        // 4. Build a coordinate basis for converting local to world space.
        Vector3 realUp = Vector3.up;
        Vector3 realRight = Vector3.Cross(realUp, centerDirection).normalized;
    
        // 5. Convert the local candidate into world space.
        Vector3 worldOffset = realRight * localpos.x + realUp * localpos.y + centerDirection * localpos.z;
        Vector3 candidatePosition = userPosition + worldOffset;
    
        // Optionally, perform world-space checks (collision, line-of-sight, etc.)
        bool valid = IsPositionClear(candidatePosition, targetSize) && HasLineOfSight(userPosition, candidatePosition);
        if (valid)
        {
            return candidatePosition;
        }
        
        Debug.LogWarning("Candidate failed world-space checks.");
        return Vector3.zero;
    }

    // public Vector3 GetValidPosition(Vector3 origin, Vector3 targetSize, float y_range, float degree, float r_min, float r_max)
    // {
    //     Vector3 userPosition = Camera.main.transform.position;
    //     Vector3 centerDirection = origin - userPosition;
    //     centerDirection.y = 0; // Project onto the horizontal plane
    //     centerDirection.Normalize();

    //     float minHeight = userPosition.y - y_range;
    //     float maxHeight = userPosition.y + y_range;

    //     const int maxAttempts = 100; // Prevent infinite loops

    //     for (int i = 0; i < maxAttempts; i++)
    //     {
    //         // Generate a random direction within the allowed angle range
    //         Vector3 randomOffset = Random.insideUnitSphere * r_max;
    //         randomOffset.y = 0; // Keep it on the horizontal plane
    //         randomOffset.Normalize();

    //         float angle = Vector3.Angle(centerDirection, randomOffset);
    //         if (angle > degree / 2) // If outside allowed field of view, retry
    //             continue;

    //         // Ensure the distance is within r_min and r_max
    //         float distance = Random.Range(r_min, r_max);
    //         Vector3 candidatePosition = userPosition + randomOffset * distance;
            
    //         float chosenY=Random.Range(minHeight, maxHeight);
    //         candidatePosition.y = chosenY;
            
    //         bool valid = IsPositionClear(candidatePosition, targetSize) && HasLineOfSight(userPosition, candidatePosition);
    //         // <<< Added: Log all the random data for this attempt >>>

    //         if (valid)
    //         {

    //             return candidatePosition;
    //         }
    //     }
    //     Debug.LogWarning("No valid position found after max attempts.");
    //     return Vector3.zero;
    // }

    // Check if the generated position is visible from the camera
    private bool HasLineOfSight(Vector3 fromPosition, Vector3 toPosition)
    {
        Vector3 direction = toPosition - fromPosition;
        float distance = direction.magnitude;
        direction.Normalize();

        // Cast a ray to check if there are obstacles in between
        if (Physics.Raycast(fromPosition, direction, out RaycastHit hit, distance, LayerMask.GetMask("MeshLayer", "Default")))
        {
            Debug.Log($"Blocked by {hit.collider.name} at {hit.point}");
            return false; // Obstacle detected, position is not in line of sight
        }

        return true; // No obstacles, position is visible
    }
    bool IsPositionClear(Vector3 position, Vector3 targetSize)
        {
            // Define the half-extents of the target (half the size of the box)
            Vector3 halfExtents = targetSize * 0.5f;

            // // Perform the box overlap check
            // if (Physics.CheckBox(position, halfExtents, Quaternion.identity, LayerMask.GetMask("Default", "MeshLayer")))
            // {
            //     Debug.Log("Overlap detected at: " + position);
            //     return false; // The position intersects with the real-world environment
            // }

            return true; // The position is clear of obstacles
        }
    private bool IsMeshingSubsystemLoaded()
    {
        if (XRGeneralSettings.Instance == null || XRGeneralSettings.Instance.Manager == null)
            return false;

        var activeLoader = XRGeneralSettings.Instance.Manager.activeLoader;
        return activeLoader != null && activeLoader.GetLoadedSubsystem<XRMeshSubsystem>() != null;
    }
}
