using System;
using MagicLeap.Android;
using System.Collections;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
using MagicLeap.OpenXR.Features.Planes;
using MagicLeap.OpenXR.Subsystems;
using UnityEngine.XR.Management;

public class PlaneDetectionSLAM : MonoBehaviour
{
    private MagicLeapPlanesFeature planeFeature;

    private ARPlaneManager _planeManager;

    [SerializeField, Tooltip("Maximum number of planes to return each query")]
    private uint maxResults = 100;

    [SerializeField, Tooltip("Minimum plane area to treat as a valid plane")]
    private float minPlaneArea = 0.25f;

    private Camera _camera;
    private bool permissionGranted = false;

    private IEnumerator Start()
    {
        _camera = Camera.main;
        yield return new WaitUntil(AreSubsystemsLoaded);
        _planeManager = FindObjectOfType<ARPlaneManager>();
        if (_planeManager == null)
        {
            Debug.LogError("Failed to find ARPlaneManager in scene. Disabling Script");
            enabled = false;
        }
        else
        {
            // disable planeManager until we have successfully requested required permissions
            _planeManager.enabled = false;
        }

        permissionGranted = false;

        // Request Spatial Mapping Permission
        Permissions.RequestPermission( Permissions.SpatialMapping,
            OnPermissionGranted,OnPermissionDenied,OnPermissionDenied);
    }

    private bool AreSubsystemsLoaded()
    {
        if (XRGeneralSettings.Instance == null) return false;
        if (XRGeneralSettings.Instance.Manager == null) return false;
        var activeLoader = XRGeneralSettings.Instance.Manager.activeLoader;
        if (activeLoader == null) return false;
        return activeLoader.GetLoadedSubsystem<XRPlaneSubsystem>() != null;
    }

    private void Update()
    {
        UpdateQuery();
    }

    private void UpdateQuery()
    {
        if (_planeManager != null && _planeManager.enabled && permissionGranted)
        {
      
            var newQuery = new MLXrPlaneSubsystem.PlanesQuery
            {
                Flags = _planeManager.requestedDetectionMode.ToMLXrQueryFlags() | MLXrPlaneSubsystem.MLPlanesQueryFlags.SemanticAll,
                BoundsCenter = _camera.transform.position,
                BoundsRotation = _camera.transform.rotation,
                BoundsExtents = Vector3.one * 20f,
                MaxResults = maxResults,
                MinPlaneArea = minPlaneArea
            };

            MLXrPlaneSubsystem.Query = newQuery;
        }
    }

    // Dispose of existing planes and stop scanning
    public void OnDisable()
    {
        if (planeFeature != null && planeFeature.enabled)
        {
            planeFeature.InvalidateCurrentPlanes();
        }
    }

    private void OnDestroy()
    {
        if(_planeManager)
            _planeManager.enabled = false;
    }

    private void OnPermissionGranted(string permission)
    {
        _planeManager.enabled = true;
        permissionGranted = true;
    }

    private void OnPermissionDenied(string permission)
    {
        Debug.LogError($"Failed to create Planes Subsystem due to missing or denied {Permissions.SpatialMapping} permission. Please add to manifest. Disabling script.");
        enabled = false;
    }
}