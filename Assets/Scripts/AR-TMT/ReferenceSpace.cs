using System.Collections;
using UnityEngine;
using UnityEngine.XR;
using UnityEngine.XR.Management;
using UnityEngine.XR.OpenXR;
using MagicLeap.OpenXR.Features;

public class ReferenceSpaceToggle : MonoBehaviour
{
    private XRInputSubsystem inputSubsystem;

    IEnumerator Start()
    {
        var referenceSpaceFeature = OpenXRSettings.Instance.GetFeature<MagicLeapReferenceSpacesFeature>();
        if (!referenceSpaceFeature.enabled)
        {
            Debug.LogError("Unbounded Tracking Space cannot be set if the OpenXR Magic Leap Reference Spaces Feature is not enabled. Stopping Script.");
            yield break;
        }

        yield return new WaitUntil(() => XRGeneralSettings.Instance != null &&
                                          XRGeneralSettings.Instance.Manager != null &&
                                          XRGeneralSettings.Instance.Manager.activeLoader != null &&
                                          XRGeneralSettings.Instance.Manager.activeLoader.GetLoadedSubsystem<XRInputSubsystem>() != null);

        inputSubsystem = XRGeneralSettings.Instance.Manager.activeLoader.GetLoadedSubsystem<XRInputSubsystem>();
        // Set the tracking origin to Unbounded
        SetSpace(TrackingOriginModeFlags.Unbounded);
    }

    private void SetSpace(TrackingOriginModeFlags flag)
    {
        if (inputSubsystem.TrySetTrackingOriginMode(flag))
        {
            Debug.Log($"Current Space: {inputSubsystem.GetTrackingOriginMode()}");
            inputSubsystem.TryRecenter();
        }
        else
        {
            Debug.LogError($"SetSpace failed to set Tracking Mode Origin to {flag}");
        }
    }
}


