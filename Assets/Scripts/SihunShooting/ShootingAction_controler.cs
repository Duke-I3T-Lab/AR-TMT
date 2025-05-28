using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.XR;
using UnityEngine.UIElements;
using UnityEngine.XR.Interaction.Toolkit.Inputs;
using UnityEngine.XR.Interaction.Toolkit;
using System.Collections;
using System.Collections.Generic;

public class ShootingAction_controler : MonoBehaviour
{
    public TargetGenerator targetGenerator; // Assign the Marker object in the Inspector
    public NoticeHandler noticeUI; // Reference to the notice UI panel`
    public EyeTrackerLogger eyeTrackerLogger;

    [SerializeField]
    private InputAction positionInputAction =
        new InputAction(binding: "<MagicLeapController>/pointerPosition", expectedControlType: "Vector3");

    [SerializeField]
    private InputAction rotationInputAction =
        new InputAction(binding: "<MagicLeapController>/pointerRotation", expectedControlType: "Quaternion");
   
    [SerializeField]
    private InputAction triggerInputAction =
        new InputAction(binding: "<XRController>/trigger", expectedControlType: "Button");
    
    private float detectionRange = 20f; // Maximum range for the raycast

    private bool hasTriggered = false;
    private GameObject lastHighlightedTarget = null;

    public AudioClip soundClip;
    private AudioSource audioSource;



    private void Start()
    {
        // Enable input actions
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
    private void ActionOnPerformed(InputAction.CallbackContext obj)
    {
        if (!SharedInfomanager.Instance.IsTaskActive)
        {
            Debug.Log("Task is not active. Ignoring target/distractor logic.");
            return;
        }
        
        
        if (!hasTriggered)
        {
            hasTriggered = true;

            // Get the pointer position and rotation from the input actions
            Vector3 rayOrigin = positionInputAction.ReadValue<Vector3>();
            Quaternion rayRotation = rotationInputAction.ReadValue<Quaternion>();
            Vector3 rayDirection = rayRotation * Vector3.forward; // Forward direction of the controller

            float time = Time.time;

            string hittype = "Miss";  // Default to "Miss"
            string label = "Unknown"; // Make sure label is properly assigned
            float distanceToCenter = -1f; // Default invalid distance
            string hitResult = "-";  // New variable for tracking the result


            // Make sound effect
            audioSource.PlayOneShot(soundClip,1.0f);
            SharedInfomanager.Instance.shootingdata.Add(new SharedInfomanager.ShootingData(hittype, label, distanceToCenter, time, hitResult));

            // Perform a raycast from the pointer position
            if (Physics.Raycast(rayOrigin, rayDirection, out RaycastHit hit, detectionRange))
            {
                Debug.Log($"Hit object: {hit.collider.gameObject.name}, Hit point: {hit.point}");
                Vector3 objectCenter = hit.collider.bounds.center;
                distanceToCenter = Vector3.Distance(objectCenter, hit.point);
                label = hit.collider.gameObject.name; 



                // Check if the hit object is tagged as "Target"
                if (hit.collider.CompareTag("Target"))
                {
                    Debug.Log("Target detected!");
                    hittype = "Target";
                    hitResult = TargetHit(hit.collider.gameObject);
                }
                //  distractor
                else if(hit.collider.CompareTag("Distractor"))
                {
                    Debug.Log("Distractor hit!");
                    hittype="Distractor";

                    SharedInfomanager.Instance.Incrementhitdistractor();
                    // Get the Renderer of the distractor and change its color temporarily
                    Renderer frameRenderer = hit.collider.GetComponentInChildren<Renderer>();
                    if (frameRenderer != null)
                    {
                        StartCoroutine(ChangeColorTemporarily(frameRenderer, Color.red, 1f));
                    }
                }

            }
            else
            {
                Debug.Log("No object detected in the ray's path.");
                hittype="Miss";
                (label, distanceToCenter)=FindClosestObjectToLine(rayOrigin, rayDirection);
                SharedInfomanager.Instance.IncrementMissHit();

            }
            
        }
    }
    // private void Update()
    // {

    // }

    public string TargetHit(GameObject target)
    {

        string extractedName = target.name.Replace("GeneratedTarget ", "");
        object targetIdentifier = int.TryParse(extractedName, out int targetNumber) ? targetNumber : extractedName;
        object expectedTarget = SharedInfomanager.Instance.GetNextExpectedTarget();

        if (!expectedTarget.Equals(targetIdentifier))
        {
            Debug.LogWarning($"Target {target.name} hit out of order. Expected Target: {expectedTarget}.");
            SharedInfomanager.Instance.IncrementWrongorder();
            UpdateTargetFrameColor(target, Color.red, 1f);
            return "WrongOrder"; 
        }

        Debug.Log($"Target hit in order: {target.name}");

        // If a previous target was marked as the latest (yellow), update it to green.
        if (lastHighlightedTarget != null)
        {
            UpdateTargetFrameColor(lastHighlightedTarget, Color.green);
        }
        // Mark the current target as the latest hit (yellow).
        UpdateTargetFrameColor(target, Color.yellow);

        SharedInfomanager.Instance.AdvanceToNextTarget();
        
        
        // // Shuffle the distractors for bottom-up attention
        // if (SharedInfomanager.Instance.currentGeneration == 5)
        // {
        //     string expectedTargetStr = expectedTarget.ToString();
        //     Debug.Log($"Expected Target (as string): {expectedTargetStr}");

        //     if (expectedTargetStr.Equals("8") || expectedTargetStr.Equals("16"))
        //     {
        //         Debug.Log($"Expected Target matched for shuffling");
        //         targetGenerator.ShuffleDistractors();
        //     }
  
        // }

        // 3) Store reference for next time
        lastHighlightedTarget = target;

        // Check if all targets are hit
        if (SharedInfomanager.Instance.GetNextExpectedTarget() == null)
        {
            Debug.Log("All targets hit in sequence!");
            SharedInfomanager.Instance.FinishTask();
            
        }
        return "CorrectTarget";
    }

    // Helper method to update the target's frame color
    private void UpdateTargetFrameColor(GameObject target, Color color, float revertAfter = 0f)
    {
        Renderer frameRenderer = target.GetComponentInChildren<Renderer>();
        if (frameRenderer == null) return;

        if (revertAfter > 0)
        {
            StartCoroutine(ChangeColorTemporarily(frameRenderer, color, revertAfter));
        }
        else
        {
            frameRenderer.material.color = color;
        }
    }    
    private void OnDestroy()
    {
        triggerInputAction.Dispose();
        positionInputAction.Dispose();
        rotationInputAction.Dispose();
    }
    private void ActionOnCanceled(InputAction.CallbackContext obj)
    {
        hasTriggered = false; // Reset the trigger state when the button is released
    }

    private IEnumerator ChangeColorTemporarily(Renderer renderer, Color newColor, float duration)
    {
        // Store the original color
        Color originalColor = renderer.material.color;

        // Change to the new color (red)
        renderer.material.color = newColor;

        // Wait for the specified duration (1 second)
        yield return new WaitForSeconds(duration);

        // Revert to the original color
        renderer.material.color = originalColor;
    }
    public (string, float) FindClosestObjectToLine(Vector3 rayOrigin, Vector3 rayDirection)
    {
        if (SharedInfomanager.Instance.TargetLocations == null ||SharedInfomanager.Instance.DistractorLocations == null)
        {
            Debug.LogWarning("TargetLocations or DistractorLocations is not assigned!");
            return ("None", float.MaxValue);
        }

        // Define the line from the controller
        Vector3 lineStart = rayOrigin;
        Vector3 lineEnd = lineStart + rayDirection * 10f; // Extend 10 units forward

        // Initialize closest object variables
        string closestObjectName = "None";
        float minDistance = float.MaxValue;

        // Combine TargetLocations and DistractorLocations
        List<SharedInfomanager.LocationData> allObjects = new List<SharedInfomanager.LocationData>();
        allObjects.AddRange(SharedInfomanager.Instance.TargetLocations);
        allObjects.AddRange(SharedInfomanager.Instance.DistractorLocations);

        // Iterate through all objects and find the closest one
        foreach (var obj in allObjects)
        {
            Vector3 objectPos = obj.GetVector3();
            Vector3 closestPoint = ClosestPointOnLine(lineStart, lineEnd, objectPos);
            float distance = Vector3.Distance(closestPoint, objectPos);

            if (distance < minDistance)
            {
                minDistance = distance;
                closestObjectName = obj.Label;
            }
        }

        // Print and return results
        if (closestObjectName != "None")
        {
            Debug.Log($"Closest Object: {closestObjectName}, Distance: {minDistance}");
        }
        else
        {
            Debug.Log("No objects found.");
        }

        return (closestObjectName, minDistance);
    }
    // Returns the closest point on a line segment to a given point
    private Vector3 ClosestPointOnLine(Vector3 A, Vector3 B, Vector3 P)
    {
        Vector3 AB = B - A;
        Vector3 AP = P - A;
        float t = Vector3.Dot(AP, AB) / Vector3.Dot(AB, AB);
        t = Mathf.Clamp01(t); // Ensures the closest point is within the segment
        return A + t * AB;
    }


}