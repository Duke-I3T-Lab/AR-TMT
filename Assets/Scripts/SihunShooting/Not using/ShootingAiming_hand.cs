using UnityEngine;
using System.Collections.Generic;

public class ShootingAiming_hand : MonoBehaviour
{
    [SerializeField]
    private Transform palmTransform; // Assign this to your palm joint transform

    [SerializeField]
    private float rayLength = 0.5f; // Length of the ray

    [SerializeField]
    private float rayAngleOffset = 30f; // Angle offset in degrees

    private LineRenderer lineRenderer;
    private Vector3 currentRayOrigin;
    private Vector3 currentRayDirection;
    private GameObject lastHitTarget = null; // Keeps track of the last hit target
    private Dictionary<GameObject, Color> originalColors = new Dictionary<GameObject, Color>(); // Stores original colors

    private void Start()
    {
        // Get or add the LineRenderer component
        lineRenderer = GetComponent<LineRenderer>();
        if (lineRenderer == null)
        {
            lineRenderer = gameObject.AddComponent<LineRenderer>();
        }

        // Configure the LineRenderer
        lineRenderer.positionCount = 2; // Start and end points
        lineRenderer.startWidth = 0.01f; // Adjust thickness
        lineRenderer.endWidth = 0.01f;
        lineRenderer.material = new Material(Shader.Find("Unlit/Color"));
        lineRenderer.material.color = Color.red;
    }

    private void Update()
    {
        if (palmTransform != null)
        {
            // Get palm position and rotation
            Vector3 palmPosition = palmTransform.position;
            Quaternion palmRotation = palmTransform.rotation;

            // Adjust the ray's direction by applying a rotation offset
            Quaternion rotationOffset = Quaternion.Euler(rayAngleOffset, 0, 0); 
            Vector3 rayDirection = palmRotation * rotationOffset * Vector3.forward;

            // Calculate the end position of the ray
            Vector3 rayEndPosition = palmPosition + rayDirection * rayLength;

            // Set the positions for the LineRenderer
            lineRenderer.SetPosition(0, palmPosition);      // Start at palm position
            lineRenderer.SetPosition(1, rayEndPosition);    // End in the ray direction
            currentRayOrigin = palmPosition;
            currentRayDirection = rayDirection;
            
            // Check for hits
            PerformRaycast(palmPosition, rayDirection);
        }
    }
    // Method to perform raycast
    // Method to perform raycast
    private void PerformRaycast(Vector3 origin, Vector3 direction)
    {
        Ray ray = new Ray(origin, direction);
        if (Physics.Raycast(ray, out RaycastHit hitInfo, rayLength))
        {
            GameObject hitObject = hitInfo.collider.gameObject;

            // Change the color of the target if it has a Renderer
            Renderer targetRenderer = hitObject.GetComponent<Renderer>();
            if (targetRenderer != null)
            {
                // Check if the target is already red
                if (targetRenderer.material.color == Color.red)
                {
                    Debug.Log($"Ray hit: {hitObject.name}, but it's already red. No change.");
                    return; // Skip further processing
                }

                // Store the original color if not already stored
                if (!originalColors.ContainsKey(hitObject))
                {
                    originalColors[hitObject] = targetRenderer.material.color;
                }

                // Change color of the newly hit target
                if (lastHitTarget != hitObject)
                {
                    // Reset the color of the previously hit target
                    ResetLastHitTargetColor();

                    // Change the color of the current target
                    targetRenderer.material.color = Color.green; // Set to green
                    lastHitTarget = hitObject; // Update the last hit target
                    Debug.Log($"Ray hit: {hitObject.name}, color changed to green.");
                }
            }
        }
        else
        {
            // No target hit; reset the last hit target's color
            ResetLastHitTargetColor();
        }
    }

    // Method to reset the last hit target's color
    // Method to reset the last hit target's color
private void ResetLastHitTargetColor()
{
    if (lastHitTarget != null)
    {
        Renderer lastTargetRenderer = lastHitTarget.GetComponent<Renderer>();
        if (lastTargetRenderer != null && originalColors.ContainsKey(lastHitTarget))
        {
            // Check if the current color is red
            if (lastTargetRenderer.material.color != Color.red)
            {
                // Reset to the original color only if it's not red
                lastTargetRenderer.material.color = originalColors[lastHitTarget];
                Debug.Log($"Ray no longer hitting: {lastHitTarget.name}, color reset to original.");
            }
            else
            {
                Debug.Log($"Ray no longer hitting: {lastHitTarget.name}, but it remains red.");
            }
        }
        lastHitTarget = null; // Clear the last hit target reference
    }
}


    // Public methods to provide ray information
    public Vector3 GetRayOrigin() => currentRayOrigin;
    public Vector3 GetRayDirection() => currentRayDirection;
}
