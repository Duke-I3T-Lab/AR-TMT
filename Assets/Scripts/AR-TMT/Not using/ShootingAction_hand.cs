using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.XR;
using UnityEngine.UIElements;
using UnityEngine.XR.Interaction.Toolkit.Inputs;
using System.Collections;

public class ShootingAction_hand : MonoBehaviour
{

    public GameObject targetGenerator; // Assign the Marker object in the Inspector

    [SerializeField]
    protected Hands hand;
    public GameObject bulletPrefab;           // Assign the Bullet prefab in the Inspector
    public float bulletSpeed = 50f;           // Speed of the bullet

    public enum Hands
    {
        Left,
        Right
    }
    
    [SerializeField]
    private float detectionRange = 20f; // Maximum range for the raycast
    [SerializeField]

    protected InputActionMap actionMap;
    private ShootingAiming_hand shootingAiming; // Reference to ShootingAiming script initialized at runtime

    private bool hasgripped = false;
    private int nextExpectedTarget = 1; // Tracks the next target number to hit
    private bool hasTriggered = false;

    private void Start()
    {
        shootingAiming = GetComponent<ShootingAiming_hand>();

        // Automatically find the ShootingAiming script on the same object if not assigned in the Inspector
        if (shootingAiming == null)
        {
            Debug.LogError("ShootingAiming component not found on the same object.");
            return;
        }

        var ism = FindObjectOfType<InputActionManager>();

        var mapAsset = ism.actionAssets[0];

        actionMap = hand switch
        {
            Hands.Left => mapAsset.FindActionMap("LeftHand"),
            Hands.Right => mapAsset.FindActionMap("RightHand"),
            _ => throw new System.NotImplementedException()
        };

    }

    private void Update()
    {
 
        if (shootingAiming == null) return;

        var pose = actionMap.FindAction("Grip").ReadValue<PoseState>();
        float value = actionMap.FindAction("GraspValue").ReadValue<float>();
        bool ready = actionMap.FindAction("GraspReady").ReadValue<float>() > 0f;

        
        // Detect Grip Gesture
        if (value > 0.9f && !hasgripped)
        {
            Vector3 rayOrigin = shootingAiming.GetRayOrigin();
            Vector3 rayDirection = shootingAiming.GetRayDirection();
            // Debug.Log($"Grip detected, Ray origin: {rayOrigin}, Ray Direction: {rayDirection}");
            hasgripped = true;
            if (Physics.Raycast(rayOrigin, rayDirection, out RaycastHit hit, detectionRange))
            {
                Debug.Log($"Hit object name is: {hit.collider.gameObject.name}, Ray hit: {hit.collider.name} at position {hit.point}");

                // Check if the hit object is the target
                if (hit.collider.CompareTag("Target")) 
                {
                    Debug.Log("Target detected!");
  
                    // Check if the hit object is tagged as "Target"
                    if (hit.collider.CompareTag("Target"))
                    {
                        Debug.Log("Target detected!");
                        TargetHit(hit.collider.gameObject);
                    }
                }
                
                else
                {
                    Debug.Log("No object detected in the ray's path.");
                }

            }
        }
        else if (value <= .9f)
        {
            hasgripped = false;
        }
    }

   public void TargetHit(GameObject target)
    {

        string extractedName = target.name.Replace("GeneratedTarget ", "");
        object targetIdentifier = int.TryParse(extractedName, out int targetNumber) ? targetNumber : extractedName;

        object expectedTarget = SharedInfomanager.Instance.GetNextExpectedTarget();

        if (!expectedTarget.Equals(targetIdentifier))
        {
            Debug.LogWarning($"Target {target.name} hit out of order. Expected Target: {expectedTarget}.");
            SharedInfomanager.Instance.IncrementWrongorder();
            UpdateTargetFrameColor(target, Color.red, 1f);
            return;
        }


        Debug.Log($"Target hit in order: {target.name}");
        UpdateTargetFrameColor(target, Color.green);
        SharedInfomanager.Instance.AdvanceToNextTarget();
        
        // Check if all targets are hit
        if (SharedInfomanager.Instance.GetNextExpectedTarget() == null)
        {
            Debug.Log("All targets hit in sequence!");
            // SharedInfomanager.Instance.LogFinalData();
            targetGenerator.GetComponent<TargetGenerator>().GenerateTargets();
            // add Show Notice UI
            // if (noticeUI != null)
            // {
            //     noticeUI.SetActive(true); // Activate the Notice UI
            //     Debug.Log("Notice UI displayed.");
            // }

        }
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


    private IEnumerator SpawnBullet(Vector3 startPoint, Vector3 endPoint)
    {
        Debug.Log($"Spawning bullet at: {startPoint}");

        // Instantiate the bullet prefab at the starting point
        GameObject bullet = Instantiate(bulletPrefab, startPoint, Quaternion.identity);

        if (bullet == null)
        {
            Debug.LogError("Bullet instantiation failed! Ensure bulletPrefab is assigned in the Inspector.");
            yield break;
        }

        // Get the direction to the end point
        Vector3 direction = (endPoint - startPoint).normalized;
        float distance = Vector3.Distance(startPoint, endPoint);
        float travelTime = distance / bulletSpeed;

        float elapsedTime = 0f;
        while (elapsedTime < travelTime)
        {

            bullet.transform.position += direction * bulletSpeed * Time.deltaTime;
            elapsedTime += Time.deltaTime;
            yield return null;
        }

        Debug.Log("Bullet reached destination.");


    }


}

