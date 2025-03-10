using UnityEngine;
using UnityEngine.UI;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.XR;
using UnityEngine.XR.Interaction.Toolkit;
using System.Collections;
using UnityEngine.XR.MagicLeap;

public class NoticeHandler : MonoBehaviour
{
    public GameObject noticeUI; // Reference to the notice UI panel`
    public Button okButton; // Reference to the OK button
    public GameObject rescanbutton; // Reference to the OK button


    public MotorSpeedTest motorspeedtest;


    [SerializeField]
    private InputAction triggerInputAction =
        new InputAction(binding: "<XRController>/trigger", expectedControlType: "Button");
        [SerializeField]
    private InputAction positionInputAction =
        new InputAction(binding: "<MagicLeapController>/pointerPosition", expectedControlType: "Vector3");

    [SerializeField]
    private InputAction rotationInputAction =
        new InputAction(binding: "<MagicLeapController>/pointerRotation", expectedControlType: "Quaternion");
   
    private bool isPointerOverButton = false; // Tracks if the pointer is over the button
    private bool hasTriggered = false;
    private TestCameraRecording_CVcamera cameraScript;
    private PlaneDetectionMarker planedetectionscript;
    
    private bool calibrating = false;

    public float desiredDistance = 1f;         // desired distance (in meters) from the camera
    public float angleThreshold = 30f;           // if the angle between camera forward and canvas > 30 degrees, relocate
    public float distanceThreshold = 0.3f;       // allowable deviation from desiredDistance


    void Start()
    {
        if (noticeUI != null)
        {
            noticeUI.SetActive(false); // Disable the entire Notice UI at the start
            Debug.Log("Notice UI parent object is set  to inactive.");
        }
        positionInputAction.Enable();
        rotationInputAction.Enable();
        triggerInputAction.Enable();

        triggerInputAction.performed += ActionOnPerformed;
        triggerInputAction.canceled += ActionOnCanceled;

       // Find the camera script in the scene
        cameraScript = FindObjectOfType<TestCameraRecording_CVcamera>();
        planedetectionscript = FindObjectOfType<PlaneDetectionMarker>();

    }

    private void OnDestroy()
    {
        // Dispose input actions
        triggerInputAction.Dispose();
        positionInputAction.Dispose();
        rotationInputAction.Dispose();
    }

    void OnEnable()
    {
        // Enable the trigger input action
        triggerInputAction.Enable();
        triggerInputAction.performed += OnTriggerPressed;
    }    

    void OnDisable()
    {
        // Disable the trigger input action
        triggerInputAction.performed -= OnTriggerPressed;
        triggerInputAction.Disable();
    }

    private void ActionOnPerformed(InputAction.CallbackContext obj)
    {    
        if (SharedInfomanager.Instance.IsTaskActive || SharedInfomanager.Instance.IsMotorTestActive)
        {
            Debug.Log("Task is not active. Ignoring start button press.");
            return;
        }
         
        if (!hasTriggered)
        {
            hasTriggered = true;

            // Get the pointer position and rotation from the input actions
            Vector3 rayOrigin = positionInputAction.ReadValue<Vector3>();
            Quaternion rayRotation = rotationInputAction.ReadValue<Quaternion>();
            Vector3 rayDirection = rayRotation * Vector3.forward; // Forward direction of the controller
            Debug.Log($"Trigger pressed! Ray origin: {rayOrigin}, Ray direction: {rayDirection}");
            LayerMask layerMask = LayerMask.GetMask("UI"); // Ensure "UI" is assigned to the button

            // Perform a raycast from the pointer position
            if (Physics.Raycast(rayOrigin, rayDirection, out RaycastHit hit, 10f, layerMask))
            {
                Debug.Log($"Raycast hit: {hit.collider.gameObject.name}");
                if (hit.collider != null && hit.collider.gameObject == okButton.gameObject)
                {
                    Debug.Log("start button pressed!");
                    OnOkButtonClicked();
                }
                else if (hit.collider != null && hit.collider.gameObject == rescanbutton.gameObject)
                {
                    Debug.Log("re-scan button pressed!");
                    planedetectionscript.OnRescanButtonClicked();
                }

            }
        }
    }

    private void OnTriggerPressed(InputAction.CallbackContext context)
    {
        if (isPointerOverButton && okButton != null)
        {
            // Simulate button click
            okButton.onClick.Invoke();
            Debug.Log("Start button clicked via trigger.");
        }
    }
    private void ActionOnCanceled(InputAction.CallbackContext obj)
    {
        hasTriggered = false; // Reset the trigger state when the button is released
    }

    void OnOkButtonClicked()
    {
        if (noticeUI != null)
        {
            noticeUI.SetActive(false); // Disable the entire Notice UI at the start
            Debug.Log("Notice UI parent object is set to inactive.");
        }

        if (SharedInfomanager.Instance.currentGeneration == 0)
        {
            motorspeedtest.StartMotorSpeedTest();
        }
        // Start the delayed task
        else if(SharedInfomanager.Instance.currentGeneration == SharedInfomanager.Instance.wall_tmtA || SharedInfomanager.Instance.currentGeneration == SharedInfomanager.Instance.wall_tmtB)
        {
            if(calibrating)
            {
            planedetectionscript.DestroyGeneratedMarkersAndWalls();
            cameraScript.StartCameraFromExternalFlag();
            calibrating=false;
            }

            SharedInfomanager.Instance.StartTaskWithDelay();  
        }
        else
        {
            SharedInfomanager.Instance.StartTaskWithDelay();  
        }


    }

    public void noticegeneration()
    {
        if (noticeUI == null)
        {
            Debug.LogError(" Notice UI is NULL! Cannot display.");
            return;
        }
        noticeUI.SetActive(true); // Activate the Notice UI

        // Set Notice UI position, rotation, and scale
        Transform camTransform = Camera.main.transform;


        noticeUI.transform.position =camTransform.position + camTransform.forward * desiredDistance;
        noticeUI.transform.rotation = Quaternion.LookRotation(noticeUI.transform.position - camTransform.position); // Align rotation to face the camera
        noticeUI.transform.localScale = Vector3.one * 0.0051f; // Set a consistent scale

        string panelName;
        if (SharedInfomanager.Instance.currentGeneration == 0)
        {
            panelName = "MotorSpeedTest";
        }
        else
        {
            panelName = $"Stage{SharedInfomanager.Instance.currentGeneration}";
        }

        Transform activePanel = noticeUI.transform.Find(panelName); // Find the panel by name
        Transform button = noticeUI.transform.Find("Button");

        // Activate the current generation panel and the button
        if (activePanel != null) activePanel.gameObject.SetActive(true);
        if (button != null) button.gameObject.SetActive(true);

        // Disable other panels
        for (int i = 0; i < noticeUI.transform.childCount; i++)
        {
            Transform child = noticeUI.transform.GetChild(i);
            if (child != activePanel && child != button)
            {
                child.gameObject.SetActive(false); // Deactivate other children
            }
        }
        Debug.Log($"Notice UI displayed at position: {noticeUI.transform.position}");
        

        // For plane situated task, disable cvcamera to enable marker detection
        if (SharedInfomanager.Instance.currentGeneration == SharedInfomanager.Instance.wall_tmtA || SharedInfomanager.Instance.currentGeneration == SharedInfomanager.Instance.wall_tmtB)
        {
            if (SharedInfomanager.SavedWalls == null || SharedInfomanager.SavedWalls.Count == 0)
            {
                calibrating=true;
                StopCamera();
                planedetectionscript.StartPlaneDetection();
                rescanbutton.SetActive(true);

            }

        }    
    }
    public void StopCamera()
    {
        if (cameraScript != null)
        {
            Debug.Log("Calling DisconnectCameraAsync() from external script...");
            _ = cameraScript.DisconnectCameraAsync(); // Call the async method
        }
        else
        {
            Debug.LogError("TestCameraRecording_CVcamera not found!");
        }
    }


    void Update()
    {
        // Only reposition if the canvas is active (survey started)
        if (!noticeUI.activeSelf) return;

        Transform camTransform = Camera.main.transform;
        Vector3 camPos = camTransform.position;
        Vector3 camForward = camTransform.forward;

        // Compute the direction and distance from the camera to the canvas
        Vector3 directionToCanvas = noticeUI.transform.position - camPos;
        float currentDistance = directionToCanvas.magnitude;
        directionToCanvas.Normalize();

        // Compute the angle between camera's forward direction and the direction to the canvas
        float angle = Vector3.Angle(camForward, directionToCanvas);

        // Check if the canvas is out of view (angle too large) or at the wrong distance
        if (angle > angleThreshold || Mathf.Abs(currentDistance - desiredDistance) > distanceThreshold)
        {
            // Relocate the canvas in front of the camera
            Vector3 newPos = camPos + camForward * desiredDistance;
            noticeUI.transform.position = newPos;
            noticeUI.transform.rotation = Quaternion.LookRotation(newPos - camPos);
        }
    }






}