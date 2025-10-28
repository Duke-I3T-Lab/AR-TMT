using UnityEngine;
using UnityEngine.UI;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.XR;
using UnityEngine.XR.Interaction.Toolkit;
using System.Collections;
using UnityEngine.XR.MagicLeap;
using System.Linq; // Required for Contains on arrays

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
    
    public bool calibrating = false;

    public float desiredDistance = 1f;         // desired distance (in meters) from the camera
    public float angleThreshold = 30f;           // if the angle between camera forward and canvas > 30 degrees, relocate
    public float distanceThreshold = 0.3f;       // allowable deviation from desiredDistance
    private int[] validGenerations;

    void Start()
    {
        if (noticeUI != null)
        {
            noticeUI.SetActive(false); // Disable the entire Notice UI at the start
            Debug.Log("Notice UI parent object is set  to inactive..");
        }
        positionInputAction.Enable();
        rotationInputAction.Enable();
        triggerInputAction.Enable();

        triggerInputAction.performed += ActionOnPerformed;
        triggerInputAction.canceled += ActionOnCanceled;

       // Find the camera script in the scene
        cameraScript = FindObjectOfType<TestCameraRecording_CVcamera>();
        planedetectionscript = FindObjectOfType<PlaneDetectionMarker>();

        validGenerations = new[] 
        {
            SharedInfomanager.Instance.wall_tmtA,
            SharedInfomanager.Instance.wall_tmtB,
            SharedInfomanager.Instance.wall_tmtA_nuetral,
            SharedInfomanager.Instance.wall_tmtA_topdown,
            SharedInfomanager.Instance.wall_tmtA_bottomup
        };
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

                    
                    // Check if the currentGeneration is in the valid cases
                    if (validGenerations.Contains(SharedInfomanager.Instance.currentGeneration))
                    {
                        if (SharedInfomanager.Instance.wall_calibrated == false)
                        {
                            return;
                        }
                    }
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
                    
        else if(validGenerations.Contains(SharedInfomanager.Instance.currentGeneration))
        {
            if(calibrating)
            {
            planedetectionscript.DestroyGeneratedMarkersAndWalls();
            // cameraScript.StartCameraFromExternalFlag();
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

        SharedInfomanager.Instance.initializeUIposition(noticeUI, 0.0051f);

       
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
        if (validGenerations.Contains(SharedInfomanager.Instance.currentGeneration))
        {
            if (SharedInfomanager.SavedWalls == null || SharedInfomanager.SavedWalls.Count == 0)
            {
                Debug.Log("No saved walls found. Starting camera and plane detection.");
                calibrating=true;
                // StopCamera();
                planedetectionscript.StartPlaneDetection();
                rescanbutton.SetActive(true);

            }
            else
            {
                rescanbutton.SetActive(false);

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
        SharedInfomanager.Instance.UpdateUIposition(noticeUI);

    }





}