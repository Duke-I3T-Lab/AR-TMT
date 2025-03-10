using UnityEngine;
using UnityEngine.UI;
using UnityEngine.InputSystem;
using UnityEngine.XR.Interaction.Toolkit;
using System.Collections.Generic;
using TMPro;

using System.Collections;
public class SelectionNoticeHandler : MonoBehaviour
{
    public GameObject SelectionUI; // Reference to the notice UI panel
    public NoticeHandler noticeUI;
    public Button button_motorspeed; // Separate reference for the motor speed button
    
    [SerializeField]
    public GameObject descriptionObject;

    private List<Button> stageButtons = new List<Button>(); // Stores all stage buttons dynamically
    private InputAction triggerInputAction =
        new InputAction(binding: "<XRController>/trigger", expectedControlType: "Button");

    [SerializeField]
    private InputAction positionInputAction =
        new InputAction(binding: "<MagicLeapController>/pointerPosition", expectedControlType: "Vector3");

    [SerializeField]
    private InputAction rotationInputAction =
        new InputAction(binding: "<MagicLeapController>/pointerRotation", expectedControlType: "Quaternion");

    private bool hasTriggered = false;
    public float desiredDistance = 1f;         // desired distance (in meters) from the camera
    public float angleThreshold = 30f;           // if the angle between camera forward and canvas > 30 degrees, relocate
    public float distanceThreshold = 0.3f;       // allowable deviation from desiredDistance

    void Start()
    {
        if (SelectionUI != null)
        {
            SelectionUI.SetActive(false); // Hide UI initially
            Debug.Log("Notice UI parent object is set to inactive.");
        }

        // ✅ Automatically find all buttons inside SelectionUI
        RegisterButtons();

        // Enable Input Actions
        positionInputAction.Enable();
        rotationInputAction.Enable();
        triggerInputAction.Enable();

        triggerInputAction.performed += ActionOnPerformed;
        triggerInputAction.canceled += ActionOnCanceled;
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
        triggerInputAction.Enable();
        triggerInputAction.performed += OnTriggerPressed;
    }

    void OnDisable()
    {
        triggerInputAction.performed -= OnTriggerPressed;
        triggerInputAction.Disable();
    }

    // ✅ Automatically detect and register all buttons
    private void RegisterButtons()
    {
        if (SelectionUI != null)
        {
            stageButtons.Clear(); // Clear old list in case of UI refresh
            Button[] buttons = SelectionUI.GetComponentsInChildren<Button>(true); // Find all buttons inside SelectionUI

            foreach (Button button in buttons)
            {
                if (button != button_motorspeed) // Exclude motor speed button from stage buttons
                {
                    stageButtons.Add(button);
                }
            }
        }
    }

    private void ActionOnPerformed(InputAction.CallbackContext obj)
    {
        if (SharedInfomanager.Instance.IsTaskActive || SharedInfomanager.Instance.IsMotorTestActive)
        {
            Debug.Log("Task is already active. Ignoring start button press.");
            return;
        }

        if (!hasTriggered)
        {
            hasTriggered = true;

            Vector3 rayOrigin = positionInputAction.ReadValue<Vector3>();
            Quaternion rayRotation = rotationInputAction.ReadValue<Quaternion>();
            Vector3 rayDirection = rayRotation * Vector3.forward;
            Debug.Log($"Trigger pressed! Ray origin: {rayOrigin}, Ray direction: {rayDirection}");
            LayerMask layerMask = LayerMask.GetMask("UI"); // Ensure "UI" is assigned to the button

            if (Physics.Raycast(rayOrigin, rayDirection, out RaycastHit hit, 10f, layerMask))
            {
                Debug.Log($"Raycast hit: {hit.collider.gameObject.name}");

                if (hit.collider != null)
                {
                    Button pressedButton = hit.collider.gameObject.GetComponent<Button>();

                    if (pressedButton != null)
                    {
                        if (pressedButton == button_motorspeed)
                        {
                            Debug.Log($"Motor Speed Test Button pressed!");
                            SharedInfomanager.Instance.currentGeneration = 0;
                        }
                        else
                        {
                            int buttonIndex = GetButtonIndex(pressedButton);
                            if (buttonIndex != -1)
                            {
                                Debug.Log($"Button {buttonIndex} pressed!");
                                SharedInfomanager.Instance.currentGeneration = buttonIndex;
                            }
                        }
                        noticeUI.noticegeneration();
                    }

                    if (SelectionUI != null)
                    {
                        SelectionUI.SetActive(false); // Hide Notice UI
                        Debug.Log("Notice UI parent object is set to inactive.");
                    }
                }
            }
        }
    }

    // ✅ Dynamically Get Button Index Based on Name
    private int GetButtonIndex(Button pressedButton)
    {
        int index = stageButtons.IndexOf(pressedButton);
        return index != -1 ? index + 1 : -1; // Convert 0-based index to 1-based
    }

    private void OnTriggerPressed(InputAction.CallbackContext context)
    {
        foreach (Button button in stageButtons)
        {
            button.onClick.Invoke();
        }

        if (button_motorspeed != null)
        {
            button_motorspeed.onClick.Invoke();
        }

        Debug.Log("Start button clicked via trigger.");
    }

    private void ActionOnCanceled(InputAction.CallbackContext obj)
    {
        hasTriggered = false;
    }

    // ✅ Displays UI and Enables All Buttons Automatically
    public void selection_noticegeneration()
    {
        
        SelectionUI.SetActive(true);
        Transform camTransform = Camera.main.transform;
        // Set Notice UI position, rotation, and scale
        SelectionUI.transform.position = camTransform.position + camTransform.forward * desiredDistance;
        SelectionUI.transform.rotation = Quaternion.LookRotation(SelectionUI.transform.position - camTransform.position); // Align rotation to face the camera
        SelectionUI.transform.localScale = Vector3.one * 0.0051f; // Set a consistent scale



        if (SelectionUI != null)
        {
            SelectionUI.SetActive(true); // Activate UI
            Debug.Log("Notice UI displayed.");
            UpdateDescriptionText("Select the stage after refresh time: 30s");

            // Start the countdown coroutine
            StartCoroutine(CountdownTimer(30));
        }


        RegisterButtons(); // Ensure all buttons are registered

        foreach (Button button in stageButtons)
        {
            button.gameObject.SetActive(true);
        }

        if (button_motorspeed != null) button_motorspeed.gameObject.SetActive(true);

        Debug.Log("All stage buttons are set to active.");
    }

    // Coroutine for the countdown timer
    private IEnumerator CountdownTimer(int seconds)
    {
        int remainingTime = seconds;
        while (remainingTime > 0)
        {
            
            UpdateDescriptionText("Select the stage to start after refresh time: " + remainingTime + "s");
            yield return new WaitForSeconds(1f);
            remainingTime--;
        }

        UpdateDescriptionText("Select the stage to start");

    }
    public void UpdateDescriptionText(string newText)
    {
        // Make sure the Description object is assigned
        if (descriptionObject != null)
        {
            // Find the TextMeshProUGUI component in the children of Description
            TextMeshProUGUI tmpText = descriptionObject.GetComponentInChildren<TextMeshProUGUI>();
            if (tmpText != null)
            {
                tmpText.text = newText;
            }
        }
    }
    void Update()
    {
        // Only reposition if the canvas is active (survey started)
        if (!SelectionUI.activeSelf) return;

        Transform camTransform = Camera.main.transform;
        Vector3 camPos = camTransform.position;
        Vector3 camForward = camTransform.forward;

        // Compute the direction and distance from the camera to the canvas
        Vector3 directionToCanvas = SelectionUI.transform.position - camPos;
        float currentDistance = directionToCanvas.magnitude;
        directionToCanvas.Normalize();

        // Compute the angle between camera's forward direction and the direction to the canvas
        float angle = Vector3.Angle(camForward, directionToCanvas);

        // Check if the canvas is out of view (angle too large) or at the wrong distance
        if (angle > angleThreshold || Mathf.Abs(currentDistance - desiredDistance) > distanceThreshold)
        {
            // Relocate the canvas in front of the camera
            Vector3 newPos = camPos + camForward * desiredDistance;
            SelectionUI.transform.position = newPos;
            SelectionUI.transform.rotation = Quaternion.LookRotation(newPos - camPos);
        }
    }

}
