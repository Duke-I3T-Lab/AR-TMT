using System;
using System.Collections.Generic;
using System.IO;
using TMPro;
using UnityEngine;
using UnityEngine.InputSystem;

public class QuestionnaireControl : MonoBehaviour
{
    // Start is called before the first frame update
    public List<string> headers;
    public List<string> questions;
    private int currentIndex = -1;
    public TMP_Text HeaderTextbox;
    public TMP_Text QuestionTextbox;
    private bool finished = false;

    public GameObject choices;
    public GameObject questionnaireCanvas;

    private MagicLeapInputs _magicLeapInputs;
    private MagicLeapInputs.ControllerActions _controllerActions;
    public GameObject lowTextBox;
    public GameObject highTextBox;

    public GameObject backwardButton;

    // Thresholds for relocating the canvas
    public float desiredDistance = 1f;         // desired distance (in meters) from the camera
    public float angleThreshold = 30f;           // if the angle between camera forward and canvas > 30 degrees, relocate
    public float distanceThreshold = 0.3f;       // allowable deviation from desiredDistance

    public bool IsFinished
    {
        get {return finished;}
    }

    private StreamWriter csvWriter;
    private const string csvHeader = "QuestionIndex, QuestionText, Answer, Timestamp";
    void Start()
    {
        questionnaireCanvas.SetActive(false);

    }
    public void StartSurvey()
    {
        Debug.Log("StartSurvey() was called...");

        currentIndex=-1;
        finished=false;
        
        string filepath = SharedInfomanager.Instance.GenerateUniqueFilePath("Survey_task", SharedInfomanager.Instance.currentGeneration, "csv");
        csvWriter = new StreamWriter(filepath);
        csvWriter.WriteLine(csvHeader);

        _magicLeapInputs = new MagicLeapInputs();
        _magicLeapInputs.Enable();
        _controllerActions = new MagicLeapInputs.ControllerActions(_magicLeapInputs);
        _controllerActions.Trigger.performed += HandleOnTrigger;

        // Reset UI
        questionnaireCanvas.SetActive(true);
        questionnaireCanvas.transform.SetParent(null); // ensure it's not parented to the camera

        SharedInfomanager.Instance.initializeUIposition(questionnaireCanvas, 0.05f);

        // Make sure buttons and choices are properly visible
        backwardButton.SetActive(false);
        NextQuestion();      


    }
    private void HandleOnTrigger(InputAction.CallbackContext obj)
    {
        float triggerValue = obj.ReadValue<float>();
        if (triggerValue > 0.5f && currentIndex == -1)
        {
            NextQuestion();
            choices.SetActive(true);


        }
    }


    // Update is called once per frame
    public void NextQuestion()
    {
        if (currentIndex < questions.Count - 1)
        {
            currentIndex++;
            HeaderTextbox.SetText(headers[currentIndex]);
            QuestionTextbox.SetText(questions[currentIndex]);

            // Show the backward button if we’re past the first question
            if (currentIndex > 0)
            {
                backwardButton.SetActive(true);
            }
        }
        else
        {
            EndSurvey();
        }
    }

    public void OnQuestionAnswered(int answer)
    {
        Debug.Log("Question " + currentIndex + " answered with " + answer);
        string row = (currentIndex + 1) + "," + questions[currentIndex] + "," + answer + "," + DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
        csvWriter.WriteLine(row);
        if (!finished) NextQuestion();
    }

    public void OnBackward()
    {
        if (currentIndex > 0)
        {
            currentIndex--;
            HeaderTextbox.SetText(headers[currentIndex]);
            QuestionTextbox.SetText(questions[currentIndex]);
            if (currentIndex == 0)
            {
                backwardButton.SetActive(false);
            }
        }
    }

    private void OnDestroy()
    {
        // Ensure resources are cleaned up even if object is destroyed mid-survey
        EndSurvey();
    }

    /// <summary>
    /// Ends the survey by closing the CSV, removing input callbacks,
    /// and optionally hiding the survey canvas.
    /// </summary>
    public void EndSurvey()
    {
        // If we’ve already finished, or never started, just skip
        if (finished && csvWriter == null) return;

        finished = true;

        // Close the file if it's still open
        if (csvWriter != null)
        {
            csvWriter.Close();
            csvWriter = null;
        }

        // Unsubscribe from input
        if (_magicLeapInputs != null)
        {
            _controllerActions.Trigger.performed -= HandleOnTrigger;
        }

        // Dispose input if allocated
        if (_magicLeapInputs != null)
        {
            _magicLeapInputs.Dispose();
            _magicLeapInputs = null;
        }

        // Optionally hide the survey UI
        if (questionnaireCanvas != null)
        {
            questionnaireCanvas.SetActive(false);
        }
    }

    // Update is called once per frame
    void Update()
    {
        SharedInfomanager.Instance.UpdateUIposition(questionnaireCanvas);

    }


}
