using UnityEngine;

public class ChoicesControl : MonoBehaviour
{
    [Header("Prefab & References")]
    public GameObject choicePrefab;         
    public GameObject questionnaireControl; // Where to send OnSelectEvent

    [Header("Number of Choices")]
    public int numChoices = 7;

    [Header("Spacing & Sizing")]
    [Tooltip("Fraction of parent width to use for gap between each child.")]
    public float spacingFraction = 0.02f;
    [Header("Coloring ")]
    public Gradient gradient;
    
    void Start()
    {
        RectTransform parentRect = GetComponent<RectTransform>();
        float totalWidth = parentRect.rect.width;

        // Gaps
        float gap = totalWidth * spacingFraction;
        float totalGaps = gap * (numChoices - 1);

        // Remaining space for planes
        float planeSpace = totalWidth - totalGaps;
        float singlePlaneWidth = planeSpace / numChoices;

        // If plane is 10x10, scale factor:
        float planeMeshSize = 10f; // or 1f if it's a Quad
        float scaledPlaneWidth = singlePlaneWidth / planeMeshSize;

        // If pivot = (0.5,0.5), the left edge is -totalWidth/2
        float startX = -totalWidth / 2f;

        for (int i = 0; i < numChoices; i++)
        {
            GameObject choice = Instantiate(choicePrefab, transform);
            // choice.transform.SetParent(questionnaireCanvas.transform, false);

            float xPos = startX + i * (singlePlaneWidth + gap) + singlePlaneWidth / 2f;
            choice.transform.localPosition = new Vector3(xPos, 0f, -0.02f);

            // Scale in X by 'scaledPlaneWidth'
            Vector3 s = choice.transform.localScale;
            choice.transform.localScale = new Vector3(scaledPlaneWidth, s.y, s.z);

            choice.transform.localEulerAngles = new Vector3(-90, 0, 0);

            // Hook up event
            var button = choice.GetComponent<ButtonControl>();
            if (button != null)
            {
                button.order = i+1;
                button.OnSelectEvent.AddListener(
                    questionnaireControl.GetComponent<QuestionnaireControl>().OnQuestionAnswered
                );
            }
        }
    }
}
