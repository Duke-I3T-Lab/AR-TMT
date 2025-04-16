using System.Collections.Generic;
using Unity.XR.CoreUtils;
using UnityEngine;
using UnityEngine.XR.OpenXR;
using MagicLeap.OpenXR.Features.MarkerUnderstanding;
using System.Linq;
using UnityEngine.XR.ARFoundation;


public class TargetGenerator : MonoBehaviour
{
    [Tooltip("Set the XR Origin so that the marker appears relative to headset's origin. If null, the script will try to find the component automatically.")]
    public XROrigin XROrigin;


    [Tooltip("The Prefabs.")]
    public GameObject Prefabs;
    

    [Tooltip("The Target object to be positioned upon marker detection.")]
    public GameObject TargetPrefab;
    [Tooltip("The Target object to be positioned upon marker detection.")]
    public GameObject TargetPrefab_motorspeed;

    [Tooltip("The distraction object to be positioned upon marker detection.")]
    public GameObject topdown_weak_DistractorPrefab;
    [Tooltip("The distraction object to be positioned upon marker detection.")]

    public GameObject topdwon_strongDistractorManager; // Assign in the Inspector
    public GameObject bottomup_DistractorManager; // Assign in the Inspector
    public GameObject clutter_distractormanager; // Assign in the Inspector

    [Tooltip("Generation R max.")]
    public float generation_r_max = 3f; // Default value, adjustable in the Inspector
    [Tooltip("Generation R min")]
    public float generation_r_min = 1f; // Default value, adjustable in the Inspector
    [Tooltip("Generation Angle")]
    public float generation_angle = 120f; // Default value, adjustable in the Inspector
    [Tooltip("Generation y range")]
    public float generation_y_range = 0.5f; // Default value, adjustable in the Inspector


    // Bottom-up distaroct
    [Tooltip("Bottom-Up distractor")]
    public GameObject[] BottomupDistractorPrefabs; // Array of distractor prefabs (triangles, squares, stars, blobs)
    public Material glowingMaterial; // Optional: Material for glowing effect
    // Clutter
    public GameObject[] clutterPrefabs; // Assign multiple different shapes in Inspector
    public float minAngular_clutterX = 6f; // Ensures moderate spacing for distractors
    public float minDistance_clutterX = 0.4f; // Minimum spacing to prevent clustering for targets
    public float minSpacing_clutter = 0.1f; // Ensure distractors are not overlapping to targets
    public float MaxRadius_clutter = 0.2f; // ✅ Defines how close clutter is placed to targets


    //setting  
    private List<GameObject> targets = new List<GameObject>(); // List to store targets
    private List<GameObject> distractors = new List<GameObject>(); // List to store distractors
    private HashSet<GameObject> hitTargets = new HashSet<GameObject>(); // Track which targets have been hit
    public static List<WallData> SavedWalls { get; private set; } = new List<WallData>();

    public static void SaveWalls(List<WallData> walls)
    {
        SavedWalls = new List<WallData>(walls);
    }
    public class WallData
    {
        public Vector3 Center { get; }
        public Vector3 Normal { get; }
        public float Width { get; }
        public float Height { get; }

        public WallData(Vector3 center, Vector3 normal, float width, float height)
        {
            Center = center;
            Normal = normal;
            Width = width;
            Height = height;
        }
    }
    private void Start()
        {
            // Ensure XR Origin and Target Prefab are assigned
            if (XROrigin == null)
            {
                Debug.LogError("XROrigin is not assigned. ..");
                return;
            }

            if (TargetPrefab  == null)
            {
                Debug.LogError("Target prefab is not assigned.");
                return;
            }
            
            if (topdown_weak_DistractorPrefab  == null)
            {
                Debug.LogError("Distractor prefab is not assigned.");
                return;
            }
            // Initially hide the TargetPrefab until the marker is detected
            Prefabs.SetActive(false);

            TargetPrefab.SetActive(false);
            TargetPrefab_motorspeed.SetActive(false);

            topdown_weak_DistractorPrefab.SetActive(false);
            topdwon_strongDistractorManager.SetActive(false);
            bottomup_DistractorManager.SetActive(false);
            clutter_distractormanager.SetActive(false);
        }
    public PlaceObjectAvoidingMesh placementHelper; // Reference to the PlaceObjectAvoidingMesh script
    // Track the current generation

    // Method to generate multiple targets at random positions
public void GenerateTargets(bool useWalls = false)
{
        switch (SharedInfomanager.Instance.currentGeneration)
        {
            case 1:
                Random.InitState(10);
                break;
            case 2:
                Random.InitState(20);
                break;
            case 3:
                Random.InitState(30);
                break;
            case 4:
                Random.InitState(40);
                break;
            case 5:
                Random.InitState(50);
                break;
            case 6:
                Random.InitState(60);
                break;
            case 7:
                Random.InitState(80);
                break;

        }

    SharedInfomanager.Instance.SetTargetHitSequenceByIndex(SharedInfomanager.Instance.currentGeneration - 1);

    List<object> currentSequence = SharedInfomanager.Instance.TargetHitSequence;
    int numberOfTargetsToGenerate = currentSequence?.Count ?? 0;
    float markerSize = SharedInfomanager.Instance.MarkerSize;
    Pose markerPose = SharedInfomanager.Instance.MarkerPosition;
    Vector3 targetSize = GetTargetSize(TargetPrefab);

    if (useWalls)
    {
        Debug.Log("Placing targets on saved walls...");

        List<SharedInfomanager.WallData> savedWalls = SharedInfomanager.SavedWalls;
        if (savedWalls == null || savedWalls.Count == 0)
        {
            Debug.LogError("No saved walls available for target placement!");
            return;
        }

        float minDistanceBetweenTargets = markerSize * 1.5f; // Adjust as needed to prevent overlap

        for (int i = 0; i < numberOfTargetsToGenerate; i++)
        {
            SharedInfomanager.WallData selectedWall = savedWalls[Random.Range(0, savedWalls.Count)]; // Pick a random wall

            Vector3 wallPosition = selectedWall.Center;
            Vector3 wallNormal = selectedWall.Normal;
            Vector3 wallRight = Vector3.Cross(Vector3.up, wallNormal).normalized; // Get horizontal direction
            Vector3 wallUp = Vector3.Cross(wallNormal, wallRight).normalized; // Get vertical direction

            Vector3 validPosition = Vector3.zero; // Initialize with a default value
            bool foundValidPosition = false;
            int maxAttempts = 50; // Prevent infinite loop

            for (int attempt = 0; attempt < maxAttempts; attempt++)
            {
                // Pick a random position on the wall (spread across its surface)
                float randomX = Random.Range(-selectedWall.Width / 2 + markerSize, selectedWall.Width / 2 - markerSize);
                float randomY = Random.Range(-selectedWall.Height / 2 + markerSize, selectedWall.Height / 2 - markerSize);
                validPosition = wallPosition + (wallRight * randomX) + (wallUp * randomY) + (wallNormal * 0.05f); // Offset to avoid clipping

                // Ensure new target does not overlap with previously placed targets
                bool tooClose = false;
                foreach (var existingTarget in targets)
                {
                    if (Vector3.Distance(existingTarget.transform.position, validPosition) < minDistanceBetweenTargets)
                    {
                        tooClose = true;
                        break;
                    }
                }

                if (!tooClose)
                {
                    foundValidPosition = true;
                    break;
                }
            }

            if (!foundValidPosition)
            {
                Debug.LogWarning($"Failed to find a valid position for target {i + 1}. Skipping...");
                continue;
            }

            // Instantiate and align target
            GameObject target = Instantiate(TargetPrefab, validPosition, Quaternion.identity);
            target.name = $"GeneratedTarget {currentSequence[i]}";
            target.transform.localScale = new Vector3(markerSize, target.transform.localScale.y, markerSize);

            // Ensure the front of the target is correctly facing outward
            Quaternion targetRotation = Quaternion.LookRotation(wallNormal, Vector3.up);
            target.transform.rotation = targetRotation * Quaternion.Euler(90, 0, 0); // Adjust based on prefab orientation

            target.SetActive(true);
            SharedInfomanager.Instance.AddLocation(validPosition, target.name, false);
            targets.Add(target);

            AssignTextToTarget(target, currentSequence[i]);
            Debug.Log($"Placed target {i + 1} on wall at {validPosition}");
        }
    }
    else
    {
        Debug.Log("Placing targets in free space...");

        // Pre-calculate the real-world basis using the marker (origin) and the user’s position.
        Vector3 userPos = XROrigin.Camera.transform.position;
        Vector3 centerDirection = markerPose.position - userPos;
        centerDirection.y = 0f;
        centerDirection.Normalize();
        Vector3 realUp = Vector3.up;
        Vector3 realRight = Vector3.Cross(realUp, centerDirection).normalized;


        for (int i = 0; i < numberOfTargetsToGenerate; i++)
        {
            Vector3 validPosition;
            int maxAttempts = 50;
            int attempt = 0;
            bool isTooClose;

            do
            {
                // Generate candidate in world space.

                validPosition = placementHelper.GetValidPosition_fixed(userPos, markerPose.position, targetSize, 
                    generation_y_range, generation_angle, generation_r_min, generation_r_max);
            
                // Convert the candidate from world space to local space.
                Vector3 candidateLocal = new Vector3(
                    Vector3.Dot(validPosition - userPos, realRight),
                    Vector3.Dot(validPosition - userPos, realUp),
                    Vector3.Dot(validPosition - userPos, centerDirection)
                );

                isTooClose = false;

                foreach (var existingTarget in targets)
                {
                    Vector3 existingLocal = new Vector3(
                            Vector3.Dot(existingTarget.transform.position - userPos, realRight),
                            Vector3.Dot(existingTarget.transform.position - userPos, realUp),
                            Vector3.Dot(existingTarget.transform.position - userPos, centerDirection)
                        );


                    // Check local distance and angle.
                    if (Vector3.Distance(existingLocal, candidateLocal) < minDistance_clutterX ||
                        Vector3.Angle(existingLocal.normalized, candidateLocal.normalized) < minAngular_clutterX)
                    {
                        isTooClose = true;
                        break;
                    }
                }
                attempt++;
            }
            while ((isTooClose || validPosition == Vector3.zero) && attempt < maxAttempts);

            if (!isTooClose && validPosition != Vector3.zero)
            {
                GameObject target = Instantiate(TargetPrefab, validPosition, Quaternion.identity);
                target.name = $"GeneratedTarget {currentSequence[i]}";
                target.transform.localScale = new Vector3(markerSize, target.transform.localScale.y, markerSize);
                target.transform.LookAt(XROrigin.Camera.transform.position);
                target.transform.Rotate(90, 0, 0);
                target.SetActive(true);

                SharedInfomanager.Instance.AddLocation(validPosition, target.name, false);
                targets.Add(target);

                AssignTextToTarget(target, currentSequence[i]);
                Debug.Log($"Placed target {i + 1} in free space at {validPosition}");
            }
            else
            {
                Debug.LogError($"Failed to generate target {i + 1} due to lack of valid placement positions.");
            }
        }
    }



        // neutral-baseline

        if (SharedInfomanager.Instance.currentGeneration == 3)
        {
            // Generate_neutral_Distractors(markerPose,markerSize, 20);
            GenerateTopdowndistractors(markerPose,markerSize, 20, false);

        }


        // Top-down 

        if (SharedInfomanager.Instance.currentGeneration == 4)
        {
            GenerateTopdowndistractors(markerPose,markerSize, 20, true);

        }

        // Bottom-up
        if (SharedInfomanager.Instance.currentGeneration == 5)
        {
            GenerateBottomUpDistractors(markerPose,markerSize, 20);
        }


    }

    // Helper function to assign numbers to targets
    private void AssignTextToTarget(GameObject target, object number)
    {
        TextMesh textMesh = target.GetComponentInChildren<TextMesh>();
        if (textMesh != null)
        {
            textMesh.text = number.ToString();
            float size = Mathf.Max(target.transform.localScale.x, target.transform.localScale.z);
            textMesh.characterSize = size * 0.5f;
            textMesh.fontSize = 50;
            textMesh.anchor = TextAnchor.MiddleCenter;
        }
    }

    public void GenerateTopdowndistractors(Pose markerPose, float markerSize, int numberOfDistractors, bool strong, bool cluttering=false)
    {
        int attempts = 0; // Avoid infinite loops in case of space constraints
        int maxAttempts = numberOfDistractors * 10;
        Vector3 userPos = XROrigin.Camera.transform.position;

        while (distractors.Count < numberOfDistractors && attempts < maxAttempts)
        {
            Vector3 validPosition = GetValidPosition_extra(userPos, cluttering, MaxRadius_clutter, minSpacing_clutter, markerPose, GetTargetSize(topdown_weak_DistractorPrefab));          {

                GameObject distractor = Instantiate(topdown_weak_DistractorPrefab, validPosition, Quaternion.identity);
                distractor.name = $"Distractor {distractors.Count + 1}";
                SharedInfomanager.Instance.AddLocation(validPosition, distractor.name, true);

                distractor.transform.localScale = new Vector3(markerSize, distractor.transform.localScale.y, markerSize);
                distractor.transform.LookAt(XROrigin.Camera.transform.position);
                distractor.transform.Rotate(90, 0, 0);
                distractor.SetActive(true);
                
                if (strong)
                {
                    // Find the corresponding text child in the manager
                    Transform textChild = topdwon_strongDistractorManager.transform.Find($"Resemble {distractors.Count + 1}");
                    if (textChild != null)
                    {
                        // Instantiate the text as a child of the distractor
                        Transform textInstance = Instantiate(textChild, distractor.transform);
                        textInstance.gameObject.SetActive(true); // Ensure the text is enabled
                        textInstance.localPosition = new Vector3(0, 2, 0); // Set local position (X = 0, Y = 2, Z = 0)
                        textInstance.localRotation = Quaternion.Euler(90, 180, 0); // Set local rotation (X = 90°, Y = 180°, Z = 0°)

                        // Get the TextMesh component from the instantiated text
                        TextMesh textMesh = textInstance.GetComponent<TextMesh>();
                        if (textMesh != null)
                        {
                            // Dynamically adjust the size
                            float circleDiameter = Mathf.Max(distractor.transform.localScale.x, distractor.transform.localScale.z);
                            textMesh.characterSize = circleDiameter * 0.5f;
                            textMesh.fontSize = 50;
                            textMesh.anchor = TextAnchor.MiddleCenter;

                            Debug.Log($"Set text for distractor {distractor.name}: {textMesh.text}");
                        }
                    }
                }
                // Add the target to the list
                distractors.Add(distractor);
            }
  
            attempts++;

        }
        Debug.Log($"Generated {distractors.Count} bottom-up distractors.");

    }


    void GenerateBottomUpDistractors(Pose markerPose, float markerSize, int numberOfDistractors, bool cluttering=false)
    {

        int attempts = 0; // Avoid infinite loops in case of space constraints
        int maxAttempts = numberOfDistractors * 10;
        Vector3 userPos = XROrigin.Camera.transform.position;

        while (distractors.Count < numberOfDistractors && attempts < maxAttempts)
        {
            Vector3 validPosition = GetValidPosition_extra(userPos, cluttering, MaxRadius_clutter, minSpacing_clutter, markerPose, GetTargetSize(topdown_weak_DistractorPrefab));

            if (validPosition != Vector3.zero)
            {
                // Randomly select a distractor prefab
                GameObject distractorPrefab = BottomupDistractorPrefabs[Random.Range(0, BottomupDistractorPrefabs.Length)];
                GameObject distractor = Instantiate(distractorPrefab, validPosition, Quaternion.identity);
                distractor.name = $"Distractor {distractors.Count + 1}";
                SharedInfomanager.Instance.AddLocation(validPosition, distractor.name, true);

                
                distractor.transform.LookAt(XROrigin.Camera.transform.position);
                distractor.transform.Rotate(90, 0, 0);
                distractor.transform.localScale = new Vector3(markerSize, distractor.transform.localScale.y, markerSize);

                // Randomly change color
                Renderer renderer = distractor.GetComponent<Renderer>();
                if (renderer != null)
                {
                    Color[] distractorColors = { Color.red, Color.yellow, Color.green, Color.blue, Color.magenta };
                    renderer.material.color = distractorColors[Random.Range(0, distractorColors.Length)];
                }

                // Randomly adjust size
                float sizeMultiplier = Random.Range(0.5f, 1.5f);
                distractor.transform.localScale *= sizeMultiplier;

                // Randomly add glow effect
                if (Random.Range(0, 2) == 1) // 50% chance
                {
                    distractor.GetComponent<Renderer>().material = glowingMaterial;
                }

                if (Random.Range(0, 3) == 1) // 33% chance
                {
                    distractor.AddComponent<OscillateEffect>(); // Attach movement script (must be created separately)
                }
                if (Random.Range(0, 3) == 1) // 33% chance
                {
                    distractor.AddComponent<FlickeringEffect>(); // Attach movement script (must be created separately)
                }
                if (Random.Range(0, 3) == 1) // 33% chance
                {
                    distractor.AddComponent<SpinningEffect>(); // Attach movement script (must be created separately)
                }

                distractors.Add(distractor);
                Debug.Log($"Generated distractor {distractors.Count} at {distractor.transform.position}");
            }
            attempts++;

        }

        Debug.Log($"Generated {distractors.Count} bottom-up distractors.");
    }




    public void ShuffleDistractors()
    {
        // Remove original distractors
        foreach (var distractor in distractors)
        {
            Destroy(distractor);
        }
        distractors.Clear();

        // Regenerate distractors with different locations and shapes
        GenerateBottomUpDistractors(SharedInfomanager.Instance.MarkerPosition, SharedInfomanager.Instance.MarkerSize, 20);
    
    }

    // 
    public void InitializeTargetGeneration()
    {
        // Clear existing targets
        foreach (var target in targets)
        {
            Destroy(target);
        }

        // Clear existing distractors
        foreach (var distractor in distractors)
        {
            Destroy(distractor);
        }

        // Clear the lists
        distractors.Clear();
        targets.Clear();
        hitTargets.Clear();

        // Initialization
        SharedInfomanager.Instance.ClearTaskData();
    }
    
    private Vector3 GetTargetSize(GameObject targetPrefab)
    {
        Renderer renderer = targetPrefab.GetComponentInChildren<Renderer>();
        if (renderer != null)
        {
            return renderer.bounds.size;
        }
        else
        {
            Debug.LogError("Renderer not found on TargetPrefab. Using default size.");
            return new Vector3(0.2f, 0.2f, 0.2f); // Fallback size
        }
    }


    private Vector3 GetValidPosition_extra(Vector3 userPos, bool cluttering, float clutterRadius, float minSpacing, Pose markerPose, Vector3 distractorSize)
    {
        Vector3 validPosition = Vector3.zero;
        int placementAttempts = 0;
        bool isTooClose;

        // ✅ Step 1: Pick a random anchor target if cluttering is enabled
        GameObject anchorTarget = null;
        if (cluttering && targets.Count > 0)
        {
            anchorTarget = targets[Random.Range(0, targets.Count)];
        }

        // ✅ Step 2: Try finding a valid position within constraints
        do
        {
            isTooClose = false;

            if (cluttering && anchorTarget != null)
            {
                // ✅ Generate clutter position near the chosen anchor target
                Vector3 offset = new Vector3(
                    Random.Range(-clutterRadius, clutterRadius), // X variation
                    Random.Range(-clutterRadius , clutterRadius ), // Smaller Y variation (to prevent extreme height)
                    Random.Range(-clutterRadius, clutterRadius)  // Z variation
                );
                validPosition = anchorTarget.transform.position + offset;

                // ✅ Ensure clutter does not directly overlap with a target
                foreach (var target in targets)
                {
                    if (Vector3.Distance(target.transform.position, validPosition) < minSpacing ) // Slightly more relaxed for clutter
                    {
                        isTooClose = true;
                        break;
                    }
                }

                foreach (var distractor in distractors)
                {
                    if (Vector3.Distance(distractor.transform.position, validPosition) < minSpacing ) // Slightly more relaxed for clutter
                    {
                        isTooClose = true;
                        break;
                    }
                }
            }
            else
            {
                Vector3 centerDirection = markerPose.position - userPos;
                centerDirection.y = 0f;
                centerDirection.Normalize();
                Vector3 realUp = Vector3.up;
                Vector3 realRight = Vector3.Cross(realUp, centerDirection).normalized;

                // ✅ Standard placement (non-clutter)
                validPosition = placementHelper.GetValidPosition_fixed(userPos, markerPose.position, distractorSize, generation_y_range, generation_angle, generation_r_min, generation_r_max);
                

                // Convert the candidate from world space to local space.
                Vector3 candidateLocal = new Vector3(
                    Vector3.Dot(validPosition - userPos, realRight),
                    Vector3.Dot(validPosition - userPos, realUp),
                    Vector3.Dot(validPosition - userPos, centerDirection)
                );


                foreach (var existingTarget in targets.Concat(distractors))
                {
                    Vector3 existingLocal = new Vector3(
                            Vector3.Dot(existingTarget.transform.position - userPos, realRight),
                            Vector3.Dot(existingTarget.transform.position - userPos, realUp),
                            Vector3.Dot(existingTarget.transform.position - userPos, centerDirection)
                        );


                    // Check local distance and angle.
                    if (Vector3.Distance(existingLocal, candidateLocal) < minDistance_clutterX ||
                        Vector3.Angle(existingLocal.normalized, candidateLocal.normalized) < minAngular_clutterX)
                    {
                        isTooClose = true;
                        break;
                    }
                }

            }

            placementAttempts++;
        }
        while (isTooClose && placementAttempts < 50); // Try up to 10 times

        return validPosition;
    }



}