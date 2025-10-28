using UnityEngine;

public class OscillateEffect : MonoBehaviour
{
    public float floatAmplitude = 0.05f; // Vertical movement range
    public float floatSpeed = 2.0f; // Speed of oscillation


    private Vector3 startPosition;
    private float phaseOffset; // Unique phase shift per object
        private int oscillationAxis; // 0 = X, 1 = Y, 2 = Z


    void Start()
    {
        startPosition = transform.position;
        phaseOffset = Random.Range(0f, Mathf.PI * 2);
                // Randomly select an axis: 0 = X, 1 = Y, 2 = Z
        oscillationAxis = Random.Range(0, 3);
    }
    void Update()
    {
        float offset = Mathf.Sin(Time.time * floatSpeed + phaseOffset) * floatAmplitude;
        float jitter = (Mathf.PerlinNoise(Time.time * 10f, phaseOffset) - 0.5f) * 0.01f;

        Vector3 newPosition = startPosition;

        if (oscillationAxis == 0) newPosition.x += offset + jitter;
        else if (oscillationAxis == 1) newPosition.y += offset + jitter;
        else if (oscillationAxis == 2) newPosition.z += offset + jitter;

        transform.position = newPosition;
    }
}